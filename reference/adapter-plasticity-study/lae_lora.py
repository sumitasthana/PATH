"""
Continual-learning model for every preset. With the lae_baseline preset it is LAE's LoRA baseline (Gao et al., ICCV 2023, "A Unified Continual Learning Framework with General
Parameter-Efficient Tuning") reproduced inside Mammoth: `main.py --config=vit_lora ... module.num_emas=0
module.num_freeze_epochs=0` in the LAE repo.

Training step = LAE main.py train_step: cross-entropy on the current task's logits only
(output[:, n_old:n_seen], target - n_old). Optimizer = Adam over head + adapter, re-created at the start of
every task (Mammoth reloads it in meta_begin_task, LAE in Module.setup per task). No EMA, no freeze epochs.

Every result file carries startup checks (class-order hash, per-class train counts, trainable-parameter count,
adapter identity at init, local-CE gradient mask) and the environment. Any failed check aborts the run.

Results: <ap_out>/<ap_run_name>.json after every task; per-task class-IL accuracies on the CIFAR-100 test
split (or, with --validation v, on v% of each class held out from the training set: eval_split says which), A_t = mean over seen tasks, A_T (last) and A_bar (mean of A_t over t), as reported by LAE.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel


def add_lae_args(parser):
    g = parser.add_argument_group('LAE baseline')
    g.add_argument('--ap_out', type=str, default='ap_results')
    g.add_argument('--ap_run_name', type=str, default=None)
    g.add_argument('--ap_preset', type=str, default='unspecified')
    g.add_argument('--ap_class_order', type=int, default=1, choices=[1, 2, 3])
    return parser


class CheckFailed(RuntimeError):
    pass


def _check(rec, name, ok, detail):
    rec['checks'][name] = dict(ok=bool(ok), detail=detail)
    if not ok:
        raise CheckFailed(f'{name}: {detail}')


def _write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(tmp, path)


class LaeLora(ContinualModel):
    """LAE LoRA baseline: frozen ViT-B/16 + KVLoRA, local cross-entropy, fresh Adam per task."""
    NAME = 'lae-lora'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        add_lae_args(parser)
        return parser

    # ------------------------------------------------------------------ training step
    def _offsets(self):
        if getattr(self.args, 'joint', 0):
            return 0, self.N_CLASSES
        return self.n_past_classes, self.n_seen_classes

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        past, seen = self._offsets()
        self.opt.zero_grad()
        out = self.net(inputs)[:, past:seen]
        loss = F.cross_entropy(out, labels - past)
        loss.backward()
        if not self._grad_checked:
            g = self.net.head.weight.grad
            outside = torch.cat([g[:past], g[seen:]]).abs().max().item() if (past > 0 or seen < g.shape[0]) else 0.0
            inside = g[past:seen].abs().max().item()
            _check(self._rec, f'local_ce_grad_mask_task{self.current_task + 1}', outside == 0.0 and inside > 0.0,
                   dict(max_abs_grad_outside=outside, max_abs_grad_inside=inside, rows=[past, seen]))
            lab_ok = bool(((labels >= past) & (labels < seen)).all())
            _check(self._rec, f'labels_in_task_range_task{self.current_task + 1}', lab_ok,
                   dict(min=int(labels.min()), max=int(labels.max()), range=[past, seen]))
            self._grad_checked = True
            self._save()
        self.opt.step()
        return loss.item()

    # ------------------------------------------------------------------ bookkeeping
    def _path(self):
        os.makedirs(self.args.ap_out, exist_ok=True)
        name = self.args.ap_run_name or f'lae_order{self.args.ap_class_order}_s{self.args.seed}'
        return os.path.join(self.args.ap_out, name + '.json')

    def _save(self):
        self._rec['elapsed_min'] = (time.time() - self._rec['started']) / 60
        _write_json(self._path(), self._rec)

    @torch.no_grad()
    def _rel_dW(self):
        """||W_eff - W0||_F / ||W0||_F pooled over every adapted matrix (same definition for all adapters)."""
        num = den = 0.0
        n = 0
        for d, W0 in self.net.weight_deltas():
            num += d.pow(2).sum().item()
            den += W0.pow(2).sum().item()
            n += 1
        self._n_adapted_matrices = n
        return float(np.sqrt(num / den)) if den > 0 else float('nan')

    def _init_record(self, dataset):
        from datasets.seq_cifar100_lae import LAE_ORDERS, order_sha256, mammoth_class_order
        import timm
        a = self.args
        keep = ['model', 'dataset', 'backbone', 'ap_adapter', 'ap_blocks', 'ap_targets', 'ap_lae_rank', 'ap_lora_r',
                'ap_lora_alpha', 'ap_oft_block_size', 'ap_timm_model', 'lr', 'optimizer',
                'optim_wd', 'n_epochs', 'batch_size', 'drop_last', 'seed', 'joint', 'validation', 'ap_class_order',
                'ap_preset', 'permute_classes']
        cfg = {k: getattr(a, k, None) for k in keep}
        env = dict(python=sys.version.split()[0], torch=torch.__version__, timm=timm.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                   platform=platform.platform())
        self._rec = dict(config=cfg, env=env, checks={}, tasks=[], complete=False, started=time.time())
        rec = self._rec
        order = LAE_ORDERS[a.ap_class_order]
        rec['class_order_sha256'] = order_sha256(order)
        _check(rec, 'class_order_mapping', np.array_equal(np.asarray(a.class_order), mammoth_class_order(a.ap_class_order)),
               'args.class_order == argsort(LAE order)')
        _check(rec, 'optimizer_adam', a.optimizer.lower() == 'adam', a.optimizer)
        _check(rec, 'no_weight_decay', float(a.optim_wd) == 0.0, a.optim_wd)
        n_ad = sum(p.numel() for p in self.net.adapter_parameters())
        expect = self.net.expected_params
        _check(rec, 'adapter_param_count', n_ad == expect, dict(found=n_ad, expected=expect, adapter=self.net.adapter))
        n_frozen_trainable = sum(p.numel() for n, p in self.net.base_named_parameters() if p.requires_grad)
        _check(rec, 'backbone_frozen', n_frozen_trainable == 0, n_frozen_trainable)
        dw0 = self._rel_dW()
        n_expect = len(self.net.blocks_adapted) * len(self.net.targets)
        _check(rec, 'adapted_matrix_count', self._n_adapted_matrices == n_expect,
               dict(found=self._n_adapted_matrices, expected=n_expect))
        _check(rec, 'adapter_identity_at_init', dw0 <= 1e-12, dict(rel_dW=dw0))
        hw = self.net.head.weight
        _check(rec, 'head_init', abs(hw.std().item() - 0.001) < 2e-4 and self.net.head.bias.abs().max().item() == 0.0,
               dict(weight_std=hw.std().item()))
        rec['n_trainable_adapter'] = n_ad
        rec['eval_split'] = 'validation (held out from train)' if getattr(a, 'validation', None) else 'test'
        rec['n_tasks'] = dataset.N_TASKS
        import hashlib
        w = torch.cat([p.detach().flatten().cpu().float() for n, p in self.net.base_named_parameters()])
        rec['backbone_weights_sha256'] = hashlib.sha256(w.numpy().tobytes()).hexdigest()
        rec['backbone_pretrained_cfg'] = {k: str(v) for k, v in dict(self.net.pretrained_cfg).items()
                                          if k in ('hf_hub_id', 'tag', 'architecture', 'url', 'file')}
        rec['backbone_n_params'] = int(w.numel())

    def begin_task(self, dataset):
        if self.current_task == 0:
            self._init_record(dataset)
        self._grad_checked = False
        t = self.current_task + 1
        targets = np.asarray(dataset.train_loader.dataset.targets)
        past, seen = self._offsets()
        counts = np.bincount(targets, minlength=self.N_CLASSES)
        fake = os.environ.get('AP_FAKE_DATA') == '1'
        v = float(getattr(self.args, 'validation', 0) or 0)
        want = None if fake else 500 - int(np.ceil(500 * v / 100))    # Mammoth holds out ceil(v% of 500) per class
        ok = set(np.nonzero(counts)[0].tolist()) == set(range(past, seen)) and (want is None or (counts[past:seen] == want).all())
        _check(self._rec, f'train_class_counts_task{t}', ok,
               dict(classes=np.nonzero(counts)[0].tolist(), counts=counts[past:seen].tolist()))
        self._save()

    def end_task(self, dataset):
        from utils.evaluate import evaluate
        t = self.current_task + 1
        cil, til = evaluate(self, dataset)
        row = dict(t=t, test_cil=cil, test_til=til, A_t=float(np.mean(cil)), rel_dW=self._rel_dW(),
                   scales=[float(p.item()) for n, p in self.net.body.named_parameters()
                           if self.net.adapter == 'lae_kvlora' and 'scale' in n])
        self._rec['tasks'].append(row)
        if t == dataset.N_TASKS:
            self._rec['A_T'] = row['A_t']
            self._rec['A_bar'] = float(np.mean([r['A_t'] for r in self._rec['tasks']]))
            self._rec['complete'] = True
        self._save()
