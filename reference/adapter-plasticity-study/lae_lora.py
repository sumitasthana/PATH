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
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from models.ap_utils.plasticity import (ProbeRunner, RNGGuard, add_probe_args, check_data, data_manifest,
                                        load_probe_data, probe_points, tensors_sha)
from models.ap_utils.stream_metrics import StreamMetrics, add_metric_args


def add_lae_args(parser):
    g = parser.add_argument_group('LAE baseline')
    g.add_argument('--ap_out', type=str, default='ap_results')
    g.add_argument('--ap_run_name', type=str, default=None)
    g.add_argument('--ap_preset', type=str, default='unspecified')
    g.add_argument('--ap_class_order', type=int, default=1, choices=[1, 2, 3])
    g.add_argument('--ap_t0_cache', type=str, default=None,
                   help='Directory for cached t = 0 probe results, keyed by the exact t = 0 model (default <ap_out>/t0_cache).')
    g.add_argument('--ap_head_lr', type=float, default=None,
                   help='Learning rate of the linear head; default: the same as --lr (one Adam over head + adapter, '
                        'the calibrated protocol). Set to decouple head and adapter learning rates.')
    g.add_argument('--ap_joint_points', type=int, default=0,
                   help='Joint (single-task) training only: number of evenly spaced measurement points over the run. '
                        'Point k (k = 1..K) is taken after round(k * S / K) of the S updates and is recorded exactly '
                        'like the end of task k in the continual runs (accuracy, dW, stream metrics, probes at '
                        '--ap_probe_points, checkpoint). 0 = one point at the end.')
    g.add_argument('--ap_ckpt_dir', type=str, default='',
                   help='If set, save every trainable tensor (adapter + head) at t = 0 and after every task to '
                        '<dir>/t<tt>.pt, so any probe or metric can be recomputed later. Empty = off.')
    add_probe_args(parser)
    add_metric_args(parser)
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

    def get_optimizer(self, params=None, lr=None):
        """Calibrated protocol (no --ap_head_lr): Mammoth's optimizer over every parameter at --lr, unchanged.
        With --ap_head_lr: Adam with two groups, head at --ap_head_lr and everything else at --lr."""
        hlr = getattr(self.args, 'ap_head_lr', None)
        if hlr is None or params is not None:
            return super().get_optimizer(params=params, lr=lr)
        head = {id(p) for p in self.net.head.parameters()}
        rest = [p for p in self.net.parameters() if id(p) not in head]
        return torch.optim.Adam([{'params': rest, 'lr': self.args.lr}, {'params': list(self.net.head.parameters()), 'lr': hlr}],
                                lr=self.args.lr, weight_decay=self.args.optim_wd)

    # ------------------------------------------------------------------ training step
    def _offsets(self):
        if getattr(self.args, 'joint', 0):
            return 0, self.N_CLASSES
        return self.n_past_classes, self.n_seen_classes

    def offsets_for_task(self, k):
        if getattr(self.args, 'joint', 0):
            return 0, self.N_CLASSES
        return self.get_offsets(k)

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
        if getattr(self, '_metrics', None) is not None:
            self._step += 1
            self._metrics.after_step(inputs, labels, past, seen, loss.item(), self._step)
        if getattr(self, '_jp_at', None):
            self._jp_step += 1
            k = self._jp_at.get(self._jp_step)
            if k is not None:
                self._record_point(self._dataset, k, final=False)
                self._save()
        return loss.item()

    # ------------------------------------------------------------------ bookkeeping
    def _path(self):
        os.makedirs(self.args.ap_out, exist_ok=True)
        name = self.args.ap_run_name or f'lae_order{self.args.ap_class_order}_s{self.args.seed}'
        return os.path.join(self.args.ap_out, name + '.json')

    def _ckpt(self, t):
        """Write the trainable tensors (adapter + head) at point t. Read-only on the model: training is unchanged."""
        d = getattr(self.args, 'ap_ckpt_dir', '')
        if not d:
            return None
        os.makedirs(d, exist_ok=True)
        state = {n: p.detach().cpu().clone() for n, p in self.net.named_parameters() if p.requires_grad}
        path = os.path.join(d, f't{t:02d}.pt')
        torch.save(dict(t=t, state=state, adapter=self.net.adapter,
                        backbone_weights_sha256=self._rec.get('backbone_weights_sha256')), path + '.tmp')
        os.replace(path + '.tmp', path)
        return dict(file=os.path.basename(path), adapter_sha256=tensors_sha(self.net.adapter_parameters())[:32])

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
                'ap_preset', 'permute_classes', 'ap_joint_points', 'ap_head_lr']
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
        if self.net.adapter == 'full':
            n_body = sum(p.numel() for p in self.net.body.parameters())
            _check(rec, 'full_model_trainable', n_ad == n_body, dict(trainable=n_ad, backbone=n_body))
        else:
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
        rec['optimizer_groups'] = [dict(lr=g['lr'], n_params=sum(p.numel() for p in g['params'] if p.requires_grad))
                                   for g in self.opt.param_groups]
        rec['eval_split'] = 'validation (held out from train)' if getattr(a, 'validation', None) else 'test'
        rec['n_tasks'] = dataset.N_TASKS
        import hashlib
        base = self.net.body.named_parameters() if self.net.adapter == 'full' else self.net.base_named_parameters()
        w = torch.cat([p.detach().flatten().cpu().float() for n, p in base])      # pretrained weights at init
        rec['backbone_weights_sha256'] = hashlib.sha256(w.numpy().tobytes()).hexdigest()
        rec['backbone_pretrained_cfg'] = {k: str(v) for k, v in dict(self.net.pretrained_cfg).items()
                                          if k in ('hf_hub_id', 'tag', 'architecture', 'url', 'file')}
        rec['backbone_n_params'] = int(w.numel())

    # ------------------------------------------------------------------ plasticity probes
    def _probe_cfg(self):
        keys = ['ap_probes', 'ap_probe_lrs', 'ap_probe_head_lrs', 'ap_probe_head_lr', 'ap_probe_steps_nat',
                'ap_probe_steps_rand', 'ap_rand_n', 'ap_probe_bs', 'ap_probe_seed', 'ap_probe_reps', 'ap_probe_reps_t0']
        return {k: getattr(self.args, k) for k in keys}

    def _probe_runner(self):
        if not hasattr(self, '_runner'):
            data = load_probe_data(self.args.ap_data_dir)
            ok, detail = check_data(data)
            _check(self._rec, 'probe_data_sizes', ok, detail)
            self._rec['probe_data'] = data_manifest(data)
            self._runner = ProbeRunner(self.args, self.net, self.device, data)
        return self._runner

    def _probe(self, t):
        """Fork-and-discard measurement at point t; the stream model must be unchanged afterwards."""
        reps = self.args.ap_probe_reps_t0 if t == 0 else self.args.ap_probe_reps
        runner = self._probe_runner()
        cache = None
        if t == 0:
            key = hashlib.sha256(json.dumps(dict(adapter=tensors_sha(self.net.adapter_parameters()),
                                                 backbone=self._rec['backbone_weights_sha256'],
                                                 cfg=self._probe_cfg()), sort_keys=True).encode()).hexdigest()[:20]
            d = self.args.ap_t0_cache or os.path.join(self.args.ap_out, 't0_cache')
            os.makedirs(d, exist_ok=True)
            cache = os.path.join(d, f't0_{key}.json')
            if os.path.exists(cache):
                self._rec['probes']['0'] = dict(json.load(open(cache)), cached=True)
                return
        res, before, after = runner.measure(reps)
        _check(self._rec, f'probe_discarded_t{t}', before == after, dict(adapter_sha_before=before[:16], adapter_sha_after=after[:16]))
        res['cfg'] = self._probe_cfg()
        self._rec['probes'][str(t)] = res
        if cache:
            _write_json(cache, res)

    def begin_task(self, dataset):
        if self.current_task == 0:
            self._init_record(dataset)
            self._rec['probes'] = {}
            self._rec['probe_points'] = sorted(probe_points(self.args))
            self._metrics, self._step = (StreamMetrics(self), 0) if self.args.ap_metrics else (None, 0)
            if self._metrics is not None:
                self._rec['metrics_t0'] = self._metrics.measure(dataset, 0)
            if 0 in probe_points(self.args):
                self._probe(0)
            ck = self._ckpt(0)
            if ck:
                self._rec['ckpt_t0'] = ck
        self._grad_checked = False
        self._jp_at = None
        K = getattr(self.args, 'ap_joint_points', 0)
        if K:
            if not getattr(self.args, 'joint', 0):
                raise CheckFailed('--ap_joint_points requires --joint 1')
            S = len(dataset.train_loader) * int(self.args.n_epochs)
            self._jp_at = {int(round(k * S / K)): k for k in range(1, K)}     # point K = end of training
            self._jp_step, self._dataset = 0, dataset
            self._rec['joint_points'] = dict(K=K, total_updates=S, steps=sorted(self._jp_at))
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

    def _record_point(self, dataset, t, final):
        """One measurement point: accuracy, dW, stream metrics, checkpoint, probes at --ap_probe_points."""
        from utils.evaluate import evaluate
        if final:
            cil, til = evaluate(self, dataset)
        else:                                    # inside a task: leave every random-number state untouched
            with RNGGuard():
                cil, til = evaluate(self, dataset)
        row = dict(t=t, test_cil=cil, test_til=til, A_t=float(np.mean(cil)), rel_dW=self._rel_dW(),
                   scales=[float(p.item()) for n, p in self.net.body.named_parameters()
                           if self.net.adapter == 'lae_kvlora' and 'scale' in n],
                   adapter_sha256=tensors_sha(self.net.adapter_parameters())[:32])
        if self.net.adapter == 'full':
            row['rel_dW_all'] = self.net.rel_dW_all()
        if self._metrics is not None:
            row['loss_progress'] = self._metrics.take_loss_progress()
            row['metrics'] = self._metrics.measure(dataset, t)
        ck = self._ckpt(t)
        if ck:
            row['ckpt'] = ck
        self._rec['tasks'].append(row)
        if t in probe_points(self.args):
            self._save()
            self._probe(t)
        return row

    def end_task(self, dataset):
        joint_k = getattr(self.args, 'ap_joint_points', 0) if getattr(self.args, 'joint', 0) else 0
        t = joint_k if joint_k else self.current_task + 1
        row = self._record_point(dataset, t, final=True)
        if self.current_task + 1 == dataset.N_TASKS:
            self._rec['A_T'] = row['A_t']
            self._rec['A_bar'] = float(np.mean([r['A_t'] for r in self._rec['tasks']]))
            self._rec['complete'] = True
        self._save()
