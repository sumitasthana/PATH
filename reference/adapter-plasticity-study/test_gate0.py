"""
Gate 0: CPU test suite. Runs at install time; the installer aborts if any test fails.
Run from the Mammoth root:  AP_FAKE_DATA=1 python tests/test_gate0.py

Covers the parts of the LAE-baseline pipeline that can silently produce wrong numbers:
  - KVLoRA: identity at init, Q rows never change, K/V rows = W + s B A, parameter count;
  - backbone: orig_in21k ViT-B/16 has no pre-logits and pools the CLS token after the final norm;
  - class orders: LAE orders are permutations, Mammoth mapping puts LAE position i at label i;
  - transforms: LAE train/test pipelines, bilinear, no normalization;
  - local cross-entropy: equals CE over the current task's logits, zero gradient on all other head rows,
    and a fresh Adam step leaves those rows bit-identical;
  - PEFT LoRA / OFT on the study placement: parameter counts on real ViT-B/16 shapes, identity at init,
    OFT rotation orthogonal and equal to W R^T, LoRA equal to W + (alpha/r) B A;
  - end-to-end fake runs (LAE KVLoRA, LoRA, OFT, validation split, 20 tasks): every check passes;
  - training equivalence: run side by side with the calibrated code version in this environment, training is
    bit-identical (adapter SHA-256 and accuracy after every task), stream metrics off and on;
  - metric math: Hessian power iteration on a quadratic, effective rank and srank on known matrices;
  - plasticity probes: a run with probes is bit-identical to the run without (adapter SHA-256 and accuracy after
    every task), the t = 0 cache reproduces, every random-number state is restored.
"""
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

os.environ['AP_FAKE_DATA'] = '1'
sys.argv = ['x']

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn as nn                                # noqa: E402
import torch.nn.functional as F                      # noqa: E402
import torchvision.transforms as T                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestKVLoRA(unittest.TestCase):
    def setUp(self):
        from backbone.vit_lae import KVLoRAQKV
        torch.manual_seed(0)
        self.d, self.r = 16, 5
        self.base = nn.Linear(self.d, 3 * self.d)
        self.m = KVLoRAQKV(self.base, self.r)
        self.x = torch.randn(4, 7, self.d)

    def test_identity_at_init(self):
        self.assertTrue(torch.equal(self.m(self.x), self.base(self.x)))

    def test_q_untouched_kv_exact(self):
        with torch.no_grad():
            for B in self.m.lora_B:
                B.normal_()
            self.m.scale[0].fill_(0.7)
            self.m.scale[1].fill_(1.3)
        out, ref = self.m(self.x), self.base(self.x)
        d = self.d
        self.assertTrue(torch.allclose(out[..., :d], ref[..., :d], atol=1e-6))
        for i, s in enumerate((0.7, 1.3)):
            W = self.base.weight[(i + 1) * d:(i + 2) * d] + s * self.m.lora_B[i] @ self.m.lora_A[i]
            exp = F.linear(self.x, W, self.base.bias[(i + 1) * d:(i + 2) * d])
            self.assertTrue(torch.allclose(out[..., (i + 1) * d:(i + 2) * d], exp, atol=1e-5))

    def test_param_count(self):
        n = sum(p.numel() for p in self.m.parameters() if p.requires_grad) - sum(p.numel() for p in self.base.parameters())
        self.assertEqual(n, 4 * self.r * self.d + 2)


class TestBackbone(unittest.TestCase):
    def test_orig_in21k_architecture(self):
        import timm
        vit = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=False, num_classes=0, global_pool='token')
        self.assertIsInstance(vit.head, nn.Identity)
        self.assertIsInstance(vit.fc_norm, nn.Identity)
        self.assertEqual(vit.global_pool, 'token')
        self.assertTrue(vit.blocks[0].attn.qkv.bias is not None)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            tok = vit.forward_features(x)
            self.assertTrue(torch.allclose(vit(x), tok[:, 0], atol=1e-6))
        self.assertEqual(sum(p.numel() for p in vit.parameters()), 85798656)

    def test_vit_lae_adapted_blocks(self):
        from backbone.vit_lae import VitLAE, KVLoRAQKV
        net = VitLAE(100)
        kinds = [isinstance(b.attn.qkv, KVLoRAQKV) for b in net.body.blocks]
        self.assertEqual(kinds[:5], [True] * 5)
        self.assertFalse(any(kinds[5:]))
        self.assertAlmostEqual(net.head.weight.std().item(), 0.001, delta=2e-4)
        self.assertEqual(net.head.bias.abs().max().item(), 0.0)


class TestClassOrder(unittest.TestCase):
    def test_orders(self):
        from datasets.seq_cifar100_lae import LAE_ORDERS, mammoth_class_order
        for k, L in LAE_ORDERS.items():
            self.assertEqual(sorted(L), list(range(100)), f'order {k} not a permutation')
            co = mammoth_class_order(k)
            for i, c in enumerate(L):
                self.assertEqual(co[c], i)
        self.assertEqual(LAE_ORDERS[1], list(range(100)))
        self.assertEqual(LAE_ORDERS[2][:3], [23, 8, 11])
        self.assertEqual(LAE_ORDERS[3][:3], [53, 37, 65])


class TestTransforms(unittest.TestCase):
    def test_lae_transforms(self):
        # real (non-fake) sizes need a fresh interpreter: the dataset registers itself at import
        code = '\n'.join([
            "import sys; sys.argv=['x']",
            "import torchvision.transforms as T",
            "from datasets.seq_cifar100_lae import SequentialCIFAR100LAE as D",
            "tr, te = D.TRANSFORM.transforms, D.TEST_TRANSFORM.transforms",
            "assert [type(t) for t in tr] == [T.RandomResizedCrop, T.RandomHorizontalFlip, T.ToTensor], tr",
            "assert tr[0].size == (224, 224) and tr[0].scale == (0.08, 1.0), tr[0]",
            "assert tr[0].interpolation == T.InterpolationMode.BILINEAR",
            "assert [type(t) for t in te] == [T.Resize, T.CenterCrop, T.ToTensor], te",
            "assert te[0].size == 256 and te[0].interpolation == T.InterpolationMode.BILINEAR, te[0]",
            "assert te[1].size == (224, 224), te[1]",
            "print('transforms ok')",
        ])
        r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, 'AP_FAKE_DATA': '0'})
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn('transforms ok', r.stdout)


class TestLocalCE(unittest.TestCase):
    def test_local_ce_and_adam(self):
        torch.manual_seed(0)
        head = nn.Linear(8, 100)
        x = torch.randn(24, 8)
        past, seen = 30, 40
        y = torch.randint(past, seen, (24,))
        out = head(x)
        loss = F.cross_entropy(out[:, past:seen], y - past)
        masked = out.clone()
        masked[:, :past] = -float('inf')
        masked[:, seen:] = -float('inf')
        self.assertAlmostEqual(loss.item(), F.cross_entropy(masked, y).item(), places=5)
        before = head.weight.detach().clone()
        opt = torch.optim.Adam(head.parameters(), lr=0.0028125, weight_decay=0.0)
        opt.zero_grad()
        loss.backward()
        g = head.weight.grad
        self.assertEqual(g[:past].abs().max().item(), 0.0)
        self.assertEqual(g[seen:].abs().max().item(), 0.0)
        opt.step()
        self.assertTrue(torch.equal(head.weight[:past], before[:past]))
        self.assertTrue(torch.equal(head.weight[seen:], before[seen:]))
        self.assertFalse(torch.equal(head.weight[past:seen], before[past:seen]))


class TestPeftAdapters(unittest.TestCase):
    """LoRA and OFT through PEFT, on the study placement (qkv + proj, all blocks)."""

    def _net(self, adapter):
        from backbone.vit_lae import VitLAE
        torch.manual_seed(0)
        return VitLAE(100, ap_adapter=adapter, ap_blocks=','.join(map(str, range(12))), ap_targets='qkv,proj')

    def test_param_counts_vit_base(self):
        from backbone.vit_lae import expected_adapter_params
        b = list(range(12))
        self.assertEqual(expected_adapter_params('lora', b, ['qkv', 'proj'], 768, 8, 5, 48), 442368)
        self.assertEqual(expected_adapter_params('oft', b, ['qkv', 'proj'], 768, 8, 5, 48), 433152)
        self.assertEqual(expected_adapter_params('lae_kvlora', [0, 1, 2, 3, 4], ['qkv'], 768, 8, 5, 48), 76810)
        import timm
        from peft import LoraConfig, OFTConfig, get_peft_model
        rx = r'.*blocks\.(%s)\.attn\.(qkv|proj)' % '|'.join(map(str, b))
        for cfg, n in ((LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules=rx), 442368),
                       (OFTConfig(r=0, oft_block_size=48, target_modules=rx, module_dropout=0.0, use_cayley_neumann=False), 433152)):
            vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
            for p in vit.parameters():
                p.requires_grad_(False)
            m = get_peft_model(vit, cfg)
            self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad), n)

    def test_identity_at_init_and_counts(self):
        for adapter in ('lora', 'oft'):
            net = self._net(adapter)
            x = torch.rand(2, 3, 32, 32)
            with torch.no_grad():
                out = net.body(x)
                with net.body.disable_adapter():
                    ref = net.body(x)
            self.assertTrue(torch.allclose(out, ref, atol=1e-6), adapter)
            deltas = list(net.weight_deltas())
            self.assertEqual(len(deltas), 24, adapter)
            self.assertEqual(max(d.abs().max().item() for d, _ in deltas), 0.0, adapter)
            self.assertEqual(sum(p.numel() for p in net.adapter_parameters()), net.expected_params, adapter)

    def test_oft_rotation_exact(self):
        import peft.tuners.oft.layer as L
        net = self._net('oft')
        with torch.no_grad():
            for p in net.adapter_parameters():
                p.normal_(std=0.05)
        layer = next(m for m in net.body.modules() if isinstance(m, L.OFTLayer) and 'default' in getattr(m, 'oft_R', {}))
        rot = layer.oft_R['default']
        R = torch.block_diag(*rot._cayley_batch(rot.weight, rot.block_size, rot.use_cayley_neumann,
                                                rot.num_cayley_neumann_terms).double())
        I = torch.eye(R.shape[0], dtype=R.dtype)
        self.assertTrue(torch.allclose(R.t() @ R, I, atol=1e-5), (R.t() @ R - I).abs().max().item())   # float32 Cayley
        W = layer.get_base_layer().weight.double()
        b = layer.get_base_layer().bias.double()
        x = torch.randn(5, W.shape[1], dtype=torch.float64)
        with torch.no_grad():
            out = layer(x.float()).double()
        self.assertTrue(torch.allclose(out, F.linear(x, W @ R.t(), b), atol=1e-4))
        rel = sum(d.pow(2).sum() for d, _ in net.weight_deltas()).item()
        self.assertGreater(rel, 0.0)

    def test_lora_delta_exact(self):
        import peft.tuners.lora.layer as L
        net = self._net('lora')
        with torch.no_grad():
            for p in net.adapter_parameters():
                p.normal_(std=0.05)
        layer = next(m for m in net.body.modules() if isinstance(m, L.LoraLayer) and 'default' in getattr(m, 'lora_A', {}))
        W = layer.get_base_layer().weight.double()
        b = layer.get_base_layer().bias.double()
        A, B = layer.lora_A['default'].weight.double(), layer.lora_B['default'].weight.double()
        self.assertEqual(layer.scaling['default'], 1.0)
        x = torch.randn(5, W.shape[1], dtype=torch.float64)
        with torch.no_grad():
            out = layer(x.float()).double()
        self.assertTrue(torch.allclose(out, F.linear(x, W + B @ A, b), atol=1e-4))


class TestEndToEnd(unittest.TestCase):
    def _run(self, extra, n_checks, n_tasks=10):
        with tempfile.TemporaryDirectory() as d:
            ds = 'seq-cifar100-lae-20t' if n_tasks == 20 else 'seq-cifar100-lae'
            cmd = [sys.executable, 'main.py', '--dataset', ds, '--model', 'lae-lora',
                   '--lr', '0.0028125', '--optimizer', 'adam', '--drop_last', '1', '--seed', '2', '--ap_class_order', '2',
                   '--n_epochs', '1', '--num_workers', '0', '--non_verbose', '1', '--ap_out', d,
                   '--ap_run_name', 'e2e', '--ap_preset', 'test', '--base_path', d + '/'] + extra
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'AP_FAKE_DATA': '1'})
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            rec = json.load(open(os.path.join(d, 'e2e.json')))
            self.assertTrue(rec['complete'])
            self.assertEqual(len(rec['tasks']), n_tasks)
            self.assertTrue(all(c['ok'] for c in rec['checks'].values()))
            self.assertEqual(len(rec['checks']), n_checks)
            self.assertTrue(rec['tasks'][-1]['rel_dW'] > 0)
            return rec

    def test_fake_run_lae(self):
        self._run([], 8 + 3 * 10)

    def test_fake_run_lora_study_placement(self):
        rec = self._run(['--ap_adapter', 'lora', '--ap_blocks', '0,1,2,3,4,5,6,7,8,9,10,11', '--ap_targets', 'qkv,proj'],
                        8 + 3 * 10)
        self.assertEqual(rec['config']['ap_adapter'], 'lora')

    def test_fake_run_oft_study_placement(self):
        rec = self._run(['--ap_adapter', 'oft', '--ap_blocks', '0,1,2,3,4,5,6,7,8,9,10,11', '--ap_targets', 'qkv,proj'],
                        8 + 3 * 10)
        self.assertEqual(rec['config']['ap_adapter'], 'oft')

    def test_fake_run_validation_split(self):
        rec = self._run(['--validation', '20'], 8 + 3 * 10)
        self.assertTrue(rec['eval_split'].startswith('validation'))

    def test_fake_run_20_tasks(self):
        rec = self._run(['--ap_adapter', 'oft', '--ap_blocks', '0,1,2,3,4,5,6,7,8,9,10,11', '--ap_targets', 'qkv,proj'],
                        8 + 3 * 20, n_tasks=20)
        self.assertEqual(rec['n_tasks'], 20)
        rows = [c['detail']['rows'] for k, c in rec['checks'].items() if k.startswith('local_ce_grad_mask')]
        self.assertEqual(rows[:3], [[0, 5], [5, 10], [10, 15]])


class TestProbes(unittest.TestCase):
    """Fork-and-discard probes measure the stream model without changing it."""

    PROBE = ['--ap_probe_steps_nat', '4', '--ap_probe_steps_rand', '4', '--ap_probe_reps', '1', '--ap_probe_reps_t0', '2',
             '--ap_rand_n', '40', '--ap_probe_lrs', '1e-3,3e-3', '--ap_probe_head_lrs', '1e-3,1e-2']

    def _run(self, d, name, extra):
        cmd = [sys.executable, 'main.py', '--dataset', 'seq-cifar100-lae', '--model', 'lae-lora',
               '--lr', '1e-3', '--optimizer', 'adam', '--drop_last', '1', '--seed', '3', '--ap_class_order', '3',
               '--n_epochs', '1', '--num_workers', '0', '--non_verbose', '1', '--ap_out', d, '--ap_run_name', name,
               '--ap_preset', 'test', '--base_path', d + '/', '--ap_adapter', 'oft',
               '--ap_blocks', '0,1,2,3,4,5,6,7,8,9,10,11', '--ap_targets', 'qkv,proj',
               '--ap_data_dir', os.path.join(d, 'probe_data'), '--ap_t0_cache', os.path.join(d, 't0')] + extra
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'AP_FAKE_DATA': '1'})
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return json.load(open(os.path.join(d, name + '.json')))

    def test_probes_leave_training_bit_identical(self):
        with tempfile.TemporaryDirectory() as d:
            off = self._run(d, 'off', [])
            on = self._run(d, 'on', ['--ap_probe_points', '0,2,5'] + self.PROBE)
            again = self._run(d, 'again', ['--ap_probe_points', '0,2,5'] + self.PROBE)
            for a, b, c in zip(off['tasks'], on['tasks'], again['tasks']):
                self.assertEqual(a['adapter_sha256'], b['adapter_sha256'], f"task {a['t']}: probes changed the adapter")
                self.assertEqual(a['adapter_sha256'], c['adapter_sha256'], f"task {a['t']}: cached run differs")
                self.assertEqual(a['test_cil'], b['test_cil'], f"task {a['t']}: probes changed accuracy")
            self.assertEqual(off['A_T'], on['A_T'])
            self.assertEqual(sorted(on['probes']), ['0', '2', '5'])
            self.assertTrue(all(c['ok'] for c in on['checks'].values()))
            self.assertEqual({k for k in on['checks'] if k.startswith('probe_discarded')},
                             {'probe_discarded_t0', 'probe_discarded_t2', 'probe_discarded_t5'})
            self.assertTrue(again['probes']['0'].get('cached'))
            for t in ('0', '2', '5'):
                for name in ('dtd', 'eurosat', 'flowers102', 'randlabel'):
                    res = on['probes'][t][name]
                    self.assertGreater(res['erank'], 0)
                    self.assertEqual(len(res['full_selected']), 2 if t == '0' else 1)
                    self.assertEqual(len(res['full'][0]), 2)
                    self.assertEqual(len(res['head'][0]), 2)
                    if name != 'randlabel':
                        self.assertIn('test_acc', res['full_selected'][0])

    def test_rng_guard(self):
        import random as pyrandom
        from models.ap_utils.plasticity import RNGGuard
        torch.manual_seed(5); np.random.seed(5); pyrandom.seed(5)
        ref = (torch.rand(3), np.random.rand(3), pyrandom.random())
        torch.manual_seed(5); np.random.seed(5); pyrandom.seed(5)
        with RNGGuard():
            torch.manual_seed(99); torch.rand(100); np.random.rand(7); pyrandom.random()
        got = (torch.rand(3), np.random.rand(3), pyrandom.random())
        self.assertTrue(torch.equal(ref[0], got[0]))
        self.assertTrue(np.array_equal(ref[1], got[1]))
        self.assertEqual(ref[2], got[2])


class TestNewSettings(unittest.TestCase):
    """Full-model continual learning and plain (joint, single-task) training with in-run measurement points."""

    PROBE = TestProbes.PROBE
    METRICS = ['--ap_metrics', '1', '--ap_grad_n', '4', '--ap_hess_n', '8', '--ap_hess_iters', '3', '--ap_metric_n', '16',
               '--ap_lp_every', '1']
    ALL = '0,1,2,3,4,5,6,7,8,9,10,11'

    def _run(self, d, name, extra, ds='seq-cifar100-lae'):
        cmd = [sys.executable, 'main.py', '--dataset', ds, '--model', 'lae-lora', '--optimizer', 'adam',
               '--drop_last', '1', '--seed', '2', '--ap_class_order', '2', '--n_epochs', '1', '--num_workers', '0',
               '--non_verbose', '1', '--ap_out', d, '--ap_run_name', name, '--ap_preset', 'test', '--base_path', d + '/',
               '--ap_blocks', self.ALL, '--ap_targets', 'qkv,proj',
               '--ap_data_dir', os.path.join(d, 'probe_data'), '--ap_t0_cache', os.path.join(d, 't0')] + extra
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'AP_FAKE_DATA': '1'})
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return json.load(open(os.path.join(d, name + '.json')))

    def test_full_model_continual(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._run(d, 'full', ['--ap_adapter', 'full', '--lr', '1e-4', '--ap_probe_points', '0,3'] +
                            self.METRICS + self.PROBE)
            self.assertTrue(rec['complete'])
            self.assertTrue(all(c['ok'] for c in rec['checks'].values()), rec['checks'])
            self.assertIn('full_model_trainable', rec['checks'])
            self.assertNotIn('backbone_frozen', rec['checks'])
            self.assertTrue(rec['checks']['adapter_identity_at_init']['ok'])
            last = rec['tasks'][-1]
            self.assertGreater(last['rel_dW'], 0.0)
            self.assertGreater(last['rel_dW_all'], 0.0)
            self.assertEqual(rec['checks']['adapted_matrix_count']['detail']['found'], 24)
            for k in ('erank', 'grad_cos_mean', 'hessian_lambda_max', 'dW_step'):
                self.assertTrue(np.isfinite(last['metrics'][k]), k)
            self.assertEqual(sorted(rec['probes']), ['0', '3'])
            self.assertTrue(rec['checks']['probe_discarded_t3']['ok'])

    def test_separate_head_lr(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._run(d, 'hlr', ['--ap_adapter', 'lora', '--lr', '1e-3', '--ap_head_lr', '1e-2'])
            self.assertTrue(rec['complete'] and all(c['ok'] for c in rec['checks'].values()))
            self.assertEqual([g['lr'] for g in rec['optimizer_groups']], [1e-3, 1e-2])
            self.assertEqual(rec['optimizer_groups'][1]['n_params'], 192 * 100 + 100)       # vit_tiny head
            base = self._run(d, 'base', ['--ap_adapter', 'lora', '--lr', '1e-3'])
            self.assertEqual(len(base['optimizer_groups']), 1)                                # calibrated protocol

    def test_joint_points_leave_training_bit_identical(self):
        with tempfile.TemporaryDirectory() as d:
            base = ['--ap_adapter', 'lora', '--lr', '1e-3', '--joint', '1']
            off = self._run(d, 'off', base, ds='seq-cifar100-lae-20t')
            on = self._run(d, 'on', base + ['--ap_joint_points', '4', '--ap_probe_points', '0,2,4'] + self.METRICS +
                           self.PROBE, ds='seq-cifar100-lae-20t')
            self.assertEqual(off['n_tasks'], 1)
            self.assertEqual([r['t'] for r in off['tasks']], [1])
            self.assertEqual([r['t'] for r in on['tasks']], [1, 2, 3, 4])
            self.assertEqual(on['tasks'][-1]['adapter_sha256'], off['tasks'][-1]['adapter_sha256'],
                             'measurement points changed training')
            self.assertEqual(on['tasks'][-1]['test_cil'], off['tasks'][-1]['test_cil'])
            self.assertEqual(on['A_T'], off['A_T'])
            self.assertEqual(len(set(r['adapter_sha256'] for r in on['tasks'])), 4)     # points are taken mid-run
            S = on['joint_points']['total_updates']
            self.assertEqual(on['joint_points']['steps'], [round(k * S / 4) for k in (1, 2, 3)])
            self.assertEqual(sorted(on['probes']), ['0', '2', '4'])
            self.assertTrue(all(c['ok'] for c in on['checks'].values()))
            self.assertEqual(on['tasks'][0]['test_cil'].__len__(), 1)                   # one all-class task


class TestTrainingEquivalence(unittest.TestCase):
    """Training is bit-identical to the code version that passed the calibration gates.

    The installer puts that exact code (file SHA-256s in tests/training_reference.json) in a reference copy of Mammoth
    (AP_REF_ROOT, default <mammoth>_ref). Each case runs in both copies in this environment, and the current code must
    give the same adapter SHA-256 and the same accuracies after every task, with the stream metrics off and on.
    Comparing in the same environment makes the check independent of hardware and library versions."""

    ALL = '0,1,2,3,4,5,6,7,8,9,10,11'
    CASES = {
        'lae_kvlora_10t': ['--dataset', 'seq-cifar100-lae', '--lr', '0.0028125', '--seed', '1', '--ap_class_order', '1'],
        'lora_study_10t': ['--dataset', 'seq-cifar100-lae', '--lr', '1e-4', '--seed', '2', '--ap_class_order', '2',
                           '--ap_adapter', 'lora', '--ap_blocks', ALL, '--ap_targets', 'qkv,proj', '--ap_lora_r', '8',
                           '--ap_lora_alpha', '8'],
        'oft_study_20t': ['--dataset', 'seq-cifar100-lae-20t', '--lr', '1e-4', '--seed', '3', '--ap_class_order', '3',
                          '--ap_adapter', 'oft', '--ap_blocks', ALL, '--ap_targets', 'qkv,proj'],
        'lora_lae_val': ['--dataset', 'seq-cifar100-lae', '--lr', '3e-4', '--seed', '1', '--ap_class_order', '1',
                         '--validation', '10', '--ap_adapter', 'lora', '--ap_lora_r', '5', '--ap_lora_alpha', '5'],
    }
    METRIC_ARGS = ['--ap_grad_n', '4', '--ap_hess_n', '8', '--ap_hess_iters', '3', '--ap_metric_n', '16', '--ap_lp_every', '1']

    def _run(self, root, extra, ckpt=False):
        with tempfile.TemporaryDirectory() as d:
            cmd = [sys.executable, 'main.py', '--model', 'lae-lora', '--optimizer', 'adam', '--drop_last', '1',
                   '--n_epochs', '1', '--num_workers', '0', '--non_verbose', '1', '--ap_out', d, '--ap_run_name', 'x',
                   '--ap_preset', 'eq', '--base_path', d + '/'] + extra
            if ckpt:
                cmd += ['--ap_ckpt_dir', os.path.join(d, 'ckpt')]
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env={**os.environ, 'AP_FAKE_DATA': '1'})
            self.assertEqual(r.returncode, 0, f'{root}: ' + r.stdout[-2000:] + r.stderr[-2000:])
            rec = json.load(open(os.path.join(d, 'x.json')))
            if ckpt:
                self._check_ckpt(rec, os.path.join(d, 'ckpt'))
            return rec

    def _check_ckpt(self, rec, d):
        """One checkpoint per point, and each holds exactly the adapter whose SHA-256 the run recorded."""
        from models.ap_utils.plasticity import tensors_sha
        files = sorted(os.listdir(d))
        self.assertEqual(files, [f't{t:02d}.pt' for t in range(len(rec['tasks']) + 1)])
        for row in rec['tasks']:
            ck = torch.load(os.path.join(d, row['ckpt']['file']))
            self.assertEqual(ck['t'], row['t'])
            adapter = [v for n, v in ck['state'].items() if n.startswith('body.')]
            self.assertEqual(tensors_sha(adapter)[:32], row['adapter_sha256'], f't={row["t"]}: checkpoint != trained adapter')
            self.assertTrue(any('head' in n for n in ck['state']), 'head missing from checkpoint')
            self.assertGreater(len(ck['state']), 1)

    def test_reference_is_the_calibrated_code(self):
        import hashlib
        ref = json.load(open(os.path.join(ROOT, 'tests', 'training_reference.json')))
        root = os.environ.get('AP_REF_ROOT', ROOT.rstrip('/') + '_ref')
        h = hashlib.sha256()
        for rel in sorted(ref['files']):
            data = open(os.path.join(root, rel), 'rb').read()
            self.assertEqual(hashlib.sha256(data).hexdigest(), ref['files'][rel], f'reference copy differs: {rel}')
            h.update(rel.encode() + data)
        self.assertEqual(h.hexdigest()[:16], ref['code'])

    def test_training_identical_to_reference(self):
        ref_root = os.environ.get('AP_REF_ROOT', ROOT.rstrip('/') + '_ref')
        for name, args in self.CASES.items():
            ref = self._run(ref_root, args)
            for metrics in ('0', '1'):
                cur = self._run(ROOT, args + ['--ap_metrics', metrics] + self.METRIC_ARGS, ckpt=metrics == '1')
                self.assertEqual([t['adapter_sha256'] for t in cur['tasks']], [t['adapter_sha256'] for t in ref['tasks']],
                                 f'{name}, metrics={metrics}: training differs from the calibrated code')
                self.assertEqual([t['test_cil'] for t in cur['tasks']], [t['test_cil'] for t in ref['tasks']],
                                 f'{name}, metrics={metrics}')
                if metrics == '1':
                    self._check_metrics(name, cur)

    def _check_metrics(self, name, rec):
        self.assertNotIn('dW_step', rec['metrics_t0'])
        for row in rec['tasks']:
            m = row['metrics']
            for k in ('erank', 'srank', 'dormant_0.025', 'adapter_norm', 'dW_step', 'grad_norm_mean', 'grad_cos_mean',
                      'hessian_top_eig', 'hessian_lambda_max', 'hessian_lambda_min', 'hessian_lambda_max_resid'):
                self.assertIn(k, m, f'{name} t={row["t"]}: {k} missing')
                self.assertTrue(np.isfinite(m[k]), f'{name} t={row["t"]}: {k} = {m[k]}')
            self.assertTrue(0.0 <= m['dormant_0.1'] <= 1.0)
            self.assertTrue(-1.0 <= m['grad_cos_mean'] <= 1.0)
            self.assertGreater(m['dW_step'], 0.0)
            self.assertGreaterEqual(m['hessian_lambda_max'], m['hessian_lambda_min'])
            self.assertIn('lp_rel_mean', row['loss_progress'])


class TestMetricMath(unittest.TestCase):
    def test_hessian_power_iteration_quadratic(self):
        from models.ap_utils.stream_metrics import top_hessian_eig
        torch.manual_seed(0)
        Q = torch.randn(6, 6, dtype=torch.float64)
        A = Q @ Q.t()
        p = torch.randn(6, dtype=torch.float64, requires_grad=True)
        lam = top_hessian_eig(0.5 * p @ A @ p, [p], iters=200)
        self.assertAlmostEqual(lam, torch.linalg.eigvalsh(A).max().item(), places=6)

    def test_hessian_extremes_indefinite(self):
        """Indefinite Hessian whose largest-magnitude eigenvalue is negative: lambda_max must still be the most
        positive eigenvalue, lambda_min the most negative, both with near-zero residual."""
        from models.ap_utils.stream_metrics import hessian_extremes
        torch.manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
        ev = torch.tensor([-9.0, -2.0, -0.5, 0.1, 0.7, 1.5, 3.0, 4.0], dtype=torch.float64)
        A = Q @ torch.diag(ev) @ Q.t()
        p = torch.randn(8, dtype=torch.float64, requires_grad=True)
        h = hessian_extremes(0.5 * p @ A @ p, [p], iters=300)
        self.assertAlmostEqual(h['top'], -9.0, places=5)
        self.assertAlmostEqual(h['lambda_max'], 4.0, places=5)
        self.assertAlmostEqual(h['lambda_min'], -9.0, places=5)
        self.assertLess(h['lambda_max_resid'], 1e-4)
        self.assertLess(h['lambda_min_resid'], 1e-4)

    def test_ranks(self):
        from models.ap_utils.stream_metrics import effective_rank, srank
        f = torch.zeros(50, 10)
        f[:25, 0], f[25:, 0] = 1.0, -1.0
        self.assertAlmostEqual(effective_rank(f), 1.0, places=6)
        self.assertEqual(srank(torch.eye(10)), 10)
        self.assertEqual(srank(torch.diag(torch.tensor([1.0] + [0.0] * 9))), 1)

if __name__ == '__main__':
    res = unittest.main(exit=False, verbosity=2).result
    ok = res.wasSuccessful()
    print('GATE0', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
