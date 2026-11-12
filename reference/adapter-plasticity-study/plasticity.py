"""
Plasticity probes for the calibrated pipeline (fork-and-discard protocol, Hernandez-Garcia, Figliolia and
Millidge 2026, arXiv 2606.24752).

At a measurement point t (after task t; t = 0 is the fresh adapter before task 1) the stream model is forked:
adapter tensors are snapshotted, a fresh linear head is attached, adapter + head are trained on a probe task
that is never part of the stream, the result is recorded, and the adapter is restored. The probe never
changes the stream run:
  - adapter tensors are restored exactly and their SHA-256 is checked before vs after (check `probe_discarded`);
  - gradients left on adapter tensors are cleared;
  - every random-number state (python, numpy, torch CPU, torch CUDA) is saved before and restored after;
  - the stream optimizer, head and data loader are never touched.
Gate 0 tests that a run with probes is bit-identical to the same run without probes.

Probe tasks, preprocessed once and cached (uint8, 224 x 224, the stream's test transform: Resize(256, bilinear) +
CenterCrop(224); pixels passed to the model in [0, 1] without normalization, as in the stream):
  natural  : DTD (pool = train + val partition 1, test = official test), EuroSAT (one pool, disjoint splits),
             Flowers102 (pool = train + val, test = official test). VTAB-1k style: 800 train + 200 val sampled
             from the pool with a fixed seed, test subsampled to at most 2,000 with a fixed seed.
  randlabel: CIFAR-10 training images with fixed random labels over 10 classes (memorisation capacity,
             Lyle et al.); the first --ap_rand_n of a fixed 10,000-image draw, kept at 32 x 32, resized on the fly.

For each probe and repetition r (r sets data order and head initialisation; identical across measurement
points and learning rates, i.e. a paired design):
  full : adapter + head trained at each adapter LR in --ap_probe_lrs (head at --ap_probe_head_lr); best LR
         chosen on val accuracy (ties: lower val loss) for natural probes, on final train loss for randlabel.
  head : adapter frozen, head only, on cached features, at each LR in --ap_probe_head_lrs (feature-quality control).
Recorded per run: train loss/acc at S/4, S/2, 3S/4, S (randlabel intermediate points on a fixed 2,000-image
subset), val loss/acc (natural), test loss/acc at S (natural); per probe: feature effective rank
(Roy and Vetterli 2007) on up to 2,000 probe training images. No augmentation in probes.
"""
import hashlib
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FAKE = os.environ.get('AP_FAKE_DATA') == '1'
NATURAL = ('dtd', 'eurosat', 'flowers102')
EXPECTED = {'dtd': (800, 200, 1880), 'eurosat': (800, 200, 2000), 'flowers102': (800, 200, 2000)}
CACHE_VERSION = 'v2_bilinear256_cc224'


def add_probe_args(parser):
    g = parser.add_argument_group('Plasticity probes')
    g.add_argument('--ap_probe_points', type=str, default='', help='Comma-separated t at which to probe (0 = before task 1). Empty = off.')
    g.add_argument('--ap_probes', type=str, default='dtd,eurosat,flowers102,randlabel')
    g.add_argument('--ap_probe_reps', type=int, default=2)
    g.add_argument('--ap_probe_reps_t0', type=int, default=5, help='Repetitions at t = 0 (noise floor).')
    g.add_argument('--ap_probe_lrs', type=str, default='3e-4,1e-3,3e-3')
    g.add_argument('--ap_probe_head_lrs', type=str, default='3e-4,1e-3,3e-3,1e-2')
    g.add_argument('--ap_probe_head_lr', type=float, default=1e-3)
    g.add_argument('--ap_probe_steps_nat', type=int, default=250)
    g.add_argument('--ap_probe_steps_rand', type=int, default=600)
    g.add_argument('--ap_rand_n', type=int, default=2000)
    g.add_argument('--ap_probe_bs', type=int, default=64)
    g.add_argument('--ap_probe_seed', type=int, default=12345)
    g.add_argument('--ap_data_dir', type=str, default='ap_data', help='Probe data cache directory.')
    return parser


def probe_points(args):
    return {int(x) for x in str(getattr(args, 'ap_probe_points', '')).split(',') if x.strip() != ''}


# ------------------------------------------------------------------ data
def _prep_pil(img, resize, crop):
    from torchvision.transforms import functional as TF
    from torchvision.transforms import InterpolationMode
    img = TF.resize(img.convert('RGB'), resize, interpolation=InterpolationMode.BILINEAR)
    img = TF.center_crop(img, crop)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).contiguous()


def _stack(ds, idxs, resize=256, crop=224):
    return torch.stack([_prep_pil(ds[int(i)][0], resize, crop) for i in idxs])


def _labels(ds, idxs):
    return torch.tensor([int(ds[int(i)][1]) for i in idxs])


def build_probe_data(root):
    import torchvision.datasets as D
    out = {}
    if FAKE:
        g = torch.Generator().manual_seed(0)
        for name, nc in (('dtd', 47), ('eurosat', 10), ('flowers102', 102)):
            mk = lambda n: (torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g),
                            torch.randint(0, nc, (n,), generator=g))
            (xtr, ytr), (xva, yva), (xte, yte) = mk(16), mk(8), mk(8)
            out[name] = dict(Xtr=xtr, Ytr=ytr, Xval=xva, Yval=yva, Xte=xte, Yte=yte, n_classes=nc)
        x = torch.randint(0, 256, (40, 3, 32, 32), dtype=torch.uint8, generator=g)
        out['randlabel'] = dict(X=x, Y=torch.from_numpy(np.random.default_rng(1019).integers(0, 10, 40)), n_classes=10)
        return out
    tr, va, te = (D.DTD(root, split=s, partition=1, download=True) for s in ('train', 'val', 'test'))
    pool = torch.utils.data.ConcatDataset([tr, va])
    perm = np.random.default_rng(11).permutation(len(pool))
    i_tr, i_va = perm[:800], perm[800:1000]
    i_te = np.random.default_rng(12).permutation(len(te))[:2000]
    out['dtd'] = dict(Xtr=_stack(pool, i_tr), Ytr=_labels(pool, i_tr), Xval=_stack(pool, i_va), Yval=_labels(pool, i_va),
                      Xte=_stack(te, i_te), Yte=_labels(te, i_te), n_classes=47)
    eu = D.EuroSAT(root, download=True)
    perm = np.random.default_rng(21).permutation(len(eu))
    i_tr, i_va, i_te = perm[:800], perm[800:1000], perm[1000:3000]
    out['eurosat'] = dict(Xtr=_stack(eu, i_tr), Ytr=_labels(eu, i_tr), Xval=_stack(eu, i_va), Yval=_labels(eu, i_va),
                          Xte=_stack(eu, i_te), Yte=_labels(eu, i_te), n_classes=10)
    tr, va, te = (D.Flowers102(root, split=s, download=True) for s in ('train', 'val', 'test'))
    pool = torch.utils.data.ConcatDataset([tr, va])
    perm = np.random.default_rng(31).permutation(len(pool))
    i_tr, i_va = perm[:800], perm[800:1000]
    i_te = np.random.default_rng(32).permutation(len(te))[:2000]
    out['flowers102'] = dict(Xtr=_stack(pool, i_tr), Ytr=_labels(pool, i_tr), Xval=_stack(pool, i_va), Yval=_labels(pool, i_va),
                             Xte=_stack(te, i_te), Yte=_labels(te, i_te), n_classes=102)
    c10 = D.CIFAR10(root, train=True, download=True)
    idx = np.random.default_rng(1019).permutation(len(c10.data))[:10000]
    out['randlabel'] = dict(X=torch.from_numpy(c10.data[idx]).permute(0, 3, 1, 2).contiguous(),
                            Y=torch.from_numpy(np.random.default_rng(1020).integers(0, 10, len(idx))), n_classes=10)
    return out


def load_probe_data(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"probe_data_{'fake' if FAKE else CACHE_VERSION}.pt")
    if os.path.exists(path):
        return torch.load(path)
    data = build_probe_data(os.path.join(cache_dir, 'raw'))
    torch.save(data, path)
    return data


def data_manifest(data):
    """Sizes and SHA-256 of every probe tensor (recorded in each run; checked against the expected sizes)."""
    m = {}
    for name, d in data.items():
        m[name] = {k: dict(shape=list(v.shape), sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()[:16])
                   for k, v in d.items() if torch.is_tensor(v)}
        m[name]['n_classes'] = d['n_classes']
    return m


def check_data(data):
    """(ok, detail): natural probes have the VTAB-1k sizes, randlabel has 10,000 images (real data only)."""
    if FAKE:
        return True, 'fake data'
    bad = {}
    for name, (ntr, nva, nte) in EXPECTED.items():
        got = (len(data[name]['Ytr']), len(data[name]['Yval']), len(data[name]['Yte']))
        if got != (ntr, nva, nte):
            bad[name] = got
    if len(data['randlabel']['Y']) != 10000:
        bad['randlabel'] = len(data['randlabel']['Y'])
    return not bad, bad or 'sizes as expected'


# ------------------------------------------------------------------ helpers
def effective_rank(feats):
    f = feats.double() - feats.double().mean(0, keepdim=True)
    s = torch.linalg.svdvals(f)
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def tensors_sha(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RNGGuard:
    """Saves every random-number state on entry and restores it on exit."""

    def __enter__(self):
        self.py, self.np, self.cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return self

    def __exit__(self, *exc):
        random.setstate(self.py)
        np.random.set_state(self.np)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)
        return False


class ProbeRunner:
    def __init__(self, args, net, device, data):
        self.args, self.net, self.device, self.data = args, net, device, data
        self.img = 32 if FAKE else 224
        self.amp = dict(device_type='cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else None

    def _prep(self, x_uint8):
        x = x_uint8.to(self.device, non_blocking=True).float().div_(255.0)
        if x.shape[-1] != self.img:
            x = F.interpolate(x, size=(self.img, self.img), mode='bilinear', align_corners=False).clamp_(0, 1)
        return x                                   # [0, 1], no normalization: same input convention as the stream

    def _feats(self, x_uint8):
        if self.amp:
            with torch.autocast(**self.amp):
                return self.net(self._prep(x_uint8), returnt='features').float()
        return self.net(self._prep(x_uint8), returnt='features').float()

    @torch.no_grad()
    def _all_feats(self, X, bs=256):
        return torch.cat([self._feats(X[i:i + bs]) for i in range(0, len(X), bs)])

    @torch.no_grad()
    def _eval(self, head, X, Y, feats=None, bs=256):
        loss = correct = 0.0
        for i in range(0, len(X), bs):
            f = feats[i:i + bs] if feats is not None else self._feats(X[i:i + bs])
            logits = head(f)
            y = Y[i:i + bs].to(self.device)
            loss += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(1) == y).sum().item()
        return loss / len(X), correct / len(X)

    def _train(self, P, name, lr, rep, mode, feats=None):
        nat = name in NATURAL
        X, Y = (P['Xtr'], P['Ytr']) if nat else (P['X'], P['Y'])
        steps = self.args.ap_probe_steps_nat if nat else self.args.ap_probe_steps_rand
        seed = self.args.ap_probe_seed + 1000 * rep + (NATURAL + ('randlabel',)).index(name)
        torch.manual_seed(seed)
        head = nn.Linear(self.net.num_features, P['n_classes']).to(self.device)
        g = torch.Generator().manual_seed(seed)
        if mode == 'full':
            opt = torch.optim.Adam([{'params': self.net.adapter_parameters(), 'lr': lr},
                                    {'params': head.parameters(), 'lr': self.args.ap_probe_head_lr}])
        else:
            opt = torch.optim.Adam(head.parameters(), lr=lr)
        checkpoints = sorted({max(1, steps // 4), max(1, steps // 2), max(1, 3 * steps // 4), steps})
        sub = torch.arange(min(2000, len(X)))
        rec = dict(lr=lr, rep=rep, steps=steps, curve=[])
        for s in range(1, steps + 1):
            idx = torch.randint(0, len(X), (self.args.ap_probe_bs,), generator=g)
            y = Y[idx].to(self.device)
            if mode == 'full':
                self.net.train()
                logits = head(self._feats(X[idx]))
            else:
                logits = head(feats['tr'][idx.to(feats['tr'].device)])
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if s in checkpoints:
                self.net.eval()
                pt = dict(step=s)
                if nat:
                    pt['train_loss'], pt['train_acc'] = self._eval(head, X, Y, feats=feats['tr'] if feats else None)
                    pt['val_loss'], pt['val_acc'] = self._eval(head, P['Xval'], P['Yval'], feats=feats['val'] if feats else None)
                else:
                    Xs, Ys = (X, Y) if s == steps else (X[sub], Y[sub])
                    fs = None if feats is None else (feats['tr'] if s == steps else feats['tr'][sub.to(feats['tr'].device)])
                    pt['train_loss'], pt['train_acc'] = self._eval(head, Xs, Ys, feats=fs)
                    pt['train_subset'] = s != steps
                rec['curve'].append(pt)
        self.net.eval()
        if nat:
            rec['test_loss'], rec['test_acc'] = self._eval(head, P['Xte'], P['Yte'], feats=feats['te'] if feats else None)
        last = rec['curve'][-1]
        rec.update({k: last[k] for k in last if k != 'step'})
        return rec

    @staticmethod
    def _select(runs, nat):
        if nat:
            return max(runs, key=lambda r: (round(r['val_acc'], 6), -r['val_loss']))
        return min(runs, key=lambda r: r['train_loss'])

    def measure(self, reps):
        """One measurement point. Returns (results, sha_before, sha_after) of the adapter tensors."""
        t0 = time.time()
        params = self.net.adapter_parameters()
        snap = [p.detach().clone() for p in params]
        sha_before = tensors_sha(params)
        was_training = self.net.training
        out = {}
        with RNGGuard():
            self.net.eval()
            for name in [n for n in self.args.ap_probes.split(',') if n]:
                nat = name in NATURAL
                P = {k: (v.to(self.device) if torch.is_tensor(v) and nat else v) for k, v in self.data[name].items()}
                if not nat and self.args.ap_rand_n < len(P['X']):
                    P = dict(P, X=P['X'][:self.args.ap_rand_n], Y=P['Y'][:self.args.ap_rand_n])
                res = dict(full=[], head=[], full_selected=[], head_selected=[])
                if nat:
                    feats = dict(tr=self._all_feats(P['Xtr']), val=self._all_feats(P['Xval']), te=self._all_feats(P['Xte']))
                else:
                    feats = dict(tr=self._all_feats(P['X']))
                res['erank'] = effective_rank(feats['tr'][:2000])
                for rep in range(reps):
                    runs = []
                    for lr in [float(x) for x in self.args.ap_probe_lrs.split(',')]:
                        runs.append(self._train(P, name, lr, rep, 'full'))
                        with torch.no_grad():
                            for p, s in zip(params, snap):
                                p.copy_(s)
                    res['full'].append(runs)
                    res['full_selected'].append(self._select(runs, nat))
                    hruns = [self._train(P, name, lr, rep, 'head', feats=feats)
                             for lr in [float(x) for x in self.args.ap_probe_head_lrs.split(',')]]
                    res['head'].append(hruns)
                    res['head_selected'].append(self._select(hruns, nat))
                del feats, P
                if str(self.device).startswith('cuda'):
                    torch.cuda.empty_cache()
                out[name] = res
        with torch.no_grad():
            for p, s in zip(params, snap):
                p.copy_(s)
                p.grad = None
        self.net.train(was_training)
        out['_minutes'] = (time.time() - t0) / 60
        return out, sha_before, tensors_sha(params)
