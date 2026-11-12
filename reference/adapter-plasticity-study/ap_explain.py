"""
Why do the proxy plasticity metrics track the learning rate instead of plasticity? Measurement-only re-analysis of the
saved study checkpoints (adapter + head after every task; class orders 2 and 3). Nothing is trained.

For every checkpoint t = 0..20 of every study run it records:
  head: Frobenius norm of the classifier rows of the classes seen so far (head_norm_seen) and of all rows; mean CLS
        feature norm on each image set. Tests the division-of-labour hypothesis: adapter and head share one learning
        rate, so at a low LR the head stays small and the features, not the head, must separate the classes.
  proxies on FIXED image sets, identical across runs and points (unlike the stream metrics, which use the current
        task's images): effective rank and srank of CLS features, dormant MLP units (tau 0.025), on
          cifar100_all : 2,000 CIFAR-100 test images from all 100 classes (the stream's distribution)
          cifar10      : the 2,000 CIFAR-10 images of the random-label probe (natural objects, outside the stream)
          dtd          : the 800 DTD training images of the DTD probe (textures, far from the stream)
        This separates "the representation of the stream's distribution changed" from "the representation changed".

Every checkpoint is verified: the adapter SHA-256 of the loaded tensors must equal the one recorded in the run file.
Output: <out>/study/explain_long.csv (run, arm, order, t, set, measure, value).

Usage (Colab, GPU, from any directory):
  !python /content/drive/MyDrive/ap/ap_explain.py --mammoth /content/mammoth --data /content/apdata
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument('--mammoth', default='/content/mammoth')
ap.add_argument('--data', default='/content/apdata')
ap.add_argument('--points', default='', help='Comma-separated checkpoints t to measure (default: all)')
ap.add_argument('--n_cifar', type=int, default=2000)
args = ap.parse_args()
sys.path.insert(0, args.mammoth)
os.chdir(args.mammoth)
sys.argv = [sys.argv[0]]

from backbone.vit_lae import VitLAE                                   # noqa: E402
from models.ap_utils.plasticity import load_probe_data, tensors_sha   # noqa: E402
from models.ap_utils.stream_metrics import effective_rank, srank, _mlp_acts  # noqa: E402

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
FAKE = os.environ.get('AP_FAKE_DATA') == '1'
IMG = 32 if FAKE else 224


def prep(x_uint8):
    x = x_uint8.to(dev).float().div_(255.0)
    if x.shape[-1] != IMG:
        x = F.interpolate(x, size=(IMG, IMG), mode='bilinear', align_corners=False).clamp_(0, 1)
    return x


def image_sets():
    probes = load_probe_data(os.path.join(args.data, 'probes'))
    sets = {'dtd': probes['dtd']['Xtr'], 'cifar10': probes['randlabel']['X'][:2000]}
    if FAKE:
        sets['cifar100_all'] = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
        return sets
    import torchvision
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF
    ds = torchvision.datasets.CIFAR100(os.path.join(args.data, 'mammoth', 'CIFAR100'), train=False, download=False)
    idx = np.random.default_rng(7).permutation(len(ds.data))[:args.n_cifar]          # fixed, all classes
    imgs = []
    from PIL import Image
    for i in idx:
        im = TF.center_crop(TF.resize(Image.fromarray(ds.data[i]), 256, interpolation=InterpolationMode.BILINEAR), 224)
        imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
    sets['cifar100_all'] = torch.stack(imgs)
    return sets


@torch.no_grad()
def measure(net, X):
    acts = _mlp_acts(net)
    sums, hooks = [None] * len(acts), []
    for i, a in enumerate(acts):
        def hook(_m, _i, o, i=i):
            v = o.detach().abs().float().mean(dim=tuple(range(o.dim() - 1)))
            sums[i] = v if sums[i] is None else sums[i] + v
        hooks.append(a.register_forward_hook(hook))
    feats = []
    for i in range(0, len(X), 128):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            feats.append(net(prep(X[i:i + 128]), returnt='features').float())
    for h in hooks:
        h.remove()
    f = torch.cat(feats)
    dorm = float(np.mean([float(((s / s.mean()) <= 0.025).float().mean()) for s in sums]))
    return dict(erank=effective_rank(f), srank=srank(f), dormant_0025=dorm, feat_norm=float(f.norm(dim=1).mean()))


def build(cfg):
    keys = ('ap_adapter', 'ap_blocks', 'ap_targets', 'ap_lae_rank', 'ap_lora_r', 'ap_lora_alpha', 'ap_oft_block_size',
            'ap_timm_model')
    net = VitLAE(num_classes=100, **{k: cfg[k] for k in keys if cfg.get(k) is not None}).to(dev).eval()
    return net


rows, t0 = [], time.time()
want = {int(x) for x in args.points.split(',') if x.strip()} if args.points else None
sets = image_sets()
print('image sets:', {k: tuple(v.shape) for k, v in sets.items()}, flush=True)
runs = sorted(glob.glob(os.path.join(args.out, 'runs', '*', 'ckpt', 'study_*')))
print(f'{len(runs)} runs with checkpoints', flush=True)
for cdir in runs:
    name = os.path.basename(cdir)
    rec = json.load(open(os.path.join(os.path.dirname(os.path.dirname(cdir)), name + '.json')))
    m = re.match(r'study_(.+)_order(\d)_s(\d)', name)
    arm, order = m.group(1), int(m.group(2))
    net = build(rec['config'])
    trainable = {n: p for n, p in net.named_parameters() if p.requires_grad}
    shas = {0: rec.get('ckpt_t0', {}).get('adapter_sha256')}
    shas.update({r['t']: r.get('ckpt', {}).get('adapter_sha256') for r in rec['tasks']})
    for f in sorted(glob.glob(os.path.join(cdir, 't*.pt'))):
        ck = torch.load(f, map_location=dev)
        t = int(ck['t'])
        if want is not None and t not in want:
            continue
        missing = set(trainable) ^ set(ck['state'])
        if missing:
            sys.exit(f'ABORT: {f}: parameter names differ from the model: {sorted(missing)[:5]}')
        with torch.no_grad():
            for n, p in trainable.items():
                p.copy_(ck['state'][n])
        got = tensors_sha(net.adapter_parameters())[:32]
        if shas.get(t) and got != shas[t]:
            sys.exit(f'ABORT: {f}: adapter SHA-256 {got} != recorded {shas[t]}')
        seen = 5 * t if t > 0 else 0
        hw = net.head.weight.detach().float()
        base = dict(run=name, arm=arm, order=order, t=t)
        rows.append(dict(base, set='head', measure='head_norm_all', value=float(hw.norm())))
        if seen:
            rows.append(dict(base, set='head', measure='head_norm_seen', value=float(hw[:seen].norm())))
        for sname, X in sets.items():
            for k, v in measure(net, X).items():
                rows.append(dict(base, set=sname, measure=k, value=v))
    print(f'{name}: done, {(time.time() - t0) / 60:.1f} min so far', flush=True)
    del net
    if dev == 'cuda':
        torch.cuda.empty_cache()

os.makedirs(os.path.join(args.out, 'study'), exist_ok=True)
p = os.path.join(args.out, 'study', 'explain_long.csv')
with open(p, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['run', 'arm', 'order', 't', 'set', 'measure', 'value'])
    w.writeheader()
    w.writerows(rows)
print(f'{len(rows)} rows -> {p}')
