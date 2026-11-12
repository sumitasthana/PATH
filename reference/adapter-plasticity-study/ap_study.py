"""
Analysis of the study runs (presets.yaml `study`): every plasticity measure per arm, per task, per class order.

Reads <out>/runs/<code>/study_*.json (complete runs only) and writes into <out>/study/:
  study_long.csv     one row per (run, t, measure): the raw table every figure and number comes from
  STUDY.md           tables: per arm, mean over the class orders available (n shown), at t = 0, 5, 10, 20
  fig_<measure>.png  measure vs task (left) and vs dW (right), one line per arm, mean over orders

Measures (all already in the result files; nothing is recomputed from the model):
  dW, dW_step                          adapter weight change, total and during task t
  newtask_cil, newtask_til             accuracy on task t right after learning it (Dohare et al. 2024)
  A_t                                  accuracy over all seen tasks
  probe_<task>_full / _head            fork-probe test accuracy (natural) or final train loss (randlabel),
                                       best LR per repetition, mean over repetitions; _ratio = value(t) / value(0)
  probe_<task>_erank                   effective rank of features on the probe images
  lp_rel                               per-update loss progress (Elsayed and Mahmood 2024)
  erank, srank, dormant_*, adapter_norm, grad_norm_mean, grad_cos_mean, hessian_top_eig   internal state

Usage (Colab):  !python /content/drive/MyDrive/ap/ap_study.py
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()
OUT = os.path.join(args.out, 'study')
os.makedirs(OUT, exist_ok=True)

NATURAL = ('dtd', 'eurosat', 'flowers102')
INTERNAL = ('erank', 'srank', 'dormant_0.01', 'dormant_0.025', 'dormant_0.1', 'adapter_norm', 'grad_norm_mean',
            'grad_cos_mean', 'grad_cos_neg_frac', 'hessian_top_eig', 'hessian_lambda_max', 'hessian_lambda_max_resid',
            'hessian_lambda_min', 'hessian_lambda_min_resid')
RESID_MAX = 0.1          # Hessian eigenvalues with relative residual above this are reported as not converged
# raw scale depends on the adapter's parameterisation: compare within LoRA or within OFT, never across
WITHIN_ADAPTER_ONLY = ('adapter_norm', 'grad_norm_mean', 'hessian_top_eig', 'hessian_lambda_max', 'hessian_lambda_min')


def _by_lr(runs_per_rep, key):
    """{lr: [value per repetition]} from p['full'] or p['head'] (list over reps of list over LRs)."""
    d = defaultdict(list)
    for reps in runs_per_rep:
        for r in reps:
            d[r['lr']].append(r[key])
    return d


def _budget_loss(r):
    """Random-label train loss after the first quarter of the probe budget (S/4 steps). With constant-LR Adam and a
    fixed data order this equals the final loss of a probe trained for S/4 steps, i.e. the standard fixed-budget
    memorisation measure (Lyle et al. 2022, 2023) at a budget where it has not saturated. It is evaluated on all
    --ap_rand_n images (the curve subset is the first 2,000, and ap_rand_n = 2,000)."""
    return float(r['curve'][0]['train_loss'])


def _pick(runs_per_rep, nat):
    """One LR per probe and point, by the best mean over repetitions: validation accuracy (ties: lower validation
    loss) for natural probes, train loss at the S/4 budget for randlabel. Picking per repetition on 200 validation
    images selects noise when accuracy is flat across LRs, which is what batch A showed."""
    if nat:
        acc, loss = _by_lr(runs_per_rep, 'val_acc'), _by_lr(runs_per_rep, 'val_loss')
        return max(acc, key=lambda lr: (round(float(np.mean(acc[lr])), 6), -float(np.mean(loss[lr]))))
    b = defaultdict(list)
    for reps in runs_per_rep:
        for r in reps:
            b[r['lr']].append(_budget_loss(r))
    return min(b, key=lambda lr: float(np.mean(b[lr])))


def _value(runs_per_rep, lr, nat):
    rs = [r for reps in runs_per_rep for r in reps if r['lr'] == lr]
    if not rs:
        return None, None
    if nat:
        return float(np.mean([r['test_acc'] for r in rs])), None
    return float(np.mean([r['train_loss'] for r in rs])), float(np.mean([_budget_loss(r) for r in rs]))


def _rep_sd(runs_per_rep, lr, nat):
    """Spread over repetitions at one LR (the noise floor: 5 repetitions at t = 0, 2 later), as a fraction of the mean,
    so it is on the same scale as a ratio. Natural probes: test accuracy; randlabel: the S/4 budget loss."""
    v = [r['test_acc'] if nat else _budget_loss(r) for reps in runs_per_rep for r in reps if r['lr'] == lr]
    if len(v) < 2 or not np.mean(v):
        return None
    return float(np.std(v, ddof=1) / abs(np.mean(v)))


def probe_values(p, lr0=None):
    """Probe measures at one point. lr0: {probe_kind: LR picked at t = 0 in the same run} for the fixed-LR variant."""
    out, picked = {}, {}
    for name in NATURAL + ('randlabel',):
        if name not in p or not isinstance(p[name], dict):
            continue
        nat = name in NATURAL
        for kind in ('full', 'head'):
            runs = p[name][kind]
            grid = sorted({r['lr'] for reps in runs for r in reps})
            lr = _pick(runs, nat)
            picked[f'{name}_{kind}'] = lr
            v, auc = _value(runs, lr, nat)
            out[f'probe_{name}_{kind}'] = v
            if auc is not None:
                out[f'probe_{name}_{kind}_q1'] = auc
            out[f'probe_{name}_{kind}_lr_at_edge'] = float(lr in (grid[0], grid[-1]))
            out[f'probe_{name}_{kind}_repsd'] = _rep_sd(runs, lr, nat)
            if lr0 and f'{name}_{kind}' in lr0:
                fv, fauc = _value(runs, lr0[f'{name}_{kind}'], nat)
                if fv is not None:
                    out[f'probe_{name}_{kind}_fixedlr'] = fv
                    if fauc is not None:
                        out[f'probe_{name}_{kind}_fixedlr_q1'] = fauc
        out[f'probe_{name}_erank'] = float(p[name]['erank'])
    return out, picked


rows = []
runs = {}
# one result per run name; if a run exists under several code versions, the most recently written file is used
files = {}
for f in glob.glob(os.path.join(args.out, 'runs', '*', 'study_*.json')):
    if f.endswith('.flags.json'):
        continue
    name = os.path.basename(f)[:-5]
    if name not in files or os.path.getmtime(f) > os.path.getmtime(files[name]):
        files[name] = f
code_of = {}
for name, f in sorted(files.items()):
    r = json.load(open(f))
    if not r.get('complete'):
        continue
    code_of[name] = os.path.basename(os.path.dirname(f))
    m = re.match(r'study_(.+)_order(\d)_s(\d)', name)
    arm, order = m.group(1), int(m.group(2))
    runs[name] = (arm, order, r)
    pr = r.get('probes', {})
    probes, lr0 = {}, None
    if '0' in pr:
        probes[0], lr0 = probe_values(pr['0'])
    for t, v in pr.items():
        if t != '0':
            probes[int(t)] = probe_values(v, lr0)[0]
    base = probes.get(0, {})
    pts = [(0, dict(dW=0.0, **{k: v for k, v in r.get('metrics_t0', {}).items() if k in INTERNAL}))]
    for row in r['tasks']:
        t = row['t']
        d = dict(dW=row['rel_dW'], A_t=row['A_t'], newtask_cil=row['test_cil'][-1], newtask_til=row['test_til'][-1])
        if 'rel_dW_all' in row:
            d['rel_dW_all'] = row['rel_dW_all']
        d.update({k: v for k, v in row.get('metrics', {}).items() if k in INTERNAL or k == 'dW_step'})
        if row.get('loss_progress'):
            d['lp_rel'] = row['loss_progress']['lp_rel_mean']
        pts.append((t, d))
    for t, d in pts:
        if t in probes:
            d.update(probes[t])
            for k, v in probes[t].items():
                if k.endswith(('_lr_at_edge', '_erank', '_repsd')) or v is None:
                    continue
                b = base.get(k.replace('_fixedlr', ''))     # the fixed-LR value at t = 0 is the picked value itself
                if b:
                    d[k + '_ratio'] = v / b
        for k in [k for k in d if k.startswith('hessian_lambda_') and not k.endswith('_resid')]:
            if d.get(k + '_resid', 0.0) > RESID_MAX:            # power iteration not converged: excluded
                d[k] = None
        for k, v in d.items():
            if v is None:
                continue
            rows.append(dict(run=name, arm=arm, order=order, t=t, measure=k, value=v))

with open(os.path.join(OUT, 'study_long.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['run', 'arm', 'order', 't', 'measure', 'value'])
    w.writeheader()
    w.writerows(rows)

# aggregate: arm x t x measure -> values over orders
agg = defaultdict(list)
for x in rows:
    agg[(x['arm'], x['t'], x['measure'])].append(x['value'])
def setting(arm):
    """plain = plain (single-task) training; full = full-model continual learning; adapter = adapter continual learning."""
    return 'plain' if arm.startswith('plain_') else ('full' if arm.startswith('full') else 'adapter')


def arm_key(arm):
    m = re.search(r'lr([0-9.]+e-?[0-9]+)', arm)
    return (('adapter', 'full', 'plain').index(setting(arm)), arm.replace('plain_', '').split('_')[0],
            float(m.group(1)) if m else 0.0, arm)


SETTING_TITLE = {'adapter': 'Setting 3: adapter continual learning (20 tasks x 5 classes)',
                 'full': 'Setting 2: full-model continual learning (20 tasks x 5 classes)',
                 'plain': 'Setting 1: plain training (one task, all 100 classes; point t = after t/20 of the updates)'}
arms = sorted({x['arm'] for x in rows}, key=arm_key)
measures = sorted({x['measure'] for x in rows})


def cell(arm, t, meas, fmt='{:.3g}'):
    v = agg.get((arm, t, meas))
    if not v:
        return ''
    s = fmt.format(np.mean(v))
    return s + (f' ± {np.std(v, ddof=1):.2g}' if len(v) > 1 else '') + (f' (n={len(v)})' if len(v) > 1 else '')


codes = sorted(set(code_of.values()))
L = ['# Study: plasticity of LoRA vs OFT', '',
     'Mean over the class orders available (± sd and n when more than one). Generated by ap_study.py from the result '
     'files; nothing typed by hand.', '',
     'Probes. Natural tasks (DTD, EuroSAT, Flowers102): test accuracy. randlabel: train loss on the 2,000 random-label '
     'images after a fixed budget, S/4 = 150 steps (`_q1`, primary) and S = 600 steps (`_full`, `_head`; saturates near 0 '
     'for the adapter). LR rule: one LR per probe and point, the best mean over repetitions (validation accuracy, ties by '
     'validation loss; randlabel: `_q1` loss). `_fixedlr`: the LR picked at t = 0 in the same run, held fixed at every '
     'later point. `_ratio`: value at t / value at t = 0 (the fresh adapter on the same frozen backbone). For randlabel '
     'a ratio above 1 means harder to fit (less plastic). `_repsd`: standard deviation over repetitions at the picked '
     'LR divided by the mean (relative noise; at t = 0 it is over 5 repetitions and is the noise floor for every '
     '`_ratio`: a ratio within about 2 x repsd(t = 0) of 1 is not distinguishable from no change).', '',
     f'Hessian: `hessian_lambda_max` is the most positive eigenvalue (sharpness), `hessian_lambda_min` the most negative; '
     f'values whose relative residual exceeds {RESID_MAX} are excluded as not converged. `hessian_top_eig` is the '
     'largest-magnitude eigenvalue (either sign), the only Hessian value recorded in batch A.', '',
     'Within one adapter type only (raw scale depends on the parameterisation, so LoRA and OFT values are not '
     'comparable): ' + ', '.join(f'`{m}`' for m in WITHIN_ADAPTER_ONLY) + '.', '',
     f'Runs: {len(runs)} complete, code versions {", ".join(codes)}. Arms: {", ".join(arms)}.', '']
for st in ('adapter', 'full', 'plain'):
    sarms = [a for a in arms if setting(a) == st]
    if not sarms:
        continue
    L += [f'# {SETTING_TITLE[st]}', '']
    if st == 'plain':
        L += ['In plain runs `newtask_cil` / `newtask_til` and `A_t` are accuracy on all 100 classes (there is one task).', '']
    if st == 'full':
        L += ['dW for the full model is measured on the attention matrices the adapters modify (qkv, proj); '
              '`rel_dW_all` covers every weight matrix. Probe LRs are 3e-6 to 1e-4 (full fine-tuning).', '']
    for t in (0, 5, 10, 20):
        present = [m for m in measures if any((a, t, m) in agg for a in sarms)]
        if not present:
            continue
        L += [f'## t = {t}', '', '| measure | ' + ' | '.join(sarms) + ' |', '|---|' + '---|' * len(sarms)]
        for meas in present:
            L.append(f'| {meas} | ' + ' | '.join(cell(a, t, meas) for a in sarms) + ' |')
        L.append('')
edge = [x for x in rows if x['measure'].endswith('_lr_at_edge') and x['value'] > 0]
if edge:
    L += [f'Note: in {len(edge)} of {sum(1 for x in rows if x["measure"].endswith("_lr_at_edge"))} probe measurements '
          'the picked LR is at the edge of the probe grid.', '']
open(os.path.join(OUT, 'STUDY.md'), 'w').write('\n'.join(L) + '\n')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot = [m for m in measures if not m.endswith(('_lr_at_edge', '_resid', '_repsd')) and m != 'dW']
    for meas in plot:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for arm in arms:
            ts = sorted({x['t'] for x in rows if x['arm'] == arm and x['measure'] == meas})
            if not ts:
                continue
            y = [np.mean(agg[(arm, t, meas)]) for t in ts]
            dw = [np.mean(agg.get((arm, t, 'dW'), [np.nan])) for t in ts]
            style = {'adapter': '-' if 'lora' in arm else '--', 'full': '-.', 'plain': ':'}[setting(arm)]
            a1.plot(ts, y, style, marker='o', ms=3, label=arm)
            a2.plot(dw, y, style, marker='o', ms=3, label=arm)
        a1.set_xlabel('task t')
        a2.set_xlabel('dW (relative weight change)')
        a1.set_ylabel(meas)
        a1.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'fig_{meas}.png'), dpi=120)
        plt.close(fig)
except ImportError:
    pass
print(f'{len(runs)} runs, {len(rows)} rows -> {OUT}/STUDY.md, study_long.csv, fig_*.png')
