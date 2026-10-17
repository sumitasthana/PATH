"""
Driver for the calibrated pipeline. Every training run comes from a preset in presets.yaml; there are no
free-form training flags. Gates are recorded in <out>/gates.json together with the code hash they were
passed with; a gate passed with different code counts as not passed.

Phases, in order:
  gate0   : CPU test suite (tests/test_gate0.py). Also run by the installer.
  prepare : download CIFAR-100 to --data and verify 500 train / 100 test images per class.
  gate1   : read <out>/gate1_l2p.log (unmodified Mammoth L2P) and check it against Mammoth's 83.83 +- 2.0.
  gate3   : our LAE-baseline preset on the 3 LAE class orders; pass = mean A_T in 83.24 +- 2.0 and
            mean A_bar in 88.48 +- 1.5 (Gao et al. 2023, Table 1). Requires gate0.
  gate3b  : joint-training sanity check with the same model; pass = A_T in [88, 93]. Requires gate0.
  step1   : protocol step 1, backbone weights orig_in21k -> augreg_in21k, nothing else changed.
            Requires gate3 + gate3b. Pass = all checks + |mean A_T - gate3 mean A_T| <= 5.
  step2a  : protocol step 2a: LAE KVLoRA (K,V rows of qkv, blocks 0-4, r5, learnable scales) -> PEFT LoRA r5
            on qkv (Q,K,V), blocks 0-4, LR from a validation search. Requires step1.
            Pass = all checks + |dA_T vs step1| <= 5.
  step2b  : protocol step 2b, placement: PEFT LoRA r8 on qkv + proj of all 12 blocks (the study placement).
            LR selected by a validation search (order 1, 10% of each class held out of train, test set unused);
            the grid includes LAE's 0.0028125 and the selected LR must not be at a grid edge. Then the 3 orders
            on the test set at the selected LR. Requires step2a. Pass = all checks + |dA_T vs step1| <= 5.
  calibrate: gate0, gate3, gate3b, step1, step2a, step2b in order; gates already passed with this code are skipped;
            stops at the first failure.
  status  : print the gate table.
  report  : write <out>/REPRODUCTION.md from gates.json and the run files of the current code version.
            Every number in it is read from the result files; nothing is typed by hand.

Runs write locally (--local), then each result is copied to <out>/runs/<code hash>/ and verified by SHA-256.
Results are kept per code version, so a code change never mixes old and new runs. Finished runs of the
current code are skipped, so any phase can be re-launched after a disconnect.

Usage (Colab):  !python /content/drive/MyDrive/ap/ap_run.py --phase gate3
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--phase', required=True, choices=['gate0', 'prepare', 'gate1', 'gate3', 'gate3b', 'step1', 'step2a', 'step2b', 'calibrate', 'status', 'report'])
ap.add_argument('--mammoth', default='/content/mammoth')
ap.add_argument('--out', default=HERE)
ap.add_argument('--data', default='/content/apdata')
ap.add_argument('--local', default='/content/ap_local')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--dry_run', action='store_true')
args = ap.parse_args()

PRESETS = yaml.safe_load(open(os.path.join(HERE, 'presets.yaml')))
GATES_CFG = PRESETS.pop('gates')
GATES = os.path.join(args.out, 'gates.json')
REQUIRED_TIMM = '1.0.30'


# ---------------------------------------------------------------- code identity
def code_hash():
    """SHA-256 over every plugin file in the Mammoth copy (the code that produces numbers).

    The driver and presets.yaml are not part of it: each gate also records the hash of its own preset and
    gate rule, and each run stores the exact flags it was launched with, so editing an unrelated preset or
    the driver never invalidates a passed gate, while editing a run's own definition always does.
    """
    man = json.load(open(os.path.join(HERE, 'plugin_manifest.json')))
    h = hashlib.sha256()
    for rel in sorted(man):
        p = os.path.join(args.mammoth, rel)
        data = open(p, 'rb').read()
        if hashlib.sha256(data).hexdigest() != man[rel]:
            sys.exit(f'ABORT: {p} differs from the installed version. Re-run the installer.')
        h.update(rel.encode() + data)
    return h.hexdigest()[:16]


def def_hash(name):
    """Hash of a gate's rule plus the preset(s) it runs."""
    cfg = GATES_CFG.get(name, {})
    obj = dict(rule=cfg, preset=PRESETS.get(cfg.get('preset'), {}))
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def check_env():
    import importlib.metadata as md
    v = md.version('timm')
    if v != REQUIRED_TIMM:
        sys.exit(f'ABORT: timm {v} installed, pipeline requires {REQUIRED_TIMM}. Re-run the installer.')


def load_gates():
    return json.load(open(GATES)) if os.path.exists(GATES) else {}


def record_gate(name, passed, detail):
    g = load_gates()
    g[name] = dict(passed=bool(passed), code='n/a' if name == 'gate1' else CODE, defn=def_hash(name), time=time.strftime('%Y-%m-%d %H:%M:%S'), **detail)
    tmp = GATES + '.tmp'
    json.dump(g, open(tmp, 'w'), indent=1)
    os.replace(tmp, GATES)
    print(f'\n{name}: {"PASS" if passed else "FAIL"}  ' + json.dumps(detail), flush=True)


def require(*names):
    g = load_gates()
    for n in names:
        r = g.get(n)
        if not r or not r['passed'] or r['code'] != CODE or r.get('defn') != def_hash(n):
            why = 'not run' if not r else ('failed' if not r['passed'] else 'passed with different code or definition')
            sys.exit(f'ABORT: {n} {why}. Run --phase {n} first.')


# ---------------------------------------------------------------- runs
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def copy_verified(src, dst):
    tmp = dst + '.part'
    shutil.copyfile(src, tmp)
    if sha(tmp) != sha(src):
        os.remove(tmp)
        raise IOError(f'checksum mismatch copying {src}')
    os.replace(tmp, dst)


def run_flags(preset, run, extra=None):
    return {**PRESETS[preset]['mammoth'], **{k: v for k, v in run.items() if k != 'name'}, **(extra or {})}


def result(name, flags=None):
    """Completed result of this run, or None. With flags: only if the run was launched with exactly these flags."""
    p = os.path.join(RUNS, name + '.json')
    if not os.path.exists(p):
        return None
    try:
        r = json.load(open(p))
        if flags is not None and json.load(open(os.path.join(RUNS, name + '.flags.json'))) != json.loads(json.dumps(flags)):
            return None
    except Exception:
        return None
    return r if r.get('complete') else None


def launch(preset, run, extra=None):
    name = run['name']
    flags = run_flags(preset, run, extra)
    if result(name, flags):
        print(f'skip {name} (complete)', flush=True)
        return
    cmd = [sys.executable, 'main.py']
    for k, v in flags.items():
        cmd += [f'--{k}', str(v)]
    cmd += ['--num_workers', str(args.workers), '--non_verbose', '1', '--base_path', os.path.join(args.data, 'mammoth') + '/',
            '--ap_out', LRUNS, '--ap_run_name', name, '--ap_preset', preset]
    print(f'\n=== {name}\n' + ' '.join(cmd), flush=True)
    if args.dry_run:
        return
    t0 = time.time()
    llog = os.path.join(LLOGS, name + '.log')
    with open(llog, 'w') as log:
        rc = subprocess.call(cmd, cwd=args.mammoth, stdout=log, stderr=subprocess.STDOUT)
    copy_verified(llog, os.path.join(LOGS, name + '.log'))
    lres = os.path.join(LRUNS, name + '.json')
    if os.path.exists(lres):
        copy_verified(lres, os.path.join(RUNS, name + '.json'))
        with open(os.path.join(RUNS, name + '.flags.json'), 'w') as f:
            json.dump(flags, f, indent=1)
    r = result(name, flags)
    if rc != 0 or r is None:
        tail = open(llog).read()[-2500:]
        sys.exit(f'{name}: FAILED (exit {rc}) after {(time.time() - t0) / 60:.1f} min. Log tail:\n{tail}')
    print(f'{name}: ok in {(time.time() - t0) / 60:.1f} min   A_T {r["A_T"]:.2f}   A_bar {r["A_bar"]:.2f}', flush=True)


def summarize(preset, runs=None, extra=None):
    runs = runs or PRESETS[preset]['runs']
    rows = []
    for run in runs:
        r = result(run['name'], run_flags(preset, run, extra))
        if r is None:
            sys.exit(f'ABORT: {run["name"]} missing or incomplete.')
        bad = [k for k, c in r['checks'].items() if not c['ok']]
        if bad:
            sys.exit(f'ABORT: {run["name"]} has failed checks {bad}.')
        if r['config']['ap_preset'] != preset:
            sys.exit(f'ABORT: {run["name"]} was produced by preset {r["config"]["ap_preset"]}, not {preset}.')
        rows.append(r)
    ref = PRESETS[preset]['reference']
    print(f'\n{preset}: {PRESETS[preset]["description"]}')
    print(f'{"run":28s} {"A_T":>7s} {"A_bar":>7s}  per-task A_t')
    for run, r in zip(runs, rows):
        print(f'{run["name"]:28s} {r["A_T"]:7.2f} {r["A_bar"]:7.2f}  ' + ' '.join(f'{t["A_t"]:.1f}' for t in r['tasks']))
    A_T = [r['A_T'] for r in rows]
    A_bar = [r['A_bar'] for r in rows]
    sd = (lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float('nan'))
    print(f'{"ours (mean +- sd)":28s} {np.mean(A_T):7.2f} {np.mean(A_bar):7.2f}   sd {sd(A_T):.2f} / {sd(A_bar):.2f}')
    print(f'{"published":28s} {ref.get("A_T", float("nan")):7.2f} {ref.get("A_bar", float("nan")):7.2f}   ({ref["source"]})')
    return float(np.mean(A_T)), float(np.mean(A_bar)), A_T, A_bar


def inside(x, band):
    return band[0] <= x <= band[1]


# ---------------------------------------------------------------- phases
CODE = code_hash()
RUNS, LOGS = os.path.join(args.out, 'runs', CODE), os.path.join(args.out, 'logs', CODE)
LRUNS, LLOGS = os.path.join(args.local, 'runs', CODE), os.path.join(args.local, 'logs', CODE)
for d in (RUNS, LOGS, LRUNS, LLOGS):
    os.makedirs(d, exist_ok=True)
ORDER = ['gate0', 'gate3', 'gate3b', 'step1', 'step2a', 'step2b']


def passed(name):
    r = load_gates().get(name)
    return bool(r and r['passed'] and r['code'] in (CODE, 'n/a') and r.get('defn') == def_hash(name))


def lr_search(preset, sc):
    """Train the full task sequence once per LR on class order sc['order'] with sc['validation']% of every
    class held out from the training set; select the LR with the highest final validation accuracy A_T.
    The test set is never used for selection."""
    rows = []
    for lr in sc['lrs']:
        run = {'name': f'{preset}_search_lr{lr:g}', 'ap_class_order': sc['order'], 'seed': sc['seed']}
        extra = {'lr': lr, 'validation': sc['validation']}
        launch(preset, run, extra)
        if args.dry_run:
            continue
        r = result(run['name'], run_flags(preset, run, extra))
        assert r['eval_split'].startswith('validation'), 'search run evaluated on the test set'
        rows.append(dict(lr=lr, val_A_T=r['A_T'], val_A_bar=r['A_bar'], final_rel_dW=r['tasks'][-1]['rel_dW']))
    if args.dry_run:
        return None
    best = max(rows, key=lambda x: x['val_A_T'])
    print(f'\nLR search for {preset} (order {sc["order"]}, {sc["validation"]}% validation held out of train):')
    print(f'{"lr":>10s} {"val A_T":>8s} {"val A_bar":>9s} {"final dW":>9s}')
    for x in rows:
        print(f'{x["lr"]:10g} {x["val_A_T"]:8.2f} {x["val_A_bar"]:9.2f} {x["final_rel_dW"]:9.4f}' + ('   <- selected' if x is best else ''))
    lrs = sorted(sc['lrs'])
    return dict(lrs=lrs, table=rows, selected_lr=best['lr'], at_edge=best['lr'] in (lrs[0], lrs[-1]))


def run_gate0():
    check_env()
    r = subprocess.run([sys.executable, 'tests/test_gate0.py'], cwd=args.mammoth, capture_output=True, text=True,
                       env={**os.environ, 'AP_FAKE_DATA': '1'})
    out = r.stdout + r.stderr
    open(os.path.join(LOGS, 'gate0.log'), 'w').write(out)
    print(out[-4000:])
    ok = r.returncode == 0 and 'GATE0 PASS' in out
    record_gate('gate0', ok, dict(log=f'logs/{CODE}/gate0.log'))
    return ok


def run_preset_gate(name):
    check_env()
    cfg = GATES_CFG[name]
    require(*cfg['requires'])
    preset = cfg['preset']
    runs, extra, detail, ok = PRESETS[preset]['runs'], None, {}, True
    if 'lr_search' in cfg:
        sel = lr_search(preset, cfg['lr_search'])
        if sel is None:
            return True                                   # dry run
        detail['lr_search'] = sel
        if sel['at_edge']:
            print(f'STOP: best LR {sel["selected_lr"]:g} is at the edge of the grid {sel["lrs"]}; extend the grid.')
            record_gate(name, False, detail)
            return False
        extra = {'lr': sel['selected_lr']}
        runs = [{**r, 'name': f'{r["name"]}_lr{sel["selected_lr"]:g}'} for r in runs]
    for run in runs:
        launch(preset, run, extra)
    if args.dry_run:
        return True
    A_T, A_bar, all_T, all_bar = summarize(preset, runs, extra)
    detail.update(preset=preset, A_T=A_T, A_bar=A_bar, per_run_A_T=all_T, per_run_A_bar=all_bar)
    if 'A_T' in cfg:
        ok &= inside(A_T, cfg['A_T'])
        detail['band_A_T'] = cfg['A_T']
    if 'A_bar' in cfg:
        ok &= inside(A_bar, cfg['A_bar'])
        detail['band_A_bar'] = cfg['A_bar']
    if 'vs' in cfg:
        prev = load_gates()[cfg['vs']]
        d_T, d_bar = A_T - prev['A_T'], A_bar - prev['A_bar']
        ok &= abs(d_T) <= cfg['tol_A_T']
        detail.update(vs=cfg['vs'], delta_A_T=d_T, delta_A_bar=d_bar, tol_A_T=cfg['tol_A_T'])
        print(f'change vs {cfg["vs"]}: A_T {d_T:+.2f}   A_bar {d_bar:+.2f}   (allowed |dA_T| <= {cfg["tol_A_T"]})')
    record_gate(name, ok, detail)
    return ok


if args.phase == 'gate0':
    run_gate0()

elif args.phase == 'prepare':
    import torchvision
    root = os.path.join(args.data, 'mammoth', 'CIFAR100')
    for train, n in ((True, 500), (False, 100)):
        ds = torchvision.datasets.CIFAR100(root, train=train, download=True)
        c = np.bincount(ds.targets, minlength=100)
        assert (c == n).all(), f'CIFAR-100 {"train" if train else "test"} counts wrong: {c}'
        print(f'CIFAR-100 {"train" if train else "test"}: {len(ds)} images, {n} per class: ok')

elif args.phase == 'gate1':
    cfg = GATES_CFG['gate1']
    log = os.path.join(args.out, cfg['log'])
    txt = open(log).read()
    m = re.findall(r'Accuracy for 10 task\(s\):\s*\[Class-IL\]:\s*([\d.]+)', txt)
    if not m:
        sys.exit(f'ABORT: no "Accuracy for 10 task(s)" line in {log}; the run did not finish.')
    acc = float(m[-1])
    record_gate('gate1', abs(acc - cfg['target']) <= cfg['tol'],
                dict(class_il=acc, target=cfg['target'], tol=cfg['tol'], source=cfg['source']))

elif args.phase in ('gate3', 'gate3b', 'step1', 'step2a', 'step2b'):
    run_preset_gate(args.phase)

elif args.phase == 'calibrate':
    for name in ORDER:
        if passed(name):
            print(f'{name}: already passed with code {CODE}, skipped', flush=True)
            continue
        ok = run_gate0() if name == 'gate0' else run_preset_gate(name)
        if not ok and not args.dry_run:
            sys.exit(f'STOP: {name} failed. Paste the table above.')
    print('\ncalibrate: all gates passed with code ' + CODE)

if args.phase in ('status', 'calibrate'):
    g = load_gates()
    print(f'\ncode {CODE}')
    for n in ['gate0', 'gate1'] + ORDER[1:]:
        r = g.get(n)
        if not r:
            print(f'{n:7s} not run')
            continue
        stale = '' if passed(n) or not r['passed'] else '  (STALE: passed with different code or definition)'
        keys = ('A_T', 'A_bar', 'delta_A_T', 'class_il')
        extra = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in keys}
        if 'lr_search' in r:
            extra['selected_lr'] = r['lr_search']['selected_lr']
        print(f'{n:7s} {"PASS" if r["passed"] else "FAIL"}  {r["time"]}{stale}  {json.dumps(extra)}')


# ---------------------------------------------------------------- report
GATE_TITLES = {
    'gate0': 'Gate 0: CPU test suite',
    'gate1': 'Gate 1: environment reproduces a published Mammoth number (L2P)',
    'gate3': 'Gate 3: our code reproduces the LAE LoRA baseline',
    'gate3b': 'Gate 3b: joint-training ceiling with the same model',
    'step1': 'Step 1: augreg backbone weights',
    'step2a': 'Step 2a: PEFT LoRA on LAE placement',
    'step2b': 'Step 2b: PEFT LoRA on the study placement',
}


def write_report():
    g = load_gates()
    L = ['# Pipeline calibration: reproduction report', '',
         f'Code version (SHA-256 of all training code, first 16 hex): `{CODE}`. '
         'Every number below is read from the result files of this code version by `ap_run.py --phase report`.', '']
    env = None
    for d in sorted(glob.glob(os.path.join(RUNS, '*.json'))):
        if not d.endswith('.flags.json'):
            env = json.load(open(d)).get('env')
            break
    if env:
        L += ['Environment: ' + ', '.join(f'{k} {v}' for k, v in env.items()), '']
    L += ['| Gate | Status | Result | Rule |', '|---|---|---|---|']
    for n in ['gate0', 'gate1'] + ORDER[1:]:
        r = g.get(n)
        if not r:
            L.append(f'| {GATE_TITLES[n]} | not run | | |')
            continue
        st = 'PASS' if passed(n) else ('FAIL' if not r['passed'] else 'stale (other code version)')
        if n == 'gate0':
            lg = os.path.join(LOGS, 'gate0.log')
            m = re.findall(r'Ran (\d+) tests', open(lg).read()) if os.path.exists(lg) else []
            res, rule = (f'{m[-1]} tests' if m else 'see logs/gate0.log'), 'all tests pass'
        elif n == 'gate1':
            res, rule = f"class-IL {r['class_il']:.2f}", f"{r['target']} +- {r['tol']} ({r['source']})"
        else:
            res = f"A_T {r['A_T']:.2f}, A_bar {r['A_bar']:.2f}"
            cfg = GATES_CFG[n]
            rule = []
            if 'A_T' in cfg:
                rule.append(f"A_T in {cfg['A_T']}")
            if 'A_bar' in cfg:
                rule.append(f"A_bar in {cfg['A_bar']}")
            if 'vs' in cfg:
                rule.append(f"abs(A_T - {cfg['vs']}) <= {cfg['tol_A_T']} (observed {r['delta_A_T']:+.2f})")
            if 'lr_search' in r:
                res += f", LR {r['lr_search']['selected_lr']:g} (validation search)"
            rule = '; '.join(rule)
        L.append(f'| {GATE_TITLES[n]} | {st} | {res} | {rule} |')
    L += ['', 'A_T: class-incremental accuracy over all seen classes after the last task. '
          'A_bar: mean of that accuracy over the evaluation points after each task, as defined in LAE (Gao et al. 2023). '
          'Mean over the 3 LAE class orders (seeds 1, 2, 3) unless stated.', '']
    for n in ORDER[1:]:
        r = g.get(n)
        if not r or 'preset' not in r:
            continue
        preset = r['preset']
        P = PRESETS[preset]
        L += [f'## {GATE_TITLES[n]}', '', P['description'], '']
        ref = P.get('reference', {})
        if 'A_T' in ref:
            sd = f" +- {ref['A_T_sd']}" if 'A_T_sd' in ref else ''
            L += [f"Published reference: A_T {ref['A_T']}{sd}" + (f", A_bar {ref['A_bar']} +- {ref.get('A_bar_sd', '')}" if 'A_bar' in ref else '')
                  + f" ({ref['source']}).", '']
        L += ['Configuration: ' + ', '.join(f'{k}={v}' for k, v in P['mammoth'].items() if k not in ('model', 'backbone'))
              + (f", lr={r['lr_search']['selected_lr']:g} (selected)" if 'lr_search' in r else ''), '']
        if 'lr_search' in r:
            ls = r['lr_search']
            L += [f'Learning-rate search (class order 1, seed 1, 10% of every class held out of the training set; test set not used):', '',
                  '| LR | validation A_T | validation A_bar | final dW |', '|---|---|---|---|']
            for x in ls['table']:
                mark = ' (selected)' if x['lr'] == ls['selected_lr'] else ''
                L.append(f"| {x['lr']:g}{mark} | {x['val_A_T']:.2f} | {x['val_A_bar']:.2f} | {x['final_rel_dW']:.4f} |")
            L.append('')
        runs = PRESETS[preset]['runs']
        extra = None
        if 'lr_search' in r:
            lr = r['lr_search']['selected_lr']
            extra = {'lr': lr}
            runs = [{**x, 'name': f"{x['name']}_lr{lr:g}"} for x in runs]
        L += ['| Run | A_T | A_bar | A_t after each task | final dW | checks |', '|---|---|---|---|---|---|']
        for run in runs:
            rr = result(run['name'], run_flags(preset, run, extra))
            if rr is None:
                continue
            nck = sum(c['ok'] for c in rr['checks'].values())
            L.append(f"| {run['name']} | {rr['A_T']:.2f} | {rr['A_bar']:.2f} | "
                     + ' '.join(f"{t['A_t']:.1f}" for t in rr['tasks'])
                     + f" | {rr['tasks'][-1]['rel_dW']:.4f} | {nck}/{len(rr['checks'])} passed |")
        L += ['', f"Mean: A_T {r['A_T']:.2f}, A_bar {r['A_bar']:.2f}. Per-run A_T: "
              + ', '.join(f'{x:.2f}' for x in r['per_run_A_T']) + '.', '']
    L += ['## Checks inside every run', '',
          'Each run aborts if any of these fails, and the result file stores each one: class-order SHA-256 and mapping, '
          'Adam with no weight decay, trainable adapter parameter count equal to the closed-form value, frozen backbone, '
          'adapted matrix count, adapter equal to identity at initialisation (dW = 0), head initialisation N(0, 0.001^2) with '
          'zero bias, per task: exact per-class training image counts, labels inside the current task range, and zero '
          'gradient on every head row outside the current task (local cross-entropy). Each run also stores the SHA-256 of '
          'the frozen backbone weights and the environment.', '']
    out = os.path.join(args.out, 'REPRODUCTION.md')
    open(out, 'w').write('\n'.join(L) + '\n')
    print(f'wrote {out}')


if args.phase == 'report':
    write_report()
