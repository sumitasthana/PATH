"""The correlation test again, this time through matched BatchNorm statistics.

The paper's Table 4 and Table D2 correlate each indicator's movement over the
stream with the probe's movement over the same stream. Every indicator value in
them is read through the fixed protocol: a mean of zero and a variance of one at
every BatchNorm layer, never updated. Notebook 55 read all fifteen streams a
second way, through statistics measured from each model's own weights, which is
what this script correlates instead.

The probe side does not change. It is the same selection rule and the same
eleven repetitions, imported from the script that produced the published
columns, so the two routes differ only in how the indicator was read.

Change is reported as an absolute difference, not a percent one. Through
matched statistics the dormant fraction is exactly 0.000 at boundary 0 in all
fifteen streams, so a percent change has nothing to divide by. The fixed route
is reported both ways, and its percent column is the one that ties back to the
published Table D2.

No GPU, no network. Reads the local copy of notebook 55's output.

Run it with: python scripts/upgrade_suite/matched_correlations.py
"""
import json
from pathlib import Path
from statistics import mean

from seed_clustered_correlations import (CRITICAL_R, ORDERS, SEEDS, by_seed,
                                         pearson, probes)

REPO = Path(__file__).resolve().parent.parent.parent
MATCHED = REPO / "outputs/revised_log_audit/matched_statistics_v1.jsonl"

# Each indicator and the sign the loss-of-plasticity reading predicts. The
# probe improves over these streams, so an indicator that tracks the ability to
# learn should rise with it; dormant units are counted the other way up.
MEASURES = [("effective_rank", "Effective rank", "positive"),
            ("srank", "srank", "positive"),
            ("dormant_0.01", "Dormant units, cut-off 0.01", "negative"),
            ("dormant_0.025", "Dormant units, cut-off 0.025", "negative"),
            ("dormant_0.1", "Dormant units, cut-off 0.1", "negative")]

KEYS = [(order, seed) for order in ORDERS for seed in SEEDS]


def readings(route):
    """Absolute and percent change over the stream, and the end value."""
    rows = [json.loads(line) for line in
            MATCHED.read_text(encoding="utf-8").splitlines() if line.strip()]
    at = {(r["order"], r["seed"], r["point"]): r for r in rows
          if r["buffers"] == route and r["images"] == "in_stream"}
    change, percent, end = {}, {}, {}
    for key in KEYS:
        begin, finish = at[(*key, 0)], at[(*key, 9)]
        change[key] = {f: finish[f] - begin[f] for f, _, _ in MEASURES}
        percent[key] = {f: (100.0 * (finish[f] - begin[f]) / begin[f]
                            if begin[f] else float("nan"))
                        for f, _, _ in MEASURES}
        end[key] = {f: finish[f] for f, _, _ in MEASURES}
    return change, percent, end


def column(values, probe):
    """Pearson r at both counts, for one set of indicator values."""
    wide = pearson([values[k] for k in KEYS], [probe[k] for k in KEYS])
    tight = pearson(by_seed(values), by_seed(probe))
    return wide, tight


def table(title, pairs, probe):
    """One table. `pairs` is a list of (column name, values by measure)."""
    print()
    print(title)
    head = f"{'Measure':<30}{'predicted':>11}"
    for name, _ in pairs:
        head += f"{name + ', N=15':>18}{name + ', N=5':>17}"
    print(head)
    for field, label, sign in MEASURES:
        want = 1 if sign == "positive" else -1
        line = f"{label:<30}{sign:>11}"
        for _, values in pairs:
            wide, tight = column({k: values[k][field] for k in KEYS}, probe)
            mark = "" if tight * want > 0 else " x"
            line += f"{wide:>+18.3f}{tight:>+15.3f}{mark:<2}"
        print(line)
    print(f"  An x marks a correlation whose sign is the opposite of the one "
          f"predicted, at five seeds.")
    print(f"  Separable from zero needs r above {CRITICAL_R[13]:.3f} at N=15 "
          f"and {CRITICAL_R[3]:.3f} at N=5.")


def _check():
    """One runnable check: the fixed route here must match the published log.

    Notebook 55 re-read every stream from the saved checkpoints. Those readings
    are forward passes rather than fresh training, so the fixed route should
    land on the values the study already published. If it does not, this
    script is correlating something other than what the paper reports.
    """
    published = REPO / "outputs/revised_log_audit/uncle/indicators/plasticity_revised_v1"
    _, _, end = readings("start")
    worst = 0.0
    for order, seed in KEYS:
        path = published / order / f"seed_{seed}.jsonl"
        rows = {r["t"]: r for r in (json.loads(l) for l in
                                    path.read_text().splitlines() if l.strip())}
        was = rows[9]["effective_rank_in_stream"]
        got = end[(order, seed)]["effective_rank"]
        worst = max(worst, abs(got / was - 1.0))
    assert worst < 0.01, f"fixed route is off the published log by {100*worst:.2f}%"
    return worst


if __name__ == "__main__":
    off = _check()
    print(f"Check passed: the fixed route reproduces the published log, "
          f"worst stream off by {100 * off:.3f}%.")

    probe_change, probe_end = probes()
    fixed_change, fixed_percent, fixed_end = readings("start")
    own_change, _, own_end = readings("own")

    table("CHANGE TEST: indicator movement over the stream against probe gain",
          [("fixed", fixed_change), ("matched", own_change)], probe_change)
    table("END-VALUE TEST: indicator at boundary 9 against probe at boundary 9",
          [("fixed", fixed_end), ("matched", own_end)], probe_end)

    print()
    print("The fixed route with percent change, which is how Table D2 reports it:")
    for field, label, _ in MEASURES:
        wide, tight = column({k: fixed_percent[k][field] for k in KEYS},
                             probe_change)
        print(f"  {label:<30}{wide:>+10.3f}{tight:>+10.3f}")

    print()
    print("How many streams move the way the plasticity reading predicts:")
    print(f"{'Measure':<30}{'fixed':>12}{'matched':>12}")
    for field, label, sign in MEASURES:
        want = 1 if sign == "positive" else -1
        line = f"{label:<30}"
        for values in (fixed_change, own_change):
            agree = sum((values[k][field] > 0) == (want > 0) for k in KEYS)
            line += f"{str(agree) + ' of 15':>12}"
        print(line)
