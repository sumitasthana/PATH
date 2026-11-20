"""Table D2 counted two ways: fifteen runs, and five independent seeds.

The study has fifteen runs but only five starting points. Ordering A, B and C
at seed 0 are three streams built from one generator, so their indicator
readings and their probe results are not three independent draws. Counting
them as fifteen makes every correlation look better measured than it is.

This script reports both counts side by side. The fifteen-run column
reproduces Table D2 exactly, which is the check that the pipeline is right.
The five-seed column averages the three orderings of each seed first, then
correlates the five averages.

No GPU, no network. Reads the local copies of the published logs.
"""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent.parent
LOGS = REPO / "outputs/revised_log_audit/uncle"
ORDERS = ("order_A", "order_B", "order_C")
SEEDS = range(5)

# Each indicator, and the sign the loss-of-plasticity reading predicts. A rank
# measure that falls less should leave more ability, so it predicts a positive
# correlation with the probe gain. Dormant units are counted the other way up,
# so they predict a negative one.
MEASURES = [
    ("effective_rank_in_stream", "Effective rank", "positive"),
    ("srank_in_stream", "srank", "positive"),
    ("dormant_0.01_in_stream", "Dormant units, cut-off 0.01", "negative"),
    ("dormant_0.025_in_stream", "Dormant units, cut-off 0.025", "negative"),
    ("dormant_0.1_in_stream", "Dormant units, cut-off 0.1", "negative"),
]

# Two-sided 5 percent critical values of Pearson r. A correlation smaller than
# this cannot be told from zero at that level. Three degrees of freedom is the
# five-seed row and thirteen is the fifteen-run row.
CRITICAL_R = {3: 0.878, 13: 0.514}


def indicators():
    """Percent change from boundary 0 to boundary 9, and the boundary 9 value."""
    change, end = {}, {}
    for order in ORDERS:
        for seed in SEEDS:
            path = LOGS / f"indicators/plasticity_revised_v1/{order}/seed_{seed}.jsonl"
            rows = {r["t"]: r for r in
                    (json.loads(l) for l in path.read_text().splitlines() if l.strip())}
            a, b = rows[0], rows[9]
            change[(order, seed)] = {k: 100.0 * (b[k] - a[k]) / a[k]
                                     for k, _, _ in MEASURES}
            end[(order, seed)] = {k: b[k] for k, _, _ in MEASURES}
    return change, end


def probes():
    """Probe change and boundary 9 accuracy, under the paper's selection rule.

    Inside each repetition the learning rate is chosen on that repetition's own
    validation accuracy, ties going to the lower rate. The run's value is the
    mean over its eleven selected repetitions. This is the rule Table C1 and
    Table D2 use, so the fifteen-run column has to use it too.
    """
    path = LOGS / "matched_probe/plasticity_revised_v1/revised_20class_train4800.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["order"], r["seed_label"], r["point"], r["repetition"])].append(r)
    picked = defaultdict(list)
    for key, arms in cells.items():
        best = min(arms, key=lambda a: (-a["val_accuracy"], a["learning_rate"]))
        picked[key[:3]].append(best["test_accuracy"])
    change, end = {}, {}
    for order in ORDERS:
        for seed in SEEDS:
            at0 = mean(picked[(order, seed, 0)])
            at9 = mean(picked[(order, seed, 9)])
            change[(order, seed)] = at9 - at0
            end[(order, seed)] = at9
    return change, end


def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    top = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    bottom = (sum((x - mx) ** 2 for x in xs)
              * sum((y - my) ** 2 for y in ys)) ** 0.5
    return top / bottom


def by_seed(values):
    """One number per seed: the mean of the three orderings that share it."""
    return [mean([values[(o, s)] for o in ORDERS]) for s in SEEDS]


def _check():
    """One runnable check: the fifteen-run column must reproduce Table D2.

    The two rank rows come back within 0.0002 of the published values. The
    three dormant rows come back within 0.0011, so cut-off 0.025 rounds to
    0.124 here against the published 0.125. The script that first produced
    Table D2 is not in the repository, so the cause of that last digit is not
    established and the published values are left as they are.
    """
    published = {"effective_rank_in_stream": -0.054, "srank_in_stream": -0.178,
                 "dormant_0.01_in_stream": 0.129, "dormant_0.025_in_stream": 0.125,
                 "dormant_0.1_in_stream": 0.112}
    ind, _ = indicators()
    prb, _ = probes()
    keys = [(o, s) for o in ORDERS for s in SEEDS]
    for key, want in published.items():
        got = pearson([ind[k][key] for k in keys], [prb[k] for k in keys])
        assert abs(got - want) < 0.002, (key, got, want)


def table(ind, prb, title, note):
    rows = []
    keys = [(o, s) for o in ORDERS for s in SEEDS]
    for key, label, sign in MEASURES:
        wide = pearson([ind[k][key] for k in keys], [prb[k] for k in keys])
        tight = pearson(by_seed({k: ind[k][key] for k in keys}), by_seed(prb))
        rows.append((label, sign, wide, tight))
    print()
    print(title)
    print(note)
    print(f"{'Measure':<30}{'predicted':>11}{'r, N=15':>10}{'r, N=5':>10}"
          f"{'sign at N=5':>14}")
    for label, sign, wide, tight in rows:
        want = 1 if sign == "positive" else -1
        verdict = "as predicted" if tight * want > 0 else "wrong way"
        print(f"{label:<30}{sign:>11}{wide:>+10.3f}{tight:>+10.3f}{verdict:>14}")
    print(f"  Separable from zero needs r above {CRITICAL_R[13]:.3f} at N=15 "
          f"and {CRITICAL_R[3]:.3f} at N=5.")
    return rows


if __name__ == "__main__":
    _check()
    print("Check passed: the fifteen-run column reproduces Table D2.")
    ind_change, ind_end = indicators()
    prb_change, prb_end = probes()
    table(ind_change, prb_change,
          "CHANGE TEST: indicator change over the stream against probe gain",
          "Both quantities are differences between boundary 0 and boundary 9.")
    table(ind_end, prb_end,
          "END-VALUE TEST: boundary 9 indicator against boundary 9 probe accuracy",
          "No differences are taken, so neither side inherits two endpoints' noise.")
