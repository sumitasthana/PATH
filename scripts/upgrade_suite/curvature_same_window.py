"""The curvature arms over one shared time window.

Table 2 compares a fixed-code arm read at boundaries 0 and 9 against a
trained-code arm read at boundaries 4 and 9. Four things differ between those
two columns at once: the task code, the BatchNorm statistics, the group of
pictures, and the span of stream covered. The last of those is free to remove,
because the fixed arm was also measured at boundary 4 and those 600 readings
are already in the published log.

This script reproduces Tables I1 and I2 as a check, then prints the fixed arm
over boundary 4 to boundary 9, which is the span the trained arm covers.

No GPU, no training. Reads the published curvature log.
"""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, variance

REPO = Path(__file__).resolve().parent.parent.parent
LOG = REPO / "outputs/revised_log_audit/curvature_revised_v1.jsonl"
SEEDS = range(5)

# Each reading point holds 120 measurements: 40 random directions at each of
# three step sizes. The floor is the 95 percent interval for the difference of
# two means built from those two spreads, which is the same rule the paper's
# noise floors use elsewhere.
PER_POINT = 120


def readings():
    """Every measurement, grouped by arm, seed and boundary."""
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    out = defaultdict(list)
    for r in rows:
        out[(r["arm"], r["seed"], r["point"])].append(r)
    return out


def summary(group):
    """Mean curvature and mean loss at one point."""
    assert len(group) == PER_POINT, len(group)
    return (mean(r["curvature"] for r in group),
            mean(r["loss_centre"] for r in group))


def floor(early, late):
    """The 95 percent interval for the difference between two point means."""
    return 1.96 * (variance([r["curvature"] for r in early]) / PER_POINT
                   + variance([r["curvature"] for r in late]) / PER_POINT) ** 0.5


def span(data, arm, early, late, title):
    """One table: each seed's two readings, the change, and its own floor."""
    print()
    print(title)
    print(f"{'seed':>5}{'curvature b' + str(early):>16}"
          f"{'curvature b' + str(late):>16}{'change':>13}{'floor':>12}"
          f"{'verdict':>10}{'loss b' + str(early):>13}{'loss b' + str(late):>13}"
          f"{'ratio':>12}{'curv/loss a':>12}{'curv/loss b':>12}")
    rising = 0
    for seed in SEEDS:
        a, b = data[(arm, seed, early)], data[(arm, seed, late)]
        ca, la = summary(a)
        cb, lb = summary(b)
        f = floor(a, b)
        change = cb - ca
        verdict = "rose" if change > f else "fell" if change < -f else "flat"
        rising += verdict == "rose"
        # The ratio says how many factors of ten the reading moved. Dividing
        # each reading by the loss at the same point says how much of that is
        # the loss scale rather than the shape of the surface.
        print(f"{seed:>5}{ca:>16.4g}{cb:>16.4g}{change:>13.4g}{f:>12.4g}"
              f"{verdict:>10}{la:>13.4g}{lb:>13.4g}"
              f"{cb / ca:>12.4g}{ca / la:>12.4g}{cb / lb:>12.4g}")
    print(f"  {rising} of 5 seeds rise past their own floor.")


def _check():
    """One runnable check: reproduce two published entries of Table I1 and I2."""
    data = readings()
    seed0_b0, _ = summary(data[("reference", 0, 0)])
    assert abs(seed0_b0 - 0.025) < 0.001, seed0_b0
    seed0_b4, _ = summary(data[("learned", 0, 4)])
    assert abs(seed0_b4 - 3.483e-6) < 1e-8, seed0_b4


if __name__ == "__main__":
    _check()
    print("Check passed: Tables I1 and I2 reproduce from the log.")
    data = readings()
    span(data, "reference", 0, 9,
         "FIXED CODE, FIXED STATISTICS, boundary 0 to 9 (Table I1 as published)")
    span(data, "learned", 4, 9,
         "TRAINED CODE, OWN STATISTICS, boundary 4 to 9 (Table I2 as published)")
    span(data, "reference", 4, 9,
         "FIXED CODE, FIXED STATISTICS, boundary 4 to 9 (the shared window, new)")
