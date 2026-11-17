"""Separate learning speed from the ceiling in the probe records.

The study's headline probe number is test accuracy after a fixed budget of 250
steps on a task the model has never seen. That one number mixes two things
that the loss-of-plasticity literature keeps apart:

  speed    how quickly the model fits the new task
  ceiling  how well it can do once it has finished

A model that has lost plasticity should be slower. The usual way to show that
is a learning curve, not a single endpoint. The probe already recorded four
points on that curve for every run, at steps 62, 125, 188 and 250, with the
training loss and the test accuracy at each. Nothing new has to be run.

This script reads those curves and answers three questions.

1. How does the curve move over the stream, step by step?
2. Is the model faster, or only better? The test is the fraction of a run's
   own final accuracy that it has reached by step 62. That fraction cannot be
   raised by a higher ceiling, because each run is divided by its own ceiling.
3. Does the training loss fall faster?

Runs are paired. A boundary 0 run and a boundary 9 run that share an order, a
seed, a learning rate and a repetition started from the same fresh code and saw
the same batches in the same order. They differ only in the model being probed.
Pairing them removes the spread that comes from the starting code, which is the
largest source of noise in this measurement.

Reads the published probe file from R2. No GPU, no training.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "uncle/task_partition_10x20.json").exists()), None)
if REPO is None:
    raise FileNotFoundError("Run this from the repository.")
sys.path.insert(0, str(REPO))

from uncle.storage import R2Store  # noqa: E402

BUCKET = "uncle-experiments"
STUDY_ID = "plasticity_revised_v1"
KEYS = [f"uncle/matched_probe/{STUDY_ID}/revised_20class_train4800.jsonl",
        f"uncle/matched_probe/{STUDY_ID}/revised_20class_train4800_extended.jsonl"]
STEPS = [62, 125, 188, 250]
EARLY = 62


def secret(name):
    """A credential from the environment, or from the repository's .env file."""
    value = os.environ.get(name)
    if value is not None:
        return value
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name + "="):
                return line.partition("=")[2].strip().strip('"').strip("'")
    return None


def load():
    """Every full-arm probe run, as a list of dictionaries."""
    store = R2Store.connect(secret("R2_ENDPOINT"), BUCKET,
                            secret("R2_ACCESS_KEY_ID"),
                            secret("R2_SECRET_ACCESS_KEY"))
    rows = []
    for key in KEYS:
        if store.info(key) is None:
            print(f"  {key.split('/')[-1]}: not in R2, skipped")
            continue
        body = store.client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        here = [json.loads(line) for line in body.decode("utf-8").splitlines()
                if line.strip()]
        print(f"  {key.split('/')[-1]}: {len(here)} runs")
        rows += here
    # The frozen-generator arm reads chance at every point, so it has no curve
    # to compare. Only the full arm answers a question about speed.
    return [r for r in rows if r.get("arm") == "full"]


def curve(row):
    """One run's four readings, keyed by step. Returns None if any is missing."""
    found = {reading["step"]: reading for reading in row.get("readings", [])}
    if not all(step in found for step in STEPS):
        return None
    return found


def identity(row):
    """What makes two runs a matched pair across boundaries."""
    return (row["order"], row["seed_label"], row["learning_rate"],
            row["repetition"])


# Two-sided 95 percent critical values of Student's t, by degrees of freedom.
# With five seeds there are four degrees of freedom and the right value is
# 2.776, not 1.960. Using 1.960 on five numbers makes the interval about 40
# percent too narrow, which is the difference between a result that stands and
# one that does not.
T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
        30: 2.042, 60: 2.000}


def critical(n):
    """The two-sided 95 percent multiplier for a mean of `n` observations."""
    df = n - 1
    if df > 60:
        return 1.960
    if df in T_95:
        return T_95[df]
    # Between tabulated rows, take the next lower df. That widens the interval
    # slightly rather than narrowing it, which is the safe direction.
    return T_95[max(d for d in T_95 if d < df)]


def _check_critical():
    """One runnable check: the multiplier must shrink towards 1.96 as n grows."""
    assert critical(5) == 2.776, critical(5)      # four degrees of freedom
    assert critical(2) == 12.706, critical(2)
    assert critical(500) == 1.960, critical(500)
    assert critical(14) == T_95[12], critical(14)  # df 13 falls back to 12
    values = [critical(n) for n in (3, 5, 11, 21, 31, 500)]
    assert values == sorted(values, reverse=True), values


def paired_gap(pairs):
    """Mean difference, and the half-width of its 95 percent interval.

    `pairs` holds one difference per matched pair. Because the pairs share
    their starting code and batch order, the spread of these differences is
    much smaller than the spread of either side on its own.
    """
    if len(pairs) < 2:
        return float("nan"), float("nan")
    return mean(pairs), critical(len(pairs)) * stdev(pairs) / len(pairs) ** 0.5


def table(rows, field, label, percent):
    """The curve at each boundary, and the paired gap against boundary 0."""
    by_point = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(lambda: defaultdict(list))
    base = {}
    for row in rows:
        found = curve(row)
        if found is None:
            continue
        if row["point"] == 0:
            base[identity(row)] = found
    for row in rows:
        found = curve(row)
        if found is None:
            continue
        for step in STEPS:
            by_point[row["point"]][step].append(found[step][field])
        if row["point"] != 0 and identity(row) in base:
            for step in STEPS:
                paired[row["point"]][step].append(
                    found[step][field] - base[identity(row)][step][field])

    unit = " points" if percent else ""
    print()
    print(f"{label.upper()}")
    header = "".join(f"{'step ' + str(s):>12}" for s in STEPS)
    print(f"{'boundary':<28}{header}{'runs':>7}")
    for point in sorted(by_point):
        values = "".join(f"{mean(by_point[point][s]):12.2f}" for s in STEPS)
        print(f"{'t=' + str(point):<28}{values}{len(by_point[point][STEPS[0]]):>7}")
    for point in sorted(paired):
        if not paired[point][STEPS[0]]:
            continue
        cells = []
        for step in STEPS:
            gap, half = paired_gap(paired[point][step])
            mark = "*" if abs(gap) > half else " "
            cells.append(f"{gap:+11.2f}{mark}")
        print(f"{'gap t=' + str(point) + ' minus t=0':<28}{''.join(cells)}"
              f"{len(paired[point][STEPS[0]]):>7}")
    print(f"  A star marks a gap larger than its own 95 percent interval."
          f" Units are{unit or ' the raw value'}.")
    return by_point, paired


def speed_versus_ceiling(rows):
    """Is the model faster, or only better?

    For each run, divide its accuracy at step 62 by its own accuracy at step
    250. That fraction says how much of this run's own final result was
    already reached early. Dividing by the run's own ceiling is what makes it a
    measure of speed: a model that only ends higher, at the same pace, gets the
    same fraction.
    """
    fractions = defaultdict(list)
    paired = defaultdict(list)
    base = {}
    for row in rows:
        found = curve(row)
        if found is None:
            continue
        final = found[STEPS[-1]]["test_accuracy"]
        if final <= 0:
            continue
        share = found[EARLY]["test_accuracy"] / final
        fractions[row["point"]].append(share)
        if row["point"] == 0:
            base[identity(row)] = share
    for row in rows:
        found = curve(row)
        if found is None or row["point"] == 0:
            continue
        final = found[STEPS[-1]]["test_accuracy"]
        if final <= 0 or identity(row) not in base:
            continue
        paired[row["point"]].append(
            found[EARLY]["test_accuracy"] / final - base[identity(row)])

    print()
    print("SPEED SEPARATED FROM THE CEILING")
    print(f"Share of a run's own step-{STEPS[-1]} accuracy "
          f"already reached at step {EARLY}.")
    print()
    print(f"{'boundary':<28}{'share':>10}{'runs':>7}")
    for point in sorted(fractions):
        print(f"{'t=' + str(point):<28}{mean(fractions[point]):10.3f}"
              f"{len(fractions[point]):>7}")
    for point in sorted(paired):
        gap, half = paired_gap(paired[point])
        mark = "*" if abs(gap) > half else " "
        print(f"{'gap t=' + str(point) + ' minus t=0':<28}{gap:+10.3f}{mark}"
              f"{len(paired[point]):>7}  "
              f"95 percent interval +/- {half:.3f}")
    print()
    print("  A share that does not move means the model ends higher at the")
    print("  same pace, so the gain is ceiling and not speed. A share that")
    print("  rises means it genuinely learns faster.")
    return fractions, paired


def finished_by_four(rows):
    """Does the speed gain finish by task 4, as the accuracy endpoint does?

    The existing finding is that the probe gain is complete at boundary 4 and
    flat afterwards. That was measured on the final accuracy. This asks the
    same question of the speed share, which is a different measurement on the
    same runs. Only boundary 4's own orders are used, because boundary 4 was
    run on order A alone and a comparison against all three orders would
    confuse "later in the stream" with "a different ordering".
    """
    orders = {r["order"] for r in rows if r["point"] == 4}
    if not orders:
        return
    share = {}
    for row in rows:
        if row["order"] not in orders or row["point"] not in (4, 9):
            continue
        found = curve(row)
        if found is None:
            continue
        final = found[STEPS[-1]]["test_accuracy"]
        if final > 0:
            share[(row["point"], identity(row))] = (
                found[EARLY]["test_accuracy"] / final)
    pairs = [share[(9, key)] - share[(4, key)]
             for point, key in share if point == 4 and (9, key) in share]
    if len(pairs) < 2:
        return

    # The eleven repetitions of one seed probe the same two models, so they are
    # not eleven independent observations of the difference between those
    # models. Averaging within a seed first, then taking the spread across
    # seeds, uses the seed as the unit. This is the honest interval and it is
    # several times wider than treating every repetition as independent.
    by_seed = defaultdict(list)
    for point, key in share:
        if point == 4 and (9, key) in share:
            by_seed[key[1]].append(share[(9, key)] - share[(4, key)])
    clustered = [mean(values) for values in by_seed.values()]

    gap, half = paired_gap(pairs)
    seed_gap, seed_half = paired_gap(clustered)
    print()
    print("DOES THE SPEED GAIN FINISH BY TASK 4?")
    print(f"Orders used: {', '.join(sorted(orders))}. {len(pairs)} matched pairs "
          f"from {len(clustered)} seeds.")
    print(f"speed share, t=9 minus t=4: {gap:+.3f}, "
          f"95 percent interval +/- {half:.3f} counting every repetition")
    print(f"{'':27}{seed_gap:+.3f}, "
          f"95 percent interval +/- {seed_half:.3f} counting each seed once")
    half = seed_half
    if abs(gap) <= half:
        print("  Not separable from no change. The speed gain is finished by")
        print("  task 4, which is what the accuracy endpoint already said.")
    else:
        print("  Separable from no change, so the speed gain is not finished")
        print("  at task 4. This disagrees with the accuracy endpoint and both")
        print("  numbers have to be reported.")


def main():
    _check_critical()
    print("Reading the probe records from R2")
    rows = load()
    print(f"\n{len(rows)} full-arm runs")
    points = sorted({r["point"] for r in rows})
    rates = sorted({r["learning_rate"] for r in rows})
    print(f"boundaries {points}, learning rates {rates}, "
          f"{len({r['order'] for r in rows})} orders")

    table(rows, "test_accuracy", "test accuracy on the held-out group", True)
    table(rows, "train_loss", "training loss on the held-out group", False)
    speed_versus_ceiling(rows)
    finished_by_four(rows)


if __name__ == "__main__":
    main()
