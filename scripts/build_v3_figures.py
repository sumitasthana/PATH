"""Figures for version 3 of the plasticity-indicators paper.

Plainer than the version 2 figures: no colour coding, no title inside the
figure, no claim stated on the axes. The caption in the paper carries the
reading. Version 1 used three figures in the whole paper, and version 3 keeps
to about that density.

Every value is read from the records, not copied by hand. Run:

    python scripts/build_v3_figures.py
"""
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")           # write files only, no screen needed
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/plasticity/figures-v3"
OUT.mkdir(parents=True, exist_ok=True)

for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        n, _, v = line.partition("=")
        os.environ.setdefault(n.strip(), v.strip().strip('"').strip("'"))

import boto3
from botocore.config import Config

BUCKET = "uncle-experiments"
STUDY = "plasticity_revised_v1"
s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                  aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                  config=Config(signature_version="s3v4"))


def read(key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode("utf-8")


def indicator_changes():
    """Percent change from boundary 0 to boundary 9, per run, per measure."""
    out = {}
    for order in ("order_A", "order_B", "order_C"):
        for seed in range(5):
            text = read(f"uncle/indicators/{STUDY}/{order}/seed_{seed}.jsonl")
            rows = {r["t"]: r for r in
                    (json.loads(l) for l in text.splitlines() if l.strip())}
            a, b = rows[0], rows[9]
            out[(order, seed)] = {
                m: 100.0 * (b[m] - a[m]) / a[m] for m in
                ("effective_rank_in_stream", "srank_in_stream",
                 "dormant_0.01_in_stream", "dormant_0.1_in_stream")
            }
    return out


def probe_changes():
    """Probe change per run, using the notebook's own selection rule.

    The learning rate is chosen inside each repetition on that repetition's
    validation accuracy, ties going to the lower rate. The reported value is
    the mean over the eleven selected repetitions.
    """
    text = read(f"uncle/matched_probe/{STUDY}/revised_20class_train4800.jsonl")
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["order"], r["seed_label"], r["point"], r["repetition"])].append(r)
    picked = defaultdict(list)
    for key, arms in cells.items():
        best = min(arms, key=lambda a: (-a["val_accuracy"], a["learning_rate"]))
        picked[key[:3]].append(best["test_accuracy"])
    out = {}
    for order in ("order_A", "order_B", "order_C"):
        for seed in range(5):
            at0, at9 = picked[(order, seed, 0)], picked[(order, seed, 9)]
            out[(order, seed)] = {
                "start": mean(at0),
                "end": mean(at9),
                "change": mean(at9) - mean(at0),
                "se": math.sqrt(stdev(at0) ** 2 / len(at0)
                                + stdev(at9) ** 2 / len(at9)),
            }
    return out


def trajectories():
    """Every reading of every run, keyed by measure then by boundary.

    Section 5.1 needs the whole path, not just the two endpoints, because the
    shape of the path is the finding: the move happens at the first task.
    """
    out = {}
    for order in ("order_A", "order_B", "order_C"):
        for seed in range(5):
            text = read(f"uncle/indicators/{STUDY}/{order}/seed_{seed}.jsonl")
            out[(order, seed)] = {
                r["t"]: r for r in
                (json.loads(l) for l in text.splitlines() if l.strip())
            }
    return out


def figure_trajectories(runs):
    """Three panels: one per measure, one thin line per run, mean in black.

    The thin lines show how much the runs differ. The thick line shows the
    shape they share, which is a step at the first task and then a flat tail.
    """
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200,
                         "font.family": "serif",
                         "axes.spines.top": False, "axes.spines.right": False})
    panels = [("Effective rank", "effective_rank_in_stream"),
              ("srank", "srank_in_stream"),
              ("Dormant units, cut-off 0.01", "dormant_0.01_in_stream")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    points = range(10)                   # boundary 0 to 9, the nine tasks
    for ax, (label, key) in zip(axes, panels):
        for run in runs.values():
            ax.plot(points, [run[t][key] for t in points],
                    color="0.75", linewidth=0.6)
        ax.plot(points, [mean([r[t][key] for r in runs.values()]) for t in points],
                color="black", linewidth=1.6)
        ax.set_xlabel("Task boundary")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([0, 3, 6, 9])
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "v3-fig1-indicator-trajectories.png")
    plt.close(fig)


def figure_probe_endpoints(probes):
    """Probe accuracy at the start and the end of the stream, one column per run.

    Each run gets a vertical line from its boundary-0 accuracy to its
    boundary-9 accuracy, so the direction of the move is the thing you see
    first. Chance is drawn in because the whole scale only means something
    against it.
    """
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200,
                         "font.family": "serif",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    keys = [(o, s) for o in ("order_A", "order_B", "order_C") for s in range(5)]
    for i, key in enumerate(keys):
        lo, hi = probes[key]["start"], probes[key]["end"]
        ax.plot([i, i], [lo, hi], color="0.6", linewidth=0.9)
        ax.plot(i, lo, "o", markersize=4, color="black", markerfacecolor="white")
        ax.plot(i, hi, "o", markersize=4, color="black")
    ax.axhline(5.0, color="black", linewidth=0.8, linestyle=":")
    ax.text(14.3, 6.2, "chance", fontsize=8, ha="right")
    ax.set_xticks(range(15))
    ax.set_xticklabels([f"{o[-1]}{s}" for o, s in keys], fontsize=8)
    ax.set_xlabel("Run")
    ax.set_ylabel("Probe accuracy (%)")
    ax.set_ylim(0, 50)
    fig.tight_layout()
    fig.savefig(OUT / "v3-fig2-probe-endpoints.png")
    plt.close(fig)


def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    top = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    bottom = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return top / bottom


def figure_1(indicators, probes):
    """Effective-rank change against probe change, one point per run.

    Error bars are the probe's own standard error for that run, so a reader
    can see how much of the vertical spread the measurement could produce by
    itself.
    """
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200,
                         "font.family": "serif",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    marks = {"order_A": "o", "order_B": "s", "order_C": "^"}
    for order in marks:
        xs = [indicators[(order, s)]["effective_rank_in_stream"] for s in range(5)]
        ys = [probes[(order, s)]["change"] for s in range(5)]
        es = [probes[(order, s)]["se"] for s in range(5)]
        ax.errorbar(xs, ys, yerr=es, fmt=marks[order], markersize=4.5,
                    color="black", markerfacecolor="white", linewidth=0.8,
                    capsize=2, label=order.replace("order_", "ordering "))
    xs = [indicators[k]["effective_rank_in_stream"] for k in indicators]
    ys = [probes[k]["change"] for k in indicators]
    r = pearson(xs, ys)
    slope = r * stdev(ys) / stdev(xs)
    line_x = [min(xs) - 3, max(xs) + 3]
    ax.plot(line_x, [mean(ys) + slope * (x - mean(xs)) for x in line_x],
            color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Change in effective rank over the stream (%)")
    ax.set_ylabel("Change in probe accuracy (points)")
    ax.set_ylim(0, 24)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "v3-fig3-rank-change-vs-probe-change.png")
    plt.close(fig)
    return r


if __name__ == "__main__":
    figure_trajectories(trajectories())
    ind = indicator_changes()
    prb = probe_changes()
    figure_probe_endpoints(prb)
    r = figure_1(ind, prb)
    print(f"figure 3 written, r = {r:+.3f}")
    for f in sorted(OUT.glob("*.png")):
        print("wrote", f.relative_to(REPO), f.stat().st_size, "bytes")
