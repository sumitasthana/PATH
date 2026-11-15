"""Build the figures for the plasticity-indicators paper draft.

Every value here is copied from a table in the draft. Nothing is computed from
data the draft does not show, so a reader can check any point against a table.

Run: python scripts/build_paper_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no screen needed, we only write files
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "docs/plasticity/figures"
OUT.mkdir(parents=True, exist_ok=True)

# One row per stream: order, seed, probe at t=0, probe at t=9, change, floor,
# effective-rank change in percent. This is Table 4 of the draft.
RUNS = [
    ("A", 0, 26.13, 43.72, 17.59, 1.70, -29.8),
    ("A", 1, 27.25, 43.65, 16.40, 2.72, -66.6),
    ("A", 2, 26.06, 43.75, 17.68, 2.10, -31.7),
    ("A", 3, 26.55, 42.76, 16.22, 1.26, -57.9),
    ("A", 4, 26.78, 43.90, 17.12, 2.07, -65.3),
    ("B", 0, 25.50, 42.39, 16.89, 1.62, -77.2),
    ("B", 1, 26.35, 43.81, 17.45, 1.91, -68.9),
    ("B", 2, 26.15, 44.24, 18.09, 1.81, -30.7),
    ("B", 3, 26.21, 44.03, 17.82, 2.63, -60.7),
    ("B", 4, 26.05, 45.28, 19.24, 1.60, -80.6),
    ("C", 0, 27.41, 43.31, 15.90, 1.67, -78.8),
    ("C", 1, 25.27, 44.45, 19.18, 1.82, -68.2),
    ("C", 2, 24.95, 45.44, 20.49, 1.48, -69.2),
    ("C", 3, 26.39, 45.36, 18.97, 2.31, -62.2),
    ("C", 4, 26.01, 45.04, 19.03, 2.27, -70.3),
]

COLOR = {"A": "#1b6ca8", "B": "#c8641e", "C": "#3a7d44"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def figure_1_no_relation():
    """The headline figure: rank fall against probe gain, fifteen streams.

    If effective rank reported lost ability to learn, the runs furthest to the
    left here would sit lowest. They do not, so the cloud is flat.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for order in "ABC":
        pts = [(r[6], r[4]) for r in RUNS if r[0] == order]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   s=46, color=COLOR[order], label=f"order {order}",
                   edgecolor="white", linewidth=0.6, zorder=3)
    # The fitted line, from the slope and means given in Table 6 of the draft.
    mean_rank, mean_change, slope = -61.21, 17.87, -0.0042
    xs = [-85, -25]
    ax.plot(xs, [mean_change + slope * (x - mean_rank) for x in xs],
            color="#999999", linewidth=1.2, linestyle="--", zorder=2,
            label="fitted line, r = -0.055")
    ax.set_xlabel("Change in effective rank over the stream (%)")
    ax.set_ylabel("Change in probe accuracy (points)")
    ax.set_ylim(0, 23)
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title("The size of the rank fall does not predict learning",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1-rank-fall-vs-probe.png")
    plt.close(fig)


def figure_2_probe_endpoints():
    """Every run's probe accuracy before and after the stream, with its floor.

    One line per run. The grey band is chance, 5 percent for twenty classes.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    labels = []
    for i, (order, seed, t0, t9, change, floor, _) in enumerate(RUNS):
        y = len(RUNS) - 1 - i
        ax.plot([t0, t9], [y, y], color=COLOR[order], linewidth=1.6, zorder=2)
        ax.scatter([t0], [y], s=22, color="white", edgecolor=COLOR[order],
                   linewidth=1.3, zorder=3)
        ax.scatter([t9], [y], s=30, color=COLOR[order], zorder=3)
        # The floor is how far the reading could move on repetition noise
        # alone. Drawn on the starting point so the gap to the end is visible.
        ax.plot([t0 - floor, t0 + floor], [y, y], color="#b0b0b0",
                linewidth=4, alpha=0.55, zorder=1)
        labels.append(f"order {order}, seed {seed}")
    ax.axvline(5, color="#d04545", linewidth=1.0, linestyle=":")
    ax.text(5.6, len(RUNS) - 1.2, "chance, 5%", color="#d04545", fontsize=8)
    ax.set_yticks(range(len(RUNS)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=7.5)
    ax.set_xlabel("Probe accuracy on the held-out 20-class group (%)")
    ax.set_xlim(0, 50)
    ax.set_title("Open circle: untrained generator.  Filled: after nine tasks.\n"
                 "Grey bar: the run's own noise floor.",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2-probe-endpoints.png")
    plt.close(fig)


def figure_3_rescaling():
    """How far the unlearn moves effective rank, against an ordinary step.

    Values are the averages over the fifteen before-and-after pairs given in
    Table 7 of the draft.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    names = ["An ordinary\nlearning step",
             "The unlearn step,\nweight size held fixed",
             "The unlearn step,\nas measured"]
    values = [0.162, 0.754, 1.580]
    colors = ["#9aa5ad", "#6f9bbf", "#1b6ca8"]
    bars = ax.barh(names, values, color=colors, height=0.6)
    for bar, v in zip(bars, values):
        ax.text(v + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}", va="center", fontsize=9)
    ax.set_xlabel("Change in effective rank, in-stream images")
    ax.set_xlim(0, 1.9)
    ax.set_title("Destroying a task moves the measure further than learning does,\n"
                 "and in the direction the measure calls healthy",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3-unlearn-vs-learning-step.png")
    plt.close(fig)


def figure_4_order_retraction():
    """One run per order against five seeds per order, on the same measure.

    Left: the 6 November reading, one run each, which looked like a large
    order effect. Right: the same measurement with five seeds, where the
    within-order spread covers the gap. Values from Table 9 of the draft.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)

    # The old design's three orderings, one run each. Different orderings from
    # A, B and C, so they are named as the old record named them.
    left.bar(["old 03", "old 01", "old 02"], [22.6, 44.6, 71.8],
             color="#7f8c99", width=0.6)
    for i, v in enumerate([22.6, 44.6, 71.8]):
        left.text(i, v + 1.5, f"{v}", ha="center", fontsize=9)
    left.set_title("One run per order, old design\n(6 November)",
                   fontsize=9, loc="left")
    left.set_ylabel("Fall in effective rank (%)")
    left.set_xlabel("ordering")

    means = [50.2, 63.6, 69.7]
    spreads = [18.1, 20.0, 6.0]
    right.bar(["A", "B", "C"], means, color=["#1b6ca8", "#c8641e", "#3a7d44"],
              width=0.6, yerr=spreads, capsize=5,
              error_kw={"ecolor": "#444444", "linewidth": 1.2})
    for i, v in enumerate(means):
        right.text(i, v + 1.5, f"{v}", ha="center", fontsize=9)
    right.set_title("Five seeds per order, bars show the\nspread inside one order "
                    "(14 November)", fontsize=9, loc="left")
    right.set_xlabel("ordering")
    right.set_ylim(0, 95)
    fig.tight_layout()
    fig.savefig(OUT / "fig4-order-effect-retracted.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_1_no_relation()
    figure_2_probe_endpoints()
    figure_3_rescaling()
    figure_4_order_retraction()
    for f in sorted(OUT.glob("*.png")):
        print("wrote", f.relative_to(OUT.parent.parent.parent), f.stat().st_size, "bytes")
