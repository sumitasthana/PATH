"""Draw one stream run as a single picture: steps, retention, indicators.

The run map answers "what happened in this run?" without reading any JSON. It
stacks three panels on one shared axis of boundaries, so a column is a moment
in the run and you can read straight down it:

  1. the steps    L0 to L8 are the nine tasks learned, F0 is the task broken
                  on purpose at the end. Each box carries the accuracy that
                  task reached when it was learned.
  2. retention    one row per learned task, one column per boundary. The cell
                  is that task's accuracy at that moment. A row that fades to
                  the right is a task being forgotten as later tasks arrive.
  3. indicators   effective rank on each of the three fixed image sets, each
                  divided by its own value at boundary 0 so three different
                  scales share one axis.

Panel 3 sitting under panel 2 is the point of the whole figure. At the F0
column the top row of the grid drops to chance, because that task was
destroyed deliberately. Whatever the indicator lines do in that same column is
what the indicators say about a model that is known to be damaged. If they
rise, they are calling a broken model healthy, which is the claim this study
exists to test.

Usage, with whatever files exist:

    python scripts/build_run_map.py --label order_A-seed0 \\
        --indicators outputs/job2-seeds/order_A_seed0/indicators.jsonl \\
        --history outputs/job2-seeds/order_A_seed0/history_seq1_resnet50_seed0.json

With neither file it draws the empty template, which is how the layout gets
reviewed before any run has finished. Missing values are drawn as a dash, so a
partly finished run maps fine.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "wiki" / "figures"

# The same two palettes the trajectory figures use, so the wiki looks like one
# document in either colour scheme.
LIGHT = dict(bg="#ffffff", ink="#1b2027", muted="#5e6878", line="#d5dae2",
             soft="#e6eaf0", target="#4a63b5", retain="#2f7f72", fail="#9c4a3c")
DARK = dict(bg="#1a1f26", ink="#e4e8ee", muted="#9aa5b5", line="#333c49",
            soft="#28303a", target="#8ba0e0", retain="#58b7a6", fail="#dd8b78")

STYLE_RULES = """
  svg{font-family:ui-monospace,Consolas,monospace}
  text{fill:var(--muted);font-size:11px}
  .ct{fill:var(--ink);font-family:system-ui,sans-serif;font-weight:600;font-size:15px}
  .sub{fill:var(--muted);font-family:system-ui,sans-serif;font-size:12px}
  .pt{fill:var(--ink);font-family:system-ui,sans-serif;font-weight:600;font-size:12px}
  .axis{stroke:var(--line);stroke-width:1}
  .grid{stroke:var(--soft);stroke-width:1}
  .box{fill:var(--soft);stroke:var(--line);stroke-width:1}
  .box-forget{fill:none;stroke:var(--fail);stroke-width:1.6}
  .box-start{fill:none;stroke:var(--line);stroke-width:1;stroke-dasharray:3 3}
  .cell{stroke:var(--bg);stroke-width:1}
  .val{fill:var(--ink);font-size:10px}
  .lbl{fill:var(--ink);font-weight:600}
  .lbl-fail{fill:var(--fail);font-weight:600}
  .ln-in_stream{stroke:var(--target);stroke-width:2;fill:none}
  .ln-probe{stroke:var(--retain);stroke-width:2;fill:none}
  .ln-eurosat{stroke:var(--muted);stroke-width:2;fill:none;stroke-dasharray:4 3}
  .dot{stroke:var(--bg);stroke-width:1}
  .mark{stroke:var(--fail);stroke-width:1.2;stroke-dasharray:3 3}
  .chance{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 3;opacity:.6}
"""

SETS = ("in_stream", "probe", "eurosat")


def esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def read_history(path):
    """Accuracy of every seen task at every boundary, from the run's history.

    The history has one record per finished request, and each record's "after"
    block holds every task learned so far. Boundary numbering matches the
    indicator record: the request at index i is boundary i + 1.
    """
    if path is None or not Path(path).exists():
        return {}, []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data["history"] if isinstance(data, dict) else data
    accuracy, steps = {}, []
    for record in records:
        boundary = record["index"] + 1
        steps.append({"boundary": boundary, "action": record["action"],
                      "task": str(record["task"])})
        for task, value in record.get("after", {}).items():
            accuracy.setdefault(str(task), {})[boundary] = float(value)
    return accuracy, steps


def read_indicators(path):
    """One row per boundary, keyed by boundary number."""
    if path is None or not Path(path).exists():
        return {}
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["t"])] = row
    return rows


def plan_steps(steps, learned=9):
    """The columns of the map: boundary 0, then one per request.

    Falls back to the planned shape, nine learns then one forget, when there
    is no history yet. That is what makes the empty template drawable.
    """
    if steps:
        return steps
    return ([{"boundary": i + 1, "action": "learn", "task": str(i)}
             for i in range(learned)]
            + [{"boundary": learned + 1, "action": "forget", "task": "0"}])


def ramp(value, chance, best, colours):
    """Accuracy to a fill: chance level reads as damage, high reads as held."""
    if value is None:
        return "none"
    share = max(0.0, min(1.0, (value - chance) / max(best - chance, 1e-9)))
    # Two-stop blend from the fail colour to the retain colour, in sRGB. Good
    # enough for a wiki figure and it needs no dependency.
    lo = colours["fail"].lstrip("#")
    hi = colours["retain"].lstrip("#")
    out = []
    for i in (0, 2, 4):
        a, b = int(lo[i:i + 2], 16), int(hi[i:i + 2], 16)
        out.append(f"{round(a + (b - a) * share):02x}")
    return "#" + "".join(out)


def build(label, accuracy, indicators, steps, colours, chance=5.0):
    columns = plan_steps(steps)
    learns = [s for s in columns if s["action"] == "learn"]
    order = [s["task"] for s in learns]
    last = columns[-1]["boundary"] if columns else 10

    left, top, width = 74, 0, 62          # gutter, top offset, column width
    def x(boundary):                       # boundary 0 gets its own column
        return left + boundary * width + width / 2

    chart_width = left + (last + 1) * width + 24
    box_y, box_h = 62, 54
    grid_y, cell_h = 164, 26
    grid_h = cell_h * len(order)
    ind_y, ind_h = grid_y + grid_h + 74, 150
    height = ind_y + ind_h + 64

    out = []
    css = "".join(f"--{k}:{v};" for k, v in colours.items())
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_width}" '
        f'height="{height}" viewBox="0 0 {chart_width} {height}" role="img" '
        f'aria-labelledby="map-title map-desc">')
    out.append(f'<title id="map-title">Run map for {esc(label)}</title>')
    out.append(
        f'<desc id="map-desc">Three stacked panels sharing one axis of '
        f'boundaries. The steps of the run, the accuracy of every learned task '
        f'at every boundary, and effective rank on three fixed image sets '
        f'relative to the untrained model.</desc>')
    out.append(f'<style>:root{{{css}}}{STYLE_RULES}</style>')
    out.append(f'<rect width="{chart_width}" height="{height}" fill="var(--bg)"/>')

    out.append(f'<text class="ct" x="16" y="26">Run map: {esc(label)}</text>')
    out.append(f'<text class="sub" x="16" y="45">Nine tasks learned, then one '
               f'broken on purpose. Chance is {chance:.0f}% on 20 classes.</text>')

    # -- panel 1: the steps ---------------------------------------------------
    out.append(f'<text class="pt" x="16" y="{box_y - 8}">the run</text>')
    out.append(f'<rect class="box-start" x="{x(0) - width / 2 + 5}" y="{box_y}" '
               f'width="{width - 10}" height="{box_h}" rx="4"/>')
    out.append(f'<text class="lbl" x="{x(0)}" y="{box_y + 23}" '
               f'text-anchor="middle">B0</text>')
    out.append(f'<text x="{x(0)}" y="{box_y + 40}" text-anchor="middle">'
               f'untrained</text>')

    for step in columns:
        b, forget = step["boundary"], step["action"] == "forget"
        klass = "box-forget" if forget else "box"
        name = ("F" if forget else "L") + step["task"]
        out.append(f'<rect class="{klass}" x="{x(b) - width / 2 + 5}" '
                   f'y="{box_y}" width="{width - 10}" height="{box_h}" rx="4"/>')
        out.append(f'<text class="{"lbl-fail" if forget else "lbl"}" '
                   f'x="{x(b)}" y="{box_y + 23}" text-anchor="middle">'
                   f'{esc(name)}</text>')
        got = accuracy.get(step["task"], {}).get(b)
        shown = "-" if got is None else f"{got:.1f}%"
        out.append(f'<text class="val" x="{x(b)}" y="{box_y + 41}" '
                   f'text-anchor="middle">{shown}</text>')

    # -- panel 2: retention ---------------------------------------------------
    out.append(f'<text class="pt" x="16" y="{grid_y - 10}">'
               f'every learned task, at every boundary</text>')
    best = max([v for task in accuracy.values() for v in task.values()]
               + [60.0])
    for index, task in enumerate(order):
        row_y = grid_y + index * cell_h
        out.append(f'<text x="{left - 10}" y="{row_y + 17}" '
                   f'text-anchor="end">task {esc(task)}</text>')
        learned_at = next(s["boundary"] for s in learns if s["task"] == task)
        for step in columns:
            b = step["boundary"]
            if b < learned_at:
                continue                       # not learned yet, nothing to show
            got = accuracy.get(task, {}).get(b)
            fill = ramp(got, chance, best, colours)
            out.append(f'<rect class="cell" x="{x(b) - width / 2 + 5}" '
                       f'y="{row_y + 2}" width="{width - 10}" '
                       f'height="{cell_h - 4}" fill="{fill}" rx="2"/>')
            if got is not None:
                out.append(f'<text class="val" x="{x(b)}" y="{row_y + 17}" '
                           f'text-anchor="middle" fill="#ffffff">'
                           f'{got:.0f}</text>')

    # -- panel 3: the indicators ---------------------------------------------
    out.append(f'<text class="pt" x="16" y="{ind_y - 12}">effective rank, '
               f'relative to the untrained model</text>')
    base = indicators.get(0, {})
    series = {}
    for name in SETS:
        start = base.get(f"effective_rank_{name}")
        if not start:
            continue
        points = [(b, row[f"effective_rank_{name}"] / start)
                  for b, row in sorted(indicators.items())
                  if row.get(f"effective_rank_{name}")]
        if len(points) > 1:
            series[name] = points

    values = [v for points in series.values() for _, v in points] or [0.8, 1.2]
    low, high = min(values + [1.0]), max(values + [1.0])
    pad = max((high - low) * 0.15, 0.02)
    low, high = low - pad, high + pad

    def y(value):
        return ind_y + ind_h - (value - low) / (high - low) * ind_h

    out.append(f'<line class="axis" x1="{left}" y1="{ind_y + ind_h}" '
               f'x2="{chart_width - 16}" y2="{ind_y + ind_h}"/>')
    out.append(f'<line class="chance" x1="{left}" y1="{y(1.0):.1f}" '
               f'x2="{chart_width - 16}" y2="{y(1.0):.1f}"/>')
    out.append(f'<text x="{left - 10}" y="{y(1.0) + 4:.1f}" '
               f'text-anchor="end">same as B0</text>')

    for name, points in series.items():
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(b):.1f},{y(v):.1f}"
                        for i, (b, v) in enumerate(points))
        out.append(f'<path class="ln-{name}" d="{path}"/>')
        for b, v in points:
            out.append(f'<circle class="dot" cx="{x(b):.1f}" cy="{y(v):.1f}" '
                       f'r="3" fill="var(--{"target" if name == "in_stream" else "retain" if name == "probe" else "muted"})"/>')
        end_b, end_v = points[-1]
        out.append(f'<text x="{x(end_b) + 10:.1f}" y="{y(end_v) + 4:.1f}">'
                   f'{esc(name)}</text>')

    # The column where the model was broken on purpose, marked through both
    # lower panels, because lining those two up is the whole point.
    forgets = [s["boundary"] for s in columns if s["action"] == "forget"]
    if forgets:
        b = forgets[0]
        out.append(f'<line class="mark" x1="{x(b)}" y1="{grid_y - 4}" '
                   f'x2="{x(b)}" y2="{ind_y + ind_h}"/>')
        out.append(f'<text class="lbl-fail" x="{x(b)}" y="{height - 38}" '
                   f'text-anchor="middle">broken here</text>')

    out.append(f'<text class="sub" x="16" y="{height - 14}">'
               f'Read down the "broken here" column: the top row falls to '
               f'chance, and the lines show what the indicators said about '
               f'that same model.</text>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="template",
                        help="names the run and the output file")
    parser.add_argument("--indicators", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dark", action="store_true",
                        help="use the dark palette instead of the light one")
    args = parser.parse_args()

    accuracy, steps = read_history(args.history)
    indicators = read_indicators(args.indicators)
    text = build(args.label, accuracy, indicators, steps,
                 DARK if args.dark else LIGHT)

    out = args.out or FIGURES / f"run-map-{args.label}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    filled = len(indicators), sum(len(v) for v in accuracy.values())
    print(f"Wrote {out}")
    print(f"  {filled[0]} indicator boundaries, {filled[1]} accuracy readings")
    if not filled[0] and not filled[1]:
        print("  (empty template: pass --indicators and --history to fill it)")


if __name__ == "__main__":
    main()
