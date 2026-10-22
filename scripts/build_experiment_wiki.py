"""Validate the experiment archive and render its GitHub wiki pages."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "experiments"
REPORTS = ROOT / "docs" / "reports" / "trajectory"


def read_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate(registry, traces, gradients, norms):
    ids = [row["id"] for row in registry["experiments"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment identifiers")
    thresholds = registry["thresholds"]
    for experiment in registry["experiments"]:
        for evidence in experiment["evidence"]:
            if not (ROOT / evidence["path"]).is_file():
                raise ValueError(f"Missing evidence: {evidence['path']}")
        rows = [row for row in traces if row["run"] == experiment["id"]]
        if not rows:
            continue
        steps = [int(row["step"]) for row in rows]
        complete = experiment["observations"].get("reported_status", "complete") == "complete"
        expected_length = experiment["settings"]["steps"] + 1 if complete else len(rows)
        if steps != list(range(expected_length)) or len(rows) > experiment["settings"]["steps"] + 1:
            raise ValueError(f"Missing or unordered observations: {experiment['id']}")
        initial_target = float(rows[0]["task3_accuracy_pct"])
        initial_retained = float(rows[0]["task0_accuracy_pct"])
        valid = min(initial_target, initial_retained) >= thresholds["minimum_initial_accuracy_pct"]
        passing, failures = [], []
        for row in rows:
            target = float(row["task3_accuracy_pct"])
            retained = float(row["task0_accuracy_pct"])
            drift = abs(retained - initial_retained)
            if not math.isclose(drift, float(row["task0_absolute_drift_pp"]), abs_tol=1e-9):
                raise ValueError(f"Incorrect drift: {row}")
            passed = (valid and int(row["step"]) > 0
                      and target <= thresholds["target_accuracy_at_most_pct"]
                      and drift < thresholds["retained_absolute_drift_strictly_under_pp"])
            if str(passed) != row["passes_screen"]:
                raise ValueError(f"Incorrect screen: {row}")
            if passed:
                passing.append(int(row["step"]))
            if int(row["step"]) > 0 and drift >= thresholds["retained_absolute_drift_strictly_under_pp"]:
                failures.append(int(row["step"]))
        observations = experiment["observations"]
        expected = {
            "passing_steps": passing,
            "first_retention_failure_step": failures[0] if failures else None,
            "task3_final_accuracy_pct": float(rows[-1]["task3_accuracy_pct"]),
            "task0_final_accuracy_pct": float(rows[-1]["task0_accuracy_pct"]),
            "minimum_task3_accuracy_pct": min(float(r["task3_accuracy_pct"]) for r in rows),
            "accuracy_observations": len(rows),
        }
        for key, value in expected.items():
            if observations[key] != value:
                raise ValueError(f"Registry/trace mismatch: {experiment['id']} {key}")
    for table, key_fields in ((traces, ("run", "step")),
                              (gradients, ("run", "step", "parameter_group")),
                              (norms, ("run", "step"))):
        keys = [tuple(row[key] for key in key_fields) for row in table]
        if len(keys) != len(set(keys)) or any(row["run"] not in ids for row in table):
            raise ValueError("Duplicate observations or unknown run")


def value(item):
    if item is None:
        return "Not recorded"
    if isinstance(item, (list, dict)):
        return json.dumps(item, ensure_ascii=False)
    return str(item).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(value(item) for item in row) + " |" for row in rows],
    ]) + "\n"


def render(registry, traces, gradients, norms):
    """Keep the experiment index and dated records; background pages have separate subjects.

    Edit the dated trajectory records and index for new results. Raw measurements
    stay in DATA. This build checks the archive and removes obsolete wiki pages.
    """
    base = registry["repository"]
    link = lambda title, slug: f"[{title}]({base}/wiki/{slug})"
    pages = {
        "Experiment-trajectory.md": (ROOT / "docs/wiki/Experiment-trajectory.md").read_text(encoding="utf-8"),
        "Weekly-Updates.md": (ROOT / "docs/WEEKLY_UPDATES.md").read_text(encoding="utf-8"),
        "Concepts-and-processes.md": (ROOT / "docs/CONCEPTS.md").read_text(encoding="utf-8"),
        "An-Unlearning-Framework-for-Continual-Learning.md": (ROOT / "docs/ANCHOR_PAPER.md").read_text(encoding="utf-8"),
    }
    pages.update({path.name: path.read_text(encoding="utf-8")
                  for path in sorted((ROOT / "docs/wiki").glob("Experiment-trajectory-*.md"))})
    pages["Model-diagnostics.md"] = (
        (ROOT / "docs/MODEL_DIAGNOSTICS.md").read_text(encoding="utf-8")
        .replace("(../notebooks/04_gradient_diagnostics.ipynb)",
                 f"({base}/blob/main/notebooks/04_gradient_diagnostics.ipynb)")
        .replace("(../uncle/research_diagnostic.py)",
                 f"({base}/blob/main/uncle/research_diagnostic.py)")
        .replace("(PLAN.md)", f"({base}/blob/main/docs/PLAN.md)")
    )
    entries = [
        ("Experiment trajectory", "Experiment-trajectory", "latest findings and experiment records by date"),
        ("Weekly updates", "Weekly-Updates", "what we measured each week, what it means, and what is next"),
        ("Concepts and processes", "Concepts-and-processes", "what the model does and how to read its results"),
        ("Model diagnostics", "Model-diagnostics", "inspect weights, gradients, and Fisher scores"),
        ("Anchor paper", "An-Unlearning-Framework-for-Continual-Learning", "the paper we are reproducing"),
    ]
    pages["Home.md"] = (
        "# Plasticity After Task-unlearning in Hypernetworks (PATH)\n\n"
        "Wiki for the study. It asks whether unlearning a task restores the model's\n"
        "ability to learn new ones, as the UnCLe paper claims. The repository was\n"
        "earlier called CARK and has been renamed, so links now use PATH. Old\n"
        "repository links redirect; old `github.io/CARK` pages do not.\n\n"
    ) + "\n".join(
        f"- {link(title, slug)}: {description}." for title, slug, description in entries) + "\n"
    pages["_Sidebar.md"] = "- " + link("Home", "Home") + "\n" + "\n".join(
        "- " + link(title, slug) for title, slug, _ in entries) + "\n"
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "wiki")
    parser.add_argument("--check", action="store_true", help="Validate data and compare pages without writing")
    args = parser.parse_args()
    registry = json.loads((DATA / "registry.json").read_text(encoding="utf-8"))
    traces, gradients, norms = (read_csv(name) for name in
                               ("forgetting_traces.csv", "gradient_norms.csv", "raw_output_norms.csv"))
    validate(registry, traces, gradients, norms)
    pages = render(registry, traces, gradients, norms)
    for name, text in pages.items():
        if "\u2014" in text:
            raise ValueError(f"Em dash in generated prose: {name}")
        destination = args.output / name
        if args.check:
            if not destination.exists() or destination.read_text(encoding="utf-8") != text:
                raise ValueError(f"Page needs regeneration: {destination}")
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
    # A page that is no longer generated has to leave the output directory too,
    # or it survives as an orphan that nothing links to and nobody updates.
    stale = sorted(path for path in args.output.glob("*.md") if path.name not in pages)
    if stale and args.check:
        raise ValueError("Pages no longer generated: "
                         + ", ".join(path.name for path in stale))
    for path in stale:
        path.unlink()
        print(f"removed {path.name}")

    # Trajectory pages link their figures relatively, so the figures travel
    # with them into the wiki repository.
    figures = REPORTS / "figures"
    if not args.check:
        shutil.copytree(figures, args.output / "figures", dirs_exist_ok=True)
    elif any(not (args.output / "figures" / f.relative_to(figures)).is_file()
             for f in figures.rglob("*.svg")):
        raise ValueError(f"Trajectory figures missing from {args.output}")
    print(f"Validated {len(registry['experiments'])} records and {len(traces)} accuracy observations; "
          f"{'checked' if args.check else 'generated'} {len(pages)} wiki pages.")


if __name__ == "__main__":
    main()
