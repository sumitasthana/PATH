"""Recompute the 15 November audit from downloaded R2 JSON logs, without training.

Usage: python scripts/audit_plasticity_revised_logs.py outputs/revised_log_audit
The cache must preserve R2 keys under its root. Results print as JSON.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def audit(root):
    study = "plasticity_revised_v1"
    base = root / "uncle/indicators" / study
    paths = sorted(base.glob("order_*/*.jsonl"))
    streams = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert [r["t"] for r in rows] == list(range(11)), path
        streams[(rows[0]["order"], rows[0]["seed"])] = rows
    expected = {(o, s) for o in ("order_A", "order_B", "order_C") for s in range(5)}
    assert set(streams) == expected
    probe_path = root / "uncle/matched_probe" / study / "revised_20class_train4800.jsonl"
    probes = [json.loads(line) for line in probe_path.read_text().splitlines() if line.strip()]
    cells = defaultdict(list)
    for r in probes:
        assert r["arm"] == "full" and not r["code_only"] and r["train_code"]
        assert r["steps"] == 250 and r["train_images"] == 4800
        cells[(r["order"], r["seed_label"], r["point"], r["repetition"])].append(r)
    assert len(probes) == 990
    assert set(cells) == {(o, s, t, rep) for o, s in expected for t in (0, 9) for rep in range(11)}
    assert all(sorted(r["learning_rate"] for r in v) == [0.0001, 0.0003, 0.001] for v in cells.values())
    selected = [min(v, key=lambda r: (-r["val_accuracy"], r["learning_rate"])) for v in cells.values()]
    result = {"coverage": {"streams": len(streams), "indicator_rows": 165, "probe_rows": len(probes)},
              "selected_accuracy": {t: mean(r["test_accuracy"] for r in selected if r["point"] == t) for t in (0, 9)}}
    fixed = {}
    for rate in (0.0001, 0.0003, 0.001):
        rows = [r for r in probes if r["learning_rate"] == rate]
        early_gains, final_gains = [], []
        for order, seed in sorted(expected):
            at = {t: [r for r in rows if (r["order"], r["seed_label"], r["point"]) == (order, seed, t)] for t in (0, 9)}
            baseline = mean(r["test_accuracy"] for r in at[0])
            early_gains.append(mean(r["readings"][0]["test_accuracy"] for r in at[9]) - baseline)
            final_gains.append(mean(r["test_accuracy"] for r in at[9]) - baseline)
        fixed[rate] = {"endpoints": {t: {"test": mean(r["test_accuracy"] for r in rows if r["point"] == t),
                                        "train": mean(r["final_train_accuracy"] for r in rows if r["point"] == t),
                                        "test_at_62": mean(r["readings"][0]["test_accuracy"] for r in rows if r["point"] == t)} for t in (0, 9)},
                       "early_gain_per_stream": early_gains, "final_gain_per_stream": final_gains}
    result["fixed_rates"] = fixed
    result["trajectories"] = {}
    for metric in ("effective_rank_in_stream", "effective_rank_probe", "effective_rank_eurosat", "srank_in_stream", "dormant_0.01_in_stream"):
        result["trajectories"][metric] = {
            "means": [mean(rows[t][metric] for rows in streams.values()) for t in range(11)],
            "first_step_down": sum(rows[1][metric] < rows[0][metric] for rows in streams.values()),
            "later_rebound": sum(rows[9][metric] > rows[1][metric] for rows in streams.values())}
    controls = json.loads((base / "rescaling_control.json").read_text())["readings"]
    assert len(controls) == 15
    result["rescaling"] = {}
    for domain in ("in_stream", "probe", "eurosat"):
        metric = "effective_rank_" + domain
        values = []
        for name, row in controls.items():
            a, b, c, d = [row["arms"][arm][metric] for arm in ("b9", "b10", "b10_at_b9_norm", "b9_at_b10_norm")]
            values.append({"run": name, "raw": b-a, "direction_at_old_norm": c-a,
                           "scale_at_old_direction": d-a, "scale_at_new_direction": b-c,
                           "interaction": (b-c)-(d-a)})
        averages = {k: mean(r[k] for r in values) for k in values[0] if k != "run"}
        averages["scale_share_old_direction"] = averages["scale_at_old_direction"] / averages["raw"]
        averages["scale_share_new_direction"] = averages["scale_at_new_direction"] / averages["raw"]
        result["rescaling"][domain] = {"means": averages, "runs": values}
    result["revisions"] = dict(Counter(r["code_revision"] for r in probes))
    result["sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(root.glob("uncle/**/*.json*"))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.cache), indent=2))
