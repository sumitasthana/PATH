"""Does the weight-norm drop explain the rank rise at the unlearn boundary?

Boundary 9 of order_01 seed 1 reported effective rank up and srank up, right
after task 3 was destroyed. The same step shrank the generated weights by 8.4
percent, from a norm of 286.34 to 262.36, which is below the untrained
generator's 274.19.

Both rank measures divide by the sum of the singular values, so rescaling a
feature matrix cannot change either one. But the path from generator weights to
those features is not a rescale: it runs through ReLU and through batch
normalisation with fixed statistics, so shrinking the weights moves units
across their firing threshold and can reshape the spectrum on its own.

Four arms settle it. Nothing is trained; every arm is a forward pass.

    t8            the eight-task model as it is          expect 2.9708
    t9            the post-unlearn model as it is        expect 3.2664
    t9_at_t8_norm the post-unlearn weights scaled up to the t8 norm
    t8_at_t9_norm the eight-task weights scaled down to the t9 norm

The first two reproduce numbers already in the record, which is the check that
this script measures the same thing the run measured. Then:

  * if t9_at_t8_norm falls back near 2.97 and t8_at_t9_norm rises near 3.27,
    the norm explains the whole effect and the unlearn result is an artifact;
  * if both stay where their own model was, the norm explains none of it and
    the unlearn really did change the shape of the representation;
  * anything between is a partial explanation, to be reported as such.

Runs on the CPU in about twenty minutes, most of it downloading two
checkpoints. No GPU and no Colab session.
"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))

from uncle import checkpoint
from uncle.config import Config
from uncle.hypernet import HyperNetwork, build_target
from uncle.plasticity_indicators import (REFERENCE_SEED, fixed_image_sets, flatten,
                                         reference_code, srank)
from uncle.probe import _features_and_logits, effective_rank
from uncle.storage import R2Store
from uncle.streams import build_tasks
from uncle.trainer import UnCLe

STUDY = "indicator_stream_v1"
ORDER = "order_01"
SEED = 1
PREFIX = "uncle/paired_generalization/" + STUDY
T8_KEY = f"{PREFIX}/sources/{ORDER}/seed_{SEED}/checkpoint.pt"
T9_KEY = (f"{PREFIX}/branches/{ORDER}/seed_{SEED}/forget_3/learn_15/boundary_9.pt")
CACHE = REPO / "outputs/norm_control"
CACHE.mkdir(parents=True, exist_ok=True)

store = R2Store.connect(os.environ["R2_ENDPOINT"], "uncle-experiments",
                        os.environ["R2_ACCESS_KEY_ID"], os.environ["R2_SECRET_ACCESS_KEY"])
store.progress = True


@torch.no_grad()
def features_scaled(hypernet, target, trainer, images, code, scale, batch_size=64):
    """`plasticity_indicators.features`, with the generated weights scaled.

    Copied rather than imported with an argument added, because the library
    version is what produced every number already on record and a one-off
    control has no business changing it. The only difference is the one line
    that multiplies every generated weight by `scale`.
    """
    weights = hypernet.weights_from_code(code)
    if scale != 1.0:
        weights = {name: value * scale for name, value in weights.items()}
    modes = hypernet.training, target.training
    hypernet.eval()
    target.eval()
    collected = []
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(trainer.device)
            captured, _ = _features_and_logits(hypernet, target, trainer, None, batch,
                                               weights=weights,
                                               buffers=trainer.buffer_template)
            collected.append(captured.float().cpu())
    finally:
        hypernet.train(modes[0])
        target.train(modes[1])
    return torch.cat(collected)


print("Reading the study plan")
plan = json.loads(store.download(PREFIX + "/study.json", CACHE).read_text(encoding="utf-8"))
config = replace(Config(**plan["config"]), seed=SEED, device="cpu")
ORDER_TASKS = list(plan["orders"][ORDER])
PROBE_TASK = "2"
print("Tasks, in order:", " ".join(ORDER_TASKS), "| probe:", PROBE_TASK)

print("\nBuilding the images from the local Tiny ImageNet copy")
tasks = build_tasks(root=REPO / "data/tiny-imagenet-200",
                    partition_path=REPO / "uncle/task_partition.json",
                    partition_seed=42, include=tuple(ORDER_TASKS + [PROBE_TASK]),
                    download=False)
# Same construction and same order as notebook 36, so `fixed_image_sets` draws
# the same pictures: its generator is reseeded per set and indexes by position.
sources = {"in_stream": ConcatDataset([tasks[name]["test"] for name in ORDER_TASKS]),
           "probe": tasks[PROBE_TASK]["test"]}
image_sets = fixed_image_sets(sources, count=256)
for name, images in image_sets.items():
    print(f"  {name:<10} {tuple(images.shape)}")

print("\nDownloading the two models")
paths = {"t8": store.download(T8_KEY, CACHE), "t9": store.download(T9_KEY, CACHE)}


def load(path):
    """A fresh model with one checkpoint in it, on the CPU."""
    target = build_target(config)
    hypernet = HyperNetwork(target, config)
    trainer = UnCLe(hypernet, config, target, None, progress=False)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.restore(payload, hypernet=hypernet, uncle=trainer)
    return hypernet, target, trainer


print("\nLoading and measuring the generated weight norms")
models = {}
norms = {}
for name, path in paths.items():
    hypernet, target, trainer = load(path)
    models[name] = (hypernet, target, trainer)
    code = reference_code(hypernet)
    norms[name] = float(flatten(hypernet, code)["target"].norm())
    print(f"  {name}: norm_target {norms[name]:.4f}")

# Recorded in uncle/indicators/.../seed_1_forget_3.jsonl at t=8 and t=9.
print("  recorded:      t8 286.3382   t9 262.3619")

up = norms["t8"] / norms["t9"]      # grow the unlearned weights back
down = norms["t9"] / norms["t8"]    # shrink the eight-task weights
print(f"\nScale factors: t9 up by {up:.6f}, t8 down by {down:.6f}")

ARMS = [("t8", "t8", 1.0, "the eight-task model, as it is"),
        ("t9", "t9", 1.0, "the post-unlearn model, as it is"),
        ("t9_at_t8_norm", "t9", up, "post-unlearn weights grown back to the t8 norm"),
        ("t8_at_t9_norm", "t8", down, "eight-task weights shrunk to the t9 norm")]

results = {}
for label, which, scale, description in ARMS:
    hypernet, target, trainer = models[which]
    code = reference_code(hypernet)
    row = {}
    for name, images in image_sets.items():
        matrix = features_scaled(hypernet, target, trainer, images, code, scale)
        row[f"effective_rank_{name}"] = effective_rank(matrix)
        row[f"srank_{name}"] = srank(matrix)
    results[label] = row
    print(f"\n{label}  ({description})")
    for key, value in row.items():
        print(f"    {key:<28}{value:.4f}" if isinstance(value, float)
              else f"    {key:<28}{value}")

print("\n" + "=" * 68)
print(f"{'arm':<18}{'rank in_stream':>16}{'rank probe':>14}{'srank in':>10}{'srank probe':>13}")
for label, _, _, _ in ARMS:
    r = results[label]
    print(f"{label:<18}{r['effective_rank_in_stream']:>16.4f}"
          f"{r['effective_rank_probe']:>14.4f}"
          f"{r['srank_in_stream']:>10d}{r['srank_probe']:>13d}")
print("=" * 68)
print("recorded          t8 2.9708 / 2.9684 / 175 / 178")
print("recorded          t9 3.2664 / 3.2759 / 188 / 190")

(CACHE / "norm_control.json").write_text(
    json.dumps({"norms": norms, "scales": {"up": up, "down": down},
                "results": results}, indent=2), encoding="utf-8")
print("\nSaved to", CACHE / "norm_control.json")
