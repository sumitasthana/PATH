"""Builds notebooks/48_what_the_indicators_read.ipynb from a list of cells."""
import json
from pathlib import Path

C = []
def md(text): C.append(("markdown", text.strip("\n")))
def code(text): C.append(("code", text.strip("\n")))


md(r"""
# 48. What the rank indicators are actually reading

## Why this exists

The paper says three diagnostics disagree with a direct test of plasticity.
*Plasticity* means the model's remaining ability to learn something new.

A re-reading of the fifteen saved streams on 15 November found four problems
with that claim. Each one is a measurement question, not a modelling question,
and each can be answered from models that are already saved. This notebook
answers all four in one session.

## The four tests

**Test 1. Which task code is the indicator reading?**

A hypernetwork does not have one set of weights. A *generator* writes the
weights, and it writes different weights for every *task code*, which is a list
of 32 numbers naming a task. So before you can measure anything you have to
choose a code.

The study chooses one fixed code that was never trained and belongs to no task.
If the reading changes a lot when you swap that code for a code the model
actually uses, then the indicator describes the reading route and not the
learner. This test reads the same saved model through the fixed code and
through every trained task code it has.

Forward passes only. Nothing is trained.

**Test 2. Does the average feature explain the low rank?**

*Effective rank* counts how many directions the features really use. The study
reports values between 1 and 7 out of a possible 2048.

The features come out of a ReLU, so none of them is ever negative. When every
number in a matrix is positive, the largest single direction is usually just
the average of the rows. That one direction can take almost all of the weight
and push effective rank towards 1, whatever the model is doing.

This test computes every reading twice: once as the study does, and once after
subtracting the average row first.

**What the answer will mean.** A simulated matrix of the same shape, with a
large common average and ordinary noise around it, gives these pairs:

| How much of the matrix is the average | Effective rank | srank |
| --- | --- | --- |
| 0.971 | 134 | 252 |
| 0.9996 | 2.8 | 227 |
| 0.99999 | 1.5 | 175 |

So a large average alone can push effective rank towards 1, but it leaves
srank high. The study reports effective rank near 1.7 together with srank near
66. A plain average offset does not produce that pair. The features must be
more degenerate than an offset explains.

That makes this test worth running either way. If the centred rank is large,
the published numbers were measuring the average. If it is also small, the
representation really is low-dimensional, and the finding moves to why a
low-dimensional representation still learns a new task perfectly well.

Forward passes only. Nothing is trained.

**Test 3. Does the ability to learn fall during the stream?**

The probe has only ever been measured at boundary 0, the untrained model, and
boundary 9, the end. Boundary 0 is a random network, so that comparison is
"random model against trained model", which is not a plasticity question.

The real question needs a point inside the stream. Every run saved its model at
boundary 4. This test probes it.

**Test 4. What happens when the generator really is frozen?**

Every saved probe record says `arm="full"`, `train_code=True` and
`code_only=False`. That means the probe trained the generator and the code
together. The paper describes a frozen generator with only a fresh code
training, which is a different measurement.

This test runs that second arm, so the paper can report the protocol it
describes.

## Order

Tests 1 and 2 come first because they are cheap and they change the most. They
need no training at all. Tests 3 and 4 need GPU time and are set up so that
they resume, so stopping halfway costs nothing.

## What is not here

No new stream is trained. Boundary 1 was never saved, so test 3 can only use
boundary 4. The existing probe records are read but never overwritten: new runs
go to a separate file.
""")

md("## 1. Load the code")

code(r'''
from pathlib import Path
import sys, os, json, time, subprocess, importlib.util

IN_COLAB = "google.colab" in sys.modules

# Importing uncle twice in one session gives two copies of the module with two
# copies of its constants, which is a confusing way to fail much later.
if any(name == "uncle" or name.startswith("uncle.") for name in sys.modules):
    raise RuntimeError("Restart the runtime before loading the code.")

if IN_COLAB:
    REPO = Path("/content/path-indicators")
    if not REPO.exists():
        subprocess.run(["git", "clone",
                        "https://github.com/sumitasthana/PATH.git", str(REPO)],
                       check=True)
    subprocess.run(["git", "-C", str(REPO), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "main"], check=True)
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)
else:
    REPO = next((p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "uncle/task_partition_10x20.json").exists()), None)
    if REPO is None:
        raise FileNotFoundError("Open this from a checkout that has the ten-group split.")

for package in ("boto3", "tqdm"):
    if importlib.util.find_spec(package) is None and IN_COLAB:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package],
                       check=True)
sys.path.insert(0, str(REPO))

import torch
from tqdm.auto import tqdm
from uncle import checkpoint
from uncle.config import Config, PROBE_TASK, study_defaults
from uncle.hypernet import HyperNetwork, build_target
from uncle.plasticity_indicators import (
    SRANK_DELTA, fixed_image_sets, features, indicator_key, reference_code)
from uncle.probe import probe_once, snapshot, unchanged
from uncle.storage import R2Store
from uncle.streams import build_tasks
from uncle.tasks import PARTITIONS
from uncle.trainer import UnCLe

HEAD = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                               text=True).strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
print("Running at:", HEAD)
print("Device:", GPU_NAME)
''')

md("## 2. Settings")

code(r'''
STUDY_ID = "plasticity_revised_v1"
BUCKET = "uncle-experiments"

# ---------------------------------------------------------------- tests 1, 2
# Which saved models to read. Boundary 9 is the end of the stream and is the
# one every published indicator number describes. Boundary 4 is the middle, and
# adding it turns test 1 from "the reading depends on the code" into "the
# reading depends on the code, and the trend over the stream does too".
#
# Each file is about 540 MB. Fifteen runs at two boundaries is roughly 16 GB of
# download. Each file is deleted as soon as it has been read, so the disk never
# holds more than one.
#
# To cut the cost, set READ_ORDERS to ["order_A"] for five runs instead of
# fifteen. The conclusion of test 1 does not need all three orderings, because
# it compares codes inside a single model.
READ_ORDERS = ["order_A", "order_B", "order_C"]
READ_SEEDS = [0, 1, 2, 3, 4]
READ_BOUNDARIES = [4, 9]

# Which picture sets to read. The published records use three. `eurosat` is
# satellite imagery and is left out here only to save its download; the two
# Tiny ImageNet sets already separate "data the model trained on" from "data it
# never saw", which is what tests 1 and 2 need.
READ_EUROSAT = False
IMAGES_PER_SET = 256

READ_KEY = f"uncle/indicator_protocol/{STUDY_ID}/code_sweep_and_centering.jsonl"

# ---------------------------------------------------------------- tests 3, 4
# These cost GPU time. The defaults below are one ordering, which is about
# three hours on an A100. The cell that runs them prints its own estimate
# first, and every finished run is saved, so stopping and resuming is free.
#
# Set PROBE_ORDERS to all three orderings for the full version.
PROBE_ORDERS = ["order_A"]
PROBE_SEEDS = [0, 1, 2, 3, 4]

# Test 3 adds boundary 4 to the two points already measured. The grid has to
# match the one those points used exactly, or the three numbers cannot be put
# in one table: same rates, same repetitions, same step budget.
TEST3_POINTS = [4]
TEST3_RATES = [1e-4, 3e-4, 1e-3]
TEST3_REPETITIONS = 11

# Test 4 is the frozen-generator arm. It needs both ends to report a change, so
# it runs at boundary 0 and boundary 9. Boundary 0 needs no file: the seed
# rebuilds that model exactly.
#
# Five repetitions rather than eleven. This arm only has to answer "does the
# frozen-generator protocol give the same answer as the full one", and the gap
# the full arm reports is about 18 points, which five repetitions can resolve.
TEST4_POINTS = [0, 9]
TEST4_RATES = [1e-4, 3e-4, 1e-3]
TEST4_REPETITIONS = 5

# Held fixed, matching the published probe runs, so the new numbers sit in the
# same table as the old ones.
STEPS = 250
BATCH_SIZE = 64
FEATURE_IMAGES = 2000
TRAIN_PER_CLASS = 240
VAL_PER_CLASS = 20
SPLIT_SEED = 606
PROBE_TASKS = [PROBE_TASK]
CLASSES_HINT = 20

# The published probe file. This notebook reads it to find out which runs are
# already done, and never writes to it. New runs go to the file below, so a
# crash here cannot damage the fifteen streams already measured.
PUBLISHED_KEY = f"uncle/matched_probe/{STUDY_ID}/revised_20class_train4800.jsonl"
RESUME_KEY = f"uncle/matched_probe/{STUDY_ID}/revised_20class_train4800_extended.jsonl"
RUN_LABEL = "revised_20class_train4800"
SAVE_EVERY = 10

# The task each ordering unlearns at the end. Needed to rebuild a run's
# configuration, because the request list is part of what names the run.
FORGET = {"order_A": "0", "order_B": "4", "order_C": "8"}

DATA_ROOT = (Path("/content/data/tiny-imagenet-200") if IN_COLAB
             else REPO / "data/tiny-imagenet-200")
EUROSAT_ROOT = Path("/content/data") if IN_COLAB else REPO / "data"
CACHE = Path("/content/indicators") if IN_COLAB else REPO / "outputs/indicator-protocol"
CACHE.mkdir(parents=True, exist_ok=True)

READ_RUNS = [(o, s) for o in READ_ORDERS for s in READ_SEEDS]
PROBE_RUNS = [(o, s) for o in PROBE_ORDERS for s in PROBE_SEEDS]

print(f"Tests 1 and 2: {len(READ_RUNS)} runs x {len(READ_BOUNDARIES)} boundaries "
      f"= {len(READ_RUNS) * len(READ_BOUNDARIES)} model files "
      f"(about {len(READ_RUNS) * len(READ_BOUNDARIES) * 0.54:.0f} GB to download)")
t3 = len(PROBE_RUNS) * len(TEST3_POINTS) * len(TEST3_RATES) * TEST3_REPETITIONS
t4 = len(PROBE_RUNS) * len(TEST4_POINTS) * len(TEST4_RATES) * TEST4_REPETITIONS
print(f"Test 3: {t3} probe runs | Test 4: {t4} probe runs")
print(f"At about 0.58 minutes each, that is roughly {(t3 + t4) * 0.58 / 60:.1f} hours "
      f"of GPU time before anything already finished is skipped.")
''')

md("## 3. Connect to R2")

code(r'''
def secret(name):
    """Read a credential from Colab secrets, or the environment elsewhere."""
    if IN_COLAB:
        from google.colab import userdata
        return userdata.get(name)
    value = os.environ.get(name)
    if value is None:
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(name + "="):
                    return line.partition("=")[2].strip().strip('"').strip("'")
    return value


store = R2Store.connect(secret("R2_ENDPOINT"), BUCKET,
                        secret("R2_ACCESS_KEY_ID"), secret("R2_SECRET_ACCESS_KEY"))

# The settings the streams were trained with, read back rather than retyped, so
# the model this notebook rebuilds has the shape its checkpoint expects.
plan = json.loads(store.download(
    f"uncle/indicators/{STUDY_ID}/job2_summary.json", CACHE).read_text(encoding="utf-8"))
CLASSES = plan["classes_per_task"]
assert CLASSES == CLASSES_HINT, f"settings assume {CLASSES_HINT}, the runs used {CLASSES}"
print("Streams used learning rate", plan["learning_rate"], "| epochs", plan["epochs"],
      "| batch", plan["batch_size"], "| classes per task", CLASSES)

# Fail now rather than an hour in, if a model this session needs is missing.
missing = []
for order, seed in READ_RUNS:
    for point in READ_BOUNDARIES:
        key = f"uncle/checkpoints/{STUDY_ID}/{order}/seed_{seed}/boundary_{point}.pt"
        if store.info(key) is None:
            missing.append(key)
for order, seed in PROBE_RUNS:
    for point in set(TEST3_POINTS + TEST4_POINTS):
        if point == 0:
            continue
        key = f"uncle/checkpoints/{STUDY_ID}/{order}/seed_{seed}/boundary_{point}.pt"
        if store.info(key) is None:
            missing.append(key)
if missing:
    raise RuntimeError("Missing models:\n" + "\n".join(sorted(set(missing))))
print("Every model this session needs is in R2.")
''')

md(r"""
## 4. Build the pictures

The same 256 pictures are read at every boundary and in every run, drawn by a
generator seeded inside the library. If the pictures changed between two
readings, a change in the number could mean the model moved or the pictures
moved, and there would be no way to tell which.

- `in_stream` is held-out test images of the nine groups the model learns.
- `probe` is the tenth group, which no stream ever trains on.
""")

code(r'''
from torch.utils.data import ConcatDataset, Subset

STREAM_TASKS = [str(i) for i in range(9)]
needed = tuple(STREAM_TASKS + PROBE_TASKS)
tasks = build_tasks(root=DATA_ROOT, partition_path=PARTITIONS[20],
                    partition_seed=42, include=needed, download=True)
print("Loaded Tiny ImageNet images for", len(needed), "tasks")

if set(PROBE_TASKS) & set(STREAM_TASKS):
    raise ValueError("The probe task is in the learning stream, so it is not held out.")

sources = {
    "in_stream": ConcatDataset([tasks[name]["test"] for name in STREAM_TASKS]),
    "probe": tasks[PROBE_TASK]["test"],
}
if READ_EUROSAT:
    from torchvision.datasets import EuroSAT
    from torchvision.transforms import ToTensor
    sources["eurosat"] = EuroSAT(root=str(EUROSAT_ROOT), transform=ToTensor(),
                                 download=True)

image_sets = fixed_image_sets(sources, count=IMAGES_PER_SET)
for name, images in image_sets.items():
    print(f"{name:<10} {tuple(images.shape)}")
''')

md(r"""
## 5. Building a model at one boundary

This rebuilds the run exactly as it was configured, then loads the saved state
into it. `checkpoint.restore` also recreates the task codes, so after loading a
boundary-9 model the generator holds the nine codes that stream trained.
""")

code(r'''
def build(order, seed, point):
    """The model this stream had at `point`. Point 0 needs no file.

    Point 0 is the untrained generator. It is rebuilt from the seed rather than
    loaded, which is why no boundary-0 file exists anywhere in this study.
    """
    settings = study_defaults(order)
    settings["requests"] = settings["requests"] + (("forget", FORGET[order]),)
    config = Config(**settings, seed=seed,
                    learning_rate=plan["learning_rate"], epochs=plan["epochs"],
                    batch_size=plan["batch_size"],
                    initialization=plan["initialization"], device=DEVICE)
    target = build_target(config)
    hypernet = HyperNetwork(target, config)
    trainer = UnCLe(hypernet, config, target, tasks, progress=False)
    path = None
    if point > 0:
        key = f"uncle/checkpoints/{STUDY_ID}/{order}/seed_{seed}/boundary_{point}.pt"
        path = store.download(key, CACHE)
        payload = torch.load(path, map_location=DEVICE, weights_only=False)
        checkpoint.restore(payload, hypernet=hypernet, uncle=trainer)
        del payload
    return target, hypernet, trainer, path
''')

md(r"""
## 6. Tests 1 and 2: read every code, with and without centering

Two small functions repeat what the library does, so the uncentred numbers come
out identical to the published records and can be checked against them. The
only new thing is `centre`, which subtracts the average row before the
decomposition.

**What centering means.** Each row of the matrix is one picture's 2048
features. Subtracting the average row removes what every picture has in common
and leaves only how the pictures differ from each other. The study's own
numbers keep that common part in.
""")

code(r'''
def rank_pair(matrix, centre):
    """Effective rank and srank of one feature matrix.

    `matrix` has one row per picture and 2048 columns. Setting `centre` to True
    subtracts the average row first.

    Effective rank is the exponential of the entropy of the normalised singular
    values, from Roy and Vetterli 2007. srank counts how many singular values
    are needed to reach 99 percent of their total, from Kumar et al. 2021.
    Both are copied from the library rather than imported so that the centred
    and uncentred versions go through exactly the same arithmetic.
    """
    work = matrix.double()
    if centre:
        work = work - work.mean(dim=0, keepdim=True)
    values = torch.linalg.svdvals(work)
    total = values.sum()
    if total <= 0:
        return float("nan"), 0
    weights = values / total
    weights = weights[weights > 0]
    effective = float(torch.exp(-(weights * weights.log()).sum()).item())
    shares = torch.cumsum(values, 0) / total
    return effective, int((shares < 1 - SRANK_DELTA).sum().item()) + 1


def read_one_model(order, seed, point):
    """Every code and every picture set, for one saved model.

    Returns one record per (code, picture set). The reference code is the one
    the published records use. The task codes are the ones this stream trained.
    """
    target, hypernet, trainer, path = build(order, seed, point)
    try:
        codes = {"reference": reference_code(hypernet)}
        for task in hypernet.known_tasks():
            codes[f"task_{task}"] = hypernet.task_codes[task].detach()

        rows = []
        for code_name, code in codes.items():
            for set_name, images in image_sets.items():
                matrix = features(hypernet, target, trainer, images, code,
                                  batch_size=64)
                raw_e, raw_s = rank_pair(matrix, centre=False)
                mid_e, mid_s = rank_pair(matrix, centre=True)
                # How much of the matrix is the average row. If this is near 1,
                # the uncentred reading is mostly describing that average.
                norm = float(matrix.double().norm())
                mean_share = float(
                    matrix.double().mean(dim=0).norm()
                    * (matrix.shape[0] ** 0.5) / norm) if norm > 0 else float("nan")
                rows.append(dict(
                    order=order, seed=seed, point=point, code=code_name,
                    images=set_name, trained_code=code_name != "reference",
                    effective_rank=raw_e, srank=raw_s,
                    effective_rank_centred=mid_e, srank_centred=mid_s,
                    mean_share=mean_share, n_images=int(matrix.shape[0]),
                    code_revision=HEAD, device=GPU_NAME))
        return rows
    finally:
        del trainer, hypernet, target
        import gc
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        # One model file at a time, so fifteen runs do not fill the disk.
        if path is not None and Path(path).exists():
            Path(path).unlink()
''')

code(r'''
reads = []
if store.info(READ_KEY) is not None:
    text = store.download(READ_KEY, CACHE).read_text(encoding="utf-8")
    reads = [json.loads(l) for l in text.splitlines() if l.strip()]
    print(f"Resuming: {len(reads)} readings already saved.")

READ_PARTIAL = CACHE / "code_sweep_partial.jsonl"
READ_PARTIAL.write_text("".join(json.dumps(r) + "\n" for r in reads), encoding="utf-8")

began = time.perf_counter()
for order, seed in tqdm(READ_RUNS, desc="runs", unit="run"):
    for point in READ_BOUNDARIES:
        if any((order, seed, point) == (r["order"], r["seed"], r["point"])
               for r in reads):
            continue
        rows = read_one_model(order, seed, point)
        reads += rows
        with READ_PARTIAL.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        store.upload(READ_PARTIAL, READ_KEY, overwrite=True)
print(f"\n{len(reads)} readings in {(time.perf_counter() - began) / 60:.1f} minutes.")
''')

md(r"""
## 7. What tests 1 and 2 found

Three tables.

The first asks whether the published numbers reproduce. The reference-code,
uncentred reading here should match the indicator record for the same run and
boundary. If it does not, something in this notebook differs from the run and
nothing below can be trusted.

The second is test 1: the same model read through ten different codes.

The third is test 2: the same readings with and without the average row removed.
""")

code(r'''
import statistics
from collections import defaultdict


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


# ---- check against the published records ---------------------------------
print("CHECK: reference code, uncentred, against the indicator record")
print(f"{'run':>16}{'boundary':>10}{'this notebook':>16}{'record':>10}{'gap':>9}")
worst = 0.0
for order, seed in READ_RUNS[:5]:
    key = indicator_key(STUDY_ID, order, seed)
    rows = {int(json.loads(l)["t"]): json.loads(l) for l
            in store.download(key, CACHE).read_text(encoding="utf-8").splitlines()
            if l.strip()}
    for point in READ_BOUNDARIES:
        here = [r for r in reads if (r["order"], r["seed"], r["point"], r["code"],
                                     r["images"]) == (order, seed, point,
                                                      "reference", "in_stream")]
        if not here or point not in rows:
            continue
        there = rows[point]["effective_rank_in_stream"]
        gap = abs(here[0]["effective_rank"] - there)
        worst = max(worst, gap)
        print(f"{order + ' s' + str(seed):>16}{point:>10}"
              f"{here[0]['effective_rank']:>16.4f}{there:>10.4f}{gap:>9.4f}")
print(f"Largest gap: {worst:.4f}. Anything above about 0.01 means this notebook "
      f"is not reading what the run read.")

# ---- test 1: the reading depends on the code -----------------------------
print()
print("TEST 1: the same model, read through different codes (in_stream images)")
print(f"{'boundary':>10}{'code':>14}{'effective rank':>17}{'srank':>10}{'runs':>7}")
for point in READ_BOUNDARIES:
    ref = [r["effective_rank"] for r in reads
           if r["point"] == point and r["code"] == "reference"
           and r["images"] == "in_stream"]
    trained = [r["effective_rank"] for r in reads
               if r["point"] == point and r["trained_code"]
               and r["images"] == "in_stream"]
    ref_s = [r["srank"] for r in reads if r["point"] == point
             and r["code"] == "reference" and r["images"] == "in_stream"]
    tr_s = [r["srank"] for r in reads if r["point"] == point
            and r["trained_code"] and r["images"] == "in_stream"]
    print(f"{point:>10}{'reference':>14}{mean(ref):>17.3f}{mean(ref_s):>10.1f}{len(ref):>7}")
    print(f"{point:>10}{'trained mean':>14}{mean(trained):>17.3f}{mean(tr_s):>10.1f}{len(trained):>7}")
if reads:
    by_reference = mean(r["effective_rank"] for r in reads
             if r["code"] == "reference" and r["images"] == "in_stream")
    by_trained = mean(r["effective_rank"] for r in reads
             if r["trained_code"] and r["images"] == "in_stream")
    print(f"\nTrained codes read {by_trained / by_reference:.1f} times higher than the reference.")
    print("If that number is far from 1, the published indicator is describing")
    print("the code it was read through, not the model.")

# ---- the trend over the stream, inside one code --------------------------
print()
print("TEST 1b: the trend from boundary 4 to boundary 9, per reading route")
print("Only codes that exist at both boundaries are used, so this compares a")
print("code against itself rather than against a different code.")
if set(READ_BOUNDARIES) >= {4, 9}:
    by = defaultdict(dict)
    for r in reads:
        if r["images"] == "in_stream":
            by[(r["order"], r["seed"], r["code"])][r["point"]] = r["effective_rank"]
    for label, keep in (("reference code", lambda c: c == "reference"),
                        ("trained task codes", lambda c: c != "reference")):
        pairs = [(v[4], v[9]) for (o, s, c), v in by.items()
                 if keep(c) and 4 in v and 9 in v]
        if not pairs:
            continue
        rose = sum(1 for a, b in pairs if b > a)
        print(f"  {label:<20} boundary 4 {mean(a for a, _ in pairs):7.3f}"
              f"   boundary 9 {mean(b for _, b in pairs):7.3f}"
              f"   rose in {rose}/{len(pairs)}")

# ---- test 2: centering ---------------------------------------------------
print()
print("TEST 2: the same readings, with and without the average row removed")
print(f"{'code':>14}{'images':>11}{'uncentred':>12}{'centred':>10}"
      f"{'srank unc.':>12}{'srank ctr.':>12}{'mean share':>12}")
for code_group, keep in (("reference", lambda r: r["code"] == "reference"),
                         ("trained", lambda r: r["trained_code"])):
    for set_name in sorted({r["images"] for r in reads}):
        rows = [r for r in reads if keep(r) and r["images"] == set_name]
        if not rows:
            continue
        print(f"{code_group:>14}{set_name:>11}"
              f"{mean(r['effective_rank'] for r in rows):>12.3f}"
              f"{mean(r['effective_rank_centred'] for r in rows):>10.1f}"
              f"{mean(r['srank'] for r in rows):>12.1f}"
              f"{mean(r['srank_centred'] for r in rows):>12.1f}"
              f"{mean(r['mean_share'] for r in rows):>12.3f}")
print()
print("'mean share' is how much of the matrix is the average row, between 0 and 1.")
print("Near 1 means almost every picture produces nearly the same features, and")
print("the uncentred effective rank is mostly reporting that one direction.")
''')

md(r"""
## 8. Tests 3 and 4: the probe at the middle of the stream, and with the generator frozen

From here the notebook trains. Each probe forks the model, teaches it the
held-out group of twenty classes, records the accuracy, and throws the fork
away. Guessing scores 5 percent.

Two arms run here:

- `full` trains the generator and a fresh code together. This is what every
  published probe record actually did. Test 3 adds boundary 4 to it.
- `code_only` freezes the generator and trains only the fresh code. This is
  what the paper describes. Test 4 measures it at both ends.

The published file is read but never written. New runs go to their own file.
""")

code(r'''
# The probe's own training, validation and test splits. Validation picks the
# learning rate and is never trained on, so a rate is never chosen on the same
# images that score it.
def balanced_split(labels, counts, seed):
    """Row numbers for disjoint splits, equal numbers from every class."""
    by_class = {}
    for row, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(row)
    if len(by_class) != CLASSES:
        raise RuntimeError(f"Expected {CLASSES} classes, found {len(by_class)}.")
    stream = torch.Generator().manual_seed(seed)
    splits = [[] for _ in counts]
    for label, rows in sorted(by_class.items()):
        if len(rows) < sum(counts):
            raise RuntimeError(f"Class {label} has {len(rows)} images, needs {sum(counts)}.")
        order = [rows[j] for j in torch.randperm(len(rows), generator=stream).tolist()]
        at = 0
        for which, how_many in enumerate(counts):
            splits[which] += order[at:at + how_many]
            at += how_many
    taken = [row for split in splits for row in split]
    assert len(set(taken)) == len(taken), "An image landed in two splits."
    return splits


probe_sets = {}
for task in PROBE_TASKS:
    base = tasks[task]["train"]
    train_rows, val_rows = balanced_split(
        [base[i][1] for i in range(len(base))],
        [TRAIN_PER_CLASS, VAL_PER_CLASS], SPLIT_SEED)
    probe_sets[task] = {"train": Subset(base, train_rows),
                        "val": Subset(base, val_rows),
                        "test": tasks[task]["test"]}
    print(f"probe task {task}: {len(train_rows)} train, {len(val_rows)} validation, "
          f"{len(probe_sets[task]['test'])} test")
''')

code(r'''
# Repeated here so this half of the notebook runs even if section 7 was
# skipped. Section 7 defines both already; defining them twice is harmless.
from collections import defaultdict


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def identity(run):
    """Everything that makes one probe run different from another."""
    return (run["order"], run["seed_label"], run["point"], run["task"],
            run["learning_rate"], run["arm"], run["repetition"])


def probe_seed(task, repetition):
    """Fixes the batch order and the fresh code.

    Depends on the task and the repetition, never on the probe point or the
    arm, so two points differ only by the state of the model being probed.
    This is the same rule notebook 47 used, so the new runs line up with the
    old ones.
    """
    return 12345 + 1000 * repetition + (int(task) if task.isdigit() else 99)


# Everything already measured, from both files. The published file is read
# only, so a run it already holds is never repeated.
done = set()
published = 0
if store.info(PUBLISHED_KEY) is not None:
    for line in store.download(PUBLISHED_KEY, CACHE).read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(identity(json.loads(line)))
            published += 1
print(f"{published} runs already in the published file.")

results = []
if store.info(RESUME_KEY) is not None:
    for line in store.download(RESUME_KEY, CACHE).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                print("Skipped one unreadable line at the end of the saved file.")
                continue
            results.append(run)
            done.add(identity(run))
    print(f"{len(results)} runs already in this notebook's file.")

PARTIAL = CACHE / "extended_partial.jsonl"
PARTIAL.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")

# One flat list of everything still to do, so the progress bar is honest about
# how much is left rather than restarting at every model.
TODO = []
for order, seed in PROBE_RUNS:
    for point in TEST3_POINTS:
        for rate in TEST3_RATES:
            for rep in range(TEST3_REPETITIONS):
                TODO.append((order, seed, point, rate, "full", False, True, rep))
    for point in TEST4_POINTS:
        for rate in TEST4_RATES:
            for rep in range(TEST4_REPETITIONS):
                TODO.append((order, seed, point, rate, "code_only", True, True, rep))
TODO = [t for t in TODO
        if (t[0], t[1], t[2], PROBE_TASKS[0], t[3], t[4], t[7]) not in done]
print(f"{len(TODO)} probe runs still to do, roughly "
      f"{len(TODO) * 0.58 / 60:.1f} hours.")
''')

code(r'''
new_runs = 0
began = time.perf_counter()
bar = tqdm(total=len(TODO), desc="probe runs", unit="run")

# Grouped by model, so each checkpoint is downloaded and built once rather than
# once per probe run.
by_model = defaultdict(list)
for item in TODO:
    by_model[(item[0], item[1], item[2])].append(item)

for (order, seed, point), items in by_model.items():
    target, hypernet, trainer, path = build(order, seed, point)
    reference = snapshot(hypernet, trainer)
    try:
        for order_, seed_, point_, rate, arm, code_only, train_code, rep in items:
            task = PROBE_TASKS[0]
            bar.set_postfix_str(f"{order} s{seed} t={point} {arm} lr={rate:g} rep {rep + 1}")
            started = time.perf_counter()
            run = probe_once(
                hypernet, target, trainer, task=task,
                train_set=probe_sets[task]["train"], test_set=probe_sets[task]["test"],
                validation_set=probe_sets[task]["val"],
                steps=STEPS, learning_rate=rate, batch_size=BATCH_SIZE,
                code_only=code_only, train_code=train_code,
                seed=probe_seed(task, rep), feature_images=FEATURE_IMAGES)
            # A probe forks and discards. If it changed the model, every later
            # reading in this session is measuring something that has moved.
            if not unchanged(hypernet, reference):
                raise RuntimeError("A probe changed the model. Stop and fix this "
                                   "before trusting anything.")
            run.update(order=order, point=point, endpoint="E2", repetition=rep,
                       seed_label=seed, minutes=(time.perf_counter() - started) / 60,
                       device=GPU_NAME, run_label=RUN_LABEL, code_revision=HEAD,
                       train_images=len(probe_sets[task]["train"]))
            results.append(run)
            done.add(identity(run))
            with PARTIAL.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(run) + "\n")
            new_runs += 1
            bar.update(1)
            if new_runs % SAVE_EVERY == 0:
                store.upload(PARTIAL, RESUME_KEY, overwrite=True)
    finally:
        del trainer, hypernet, target, reference
        import gc
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if path is not None and Path(path).exists():
            Path(path).unlink()
        store.upload(PARTIAL, RESUME_KEY, overwrite=True)

bar.close()
print(f"\nFinished in {(time.perf_counter() - began) / 3600:.2f} hours. "
      f"{new_runs} new runs.")
''')

md(r"""
## 9. What tests 3 and 4 found

Test 3 puts boundary 4 between the two points already measured. If probe
accuracy rises from 0 to 4 and then falls from 4 to 9, the ability to learn
does decline during the stream and the two endpoints were hiding it. If it
rises all the way, it does not.

Test 4 reports the frozen-generator arm beside the full arm at the same two
points.
""")

code(r'''
all_runs = results[:]
if store.info(PUBLISHED_KEY) is not None:
    for line in store.download(PUBLISHED_KEY, CACHE).read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_runs.append(json.loads(line))


def selected(order, seed, point, arm):
    """One accuracy per repetition: the rate with the best validation score.

    The rate is chosen inside each repetition, not across them, which is what
    notebook 47 did. Ties go to the lower rate.
    """
    out = []
    for rep in range(20):
        runs = [r for r in all_runs
                if r["order"] == order and r["seed_label"] == seed
                and r["point"] == point and r["arm"] == arm
                and r["repetition"] == rep]
        if runs:
            out.append(min(runs, key=lambda r: (-r["val_accuracy"],
                                                r["learning_rate"]))["test_accuracy"])
    return out


print("TEST 3: the probe at the middle of the stream (arm = full)")
print(f"{'run':>16}{'t=0':>9}{'t=4':>9}{'t=9':>9}{'0 to 4':>9}{'4 to 9':>9}")
mid = []
for order, seed in PROBE_RUNS:
    at = {p: selected(order, seed, p, "full") for p in (0, 4, 9)}
    if not all(at.values()):
        print(f"{order + ' s' + str(seed):>16}   incomplete")
        continue
    m = {p: mean(v) for p, v in at.items()}
    mid.append((m[4] - m[0], m[9] - m[4]))
    print(f"{order + ' s' + str(seed):>16}{m[0]:>9.2f}{m[4]:>9.2f}{m[9]:>9.2f}"
          f"{m[4] - m[0]:>+9.2f}{m[9] - m[4]:>+9.2f}")
if mid:
    print(f"{'mean':>16}{'':>27}{mean(a for a, _ in mid):>+9.2f}"
          f"{mean(b for _, b in mid):>+9.2f}")
    fell = sum(1 for _, b in mid if b < 0)
    print(f"\nProbe accuracy fell from boundary 4 to boundary 9 in {fell} of {len(mid)} runs.")
    print("If that count is high, the ability to learn does decline during the")
    print("stream, and the two endpoints were hiding it.")

print()
print("TEST 4: the frozen-generator arm beside the full arm")
print(f"{'run':>16}{'full t=0':>11}{'full t=9':>11}{'frozen t=0':>13}{'frozen t=9':>13}")
gaps = []
for order, seed in PROBE_RUNS:
    row = {}
    for arm in ("full", "code_only"):
        for p in (0, 9):
            v = selected(order, seed, p, arm)
            row[(arm, p)] = mean(v) if v else None
    if any(v is None for v in row.values()):
        print(f"{order + ' s' + str(seed):>16}   incomplete")
        continue
    gaps.append((row[("full", 9)] - row[("full", 0)],
                 row[("code_only", 9)] - row[("code_only", 0)]))
    print(f"{order + ' s' + str(seed):>16}{row[('full', 0)]:>11.2f}"
          f"{row[('full', 9)]:>11.2f}{row[('code_only', 0)]:>13.2f}"
          f"{row[('code_only', 9)]:>13.2f}")
if gaps:
    print(f"\nChange over the stream: full arm {mean(a for a, _ in gaps):+.2f} points, "
          f"frozen arm {mean(b for _, b in gaps):+.2f} points.")
    print(f"Guessing scores {100 / CLASSES:.0f}%. If the frozen arm sits near that at")
    print("both points, it cannot measure a change and the paper has to report")
    print("the full arm and describe it correctly.")
''')

md(r"""
## 10. What to do with these answers

**If test 1 shows trained codes read far higher than the reference code**, then
the published indicator numbers describe one arbitrary untrained code, not the
generator. The paper's claim becomes a claim about reading protocol, which is a
stronger and more specific contribution than "the indicator does not predict
plasticity".

**If test 2 shows the centred rank is large and flat**, then the published
values near 1 are the average feature vector and not a collapsed
representation. Every rank number in the study would need restating.

**If the centred rank is also small**, the representation is genuinely
low-dimensional. That does not rescue the indicator. It sharpens the question
to why a model using a handful of directions learns a new group of twenty
classes as well as one using two hundred, which is what the saved records
already show.

**If test 3 shows probe accuracy falling from boundary 4 to boundary 9**, there
is a real decline inside the stream, and the paper's disagreement survives in a
narrower form. If it rises, the disagreement was an artifact of comparing
against an untrained model.

**If test 4 shows the frozen arm near chance at both points**, the paper cannot
describe its probe as a frozen generator. It has to report full adaptation and
say so.

Nothing here is written into the experiment records automatically. Paste the
printed tables into the day's record in `docs/wiki/`.
""")

cells = [{"cell_type": t,
          "metadata": {},
          "source": s.splitlines(keepends=True)}
         for t, s in C]
for cell in cells:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path("C:/LangChain/Algoverse/continual-learning/hypernetwork/notebooks/"
           "48_what_the_indicators_read.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes,", len(cells), "cells")
