# Experiment trajectory: 12 November 2026

Back to the [index](Experiment-trajectory).

## The class split changes to ten groups of twenty, and every old run stops being reusable

No training ran today. This is a code change only, made so the revised design
in `ops-docs/plasticity-methodology.md` can be run at all. The runs it unblocks
have not started and are waiting on a compute budget decision.

### What the design asks for

All 200 Tiny ImageNet classes are used. They split into ten disjoint groups of
twenty instead of twenty groups of ten. Nine groups are learned one after
another. The tenth group is never trained on and is read as the `probe` image
set, the in-distribution probe. EuroSAT stays as the out-of-distribution set.
Readings happen at ten boundaries: before task 1, then after each of the nine.

Because a task now holds twenty classes, random guessing scores 5 percent, not
10. Every claim that an unlearned task "sits at chance" now means 5 percent.

### What changed in the code

The split itself is in `uncle/tasks.py`. `load_partition` now cuts the permuted
class list into ten blocks of twenty. The permutation is unchanged: sorted
WordNet IDs, NumPy PCG64 seeded at 42, independent of any training seed. The
guard that compares every later run against the saved file is unchanged too.

The new split is written to a new file, `uncle/task_partition_10x20.json`, and
the old `uncle/task_partition.json` is untouched. Old study plans name that
file and record its hash, so overwriting it would have made every earlier run
unreproducible. The new file is marked version 2 so the two cannot be confused.

`uncle/config.py` carries this study's three task orderings, which are new:

| Name | Sequence | Position 1 / 5 / 9 |
| --- | --- | --- |
| `order_A` | `L0 L1 L2 L3 L4 L5 L6 L7 L8` | 0 / 4 / 8 |
| `order_B` | `L4 L1 L2 L3 L8 L5 L6 L7 L0` | 4 / 8 / 0 |
| `order_C` | `L8 L1 L2 L3 L0 L5 L6 L7 L4` | 8 / 0 / 4 |

Tasks 1, 2, 3, 5, 6 and 7 hold the same position in all three. Tasks 0, 4 and 8
rotate through positions 1, 5 and 9, so each of the three is learned first
once, in the middle once, and last once. Task 9 is the held-out probe group and
appears in no ordering. A new function, `study_defaults`, builds a run's
settings from an ordering name.

`classes_per_task` moves from 10 to 20. The same value sets the generated
classifier's output width in `uncle/hypernet.py`, so checkpoints trained under
the old split cannot be loaded by a model built under the new one. There is no
resuming from the earlier runs. Every arm starts from scratch.

### The trap that was found and avoided

The orderings used in the earlier runs were not chosen for this study. They are
the learn-only prefix of the UnCLe paper's Table 4, Tiny ImageNet sequence 1.
Dropping the unlearn requests from `L3 L0 U3 L9 L5 L17 L1 L7 L14 L15 ...` gives
3, 0, 9, 5, 17, 1, 7, 14, which is exactly the `order_01` of the earlier runs.
Those sequences address task IDs 0 through 19, which the ten-group split no
longer has. The three orderings above replace them for this study and are not
the paper's sequences. The paper's own sequences are left in place for
reproduction work, and they still need the old 20 x 10 partition file.

### What was checked

The partition test was written first and watched fail, with 20 groups where the
design calls for 10. It now asserts ten groups of twenty, all 200 classes
present, none repeated and none missing. The generated file was then read back
directly: ten groups, every group twenty classes, 200 unique WordNet IDs.

A second new test checks the ordering structure rather than trusting it. Six
tasks must hold position in all three orderings and three must rotate through
positions 1, 5 and 9, each taking each position once. Getting that wrong would
not crash anything. It would quietly turn the order comparison into a
comparison of three different task mixes.

Three hard-coded "ten classes" assumptions turned up and were corrected:
capping images per task kept only ten labels, the learning test measured
against 10 percent chance, and the probe module's note on which external
datasets can fit the generated classifier. A fourth was found in the per-task
class count, which the `DATASETS` table did not carry: without that, a Permuted
MNIST run would have silently inherited Tiny ImageNet's twenty outputs for a
ten-class problem.

`pytest tests/` is green except for one failure that predates this work,
`test_experiment_site.py::test_site_links_and_accessible_figures`, which is
about a missing `paired-study.html` asset and is unrelated to the split.

### One deliberate departure from the written plan

The fork brief asked for `task_count` on the `tiny_imagenet` entry to move from
20 to 10, and in the same list asked for the paper's sequences to be left alone
because the unlearning experiments use them. Both cannot hold. `task_count` is
what sizes the task list for those sequences, and at 10 all three of them stop
building, because they name tasks up to 17. It was left at 20, and this study's
nine-task orderings get their task list from `study_defaults` instead. The
split's group count lives in `uncle/tasks.py`, where the split is made.

### Not changed, on purpose

`uncle/paired_study.py` still holds the old eight-task orderings and the
twenty-task config of the paired generalization study. That study is finished
and its records name task IDs that the new split does not have. Rewriting it
would not make its old results readable and would break their stored hashes.
The report-building scripts that draw a 10 percent chance line are describing
those earlier runs and are correct as they stand.

### The sweep is built and ready to run

The compute budget was confirmed, so phase 2a is prepared:
`notebooks/43_learning_rate_sweep.ipynb`, runs 1 to 5 of the thirteen. Five
short runs, one per candidate learning rate, each learning only the first three
tasks of `order_A` at seed 0. Everything else is held fixed, so the learning
rate is the only thing that differs between them: five epochs per task, batch
size 64, ResNet50 written by the generator, beta 0.01, hyperfan-in
initialization.

The candidates are 3e-4, 5e-4, 1e-3, 2e-3 and 4e-3, centred on the current
default of 1e-3. All five are written to `sweep_summary.json` whether chosen or
rejected, because the methodology reports the tuning runs in full.

The sweep reads no plasticity indicator and writes nothing to R2. It only needs
accuracy. Boundaries, fixed image sets and the held-out probe group belong to
the nine-task runs.

Nothing has been trained yet. The notebook's logic was exercised end to end on
the CPU at toy scale, with a small CNN, one epoch and 200 images per task, to
check the wiring rather than the science: the configuration builds from
`study_defaults`, the run completes, the mean accuracy over the learned tasks
comes out, and the summary file is written. Both toy values landed near 5
percent, which is chance for twenty classes and is what a model trained on 200
images for one epoch should score.

The first run prints its own wall clock and projects the rest of the sweep and
the seven nine-task runs from it. That projection is the point at which to stop
if the real cost is worse than the estimate below.

### Next decision

Phase 2 is thirteen training runs: five short learning-rate sweep runs on the
first three tasks of `order_A`, five full nine-task seeds of `order_A`, one run
each of `order_B` and `order_C`, and one unlearn step continuing the first
`order_A` seed. A task now holds 10,000 training images instead of 5,000, so
the nine-task runs alone are about 3.1 times the old study, and about 3.9 times
with the sweep. Nothing starts until the compute budget is confirmed.
