# Experiment trajectory: 13 November 2026

Back to the [index](Experiment-trajectory).

## The split change broke two things, and the two run notebooks are written

No training ran today. This is a code fix and two new notebooks. The runs they
drive have not started and are still waiting on the compute budget decision.

## Two defects in the 12 November split change

The split change of 12 November was reviewed today. The split itself holds up:
a new partition file rather than an edit of the old one, the version bumped to
2, the verify-against-saved-file guard kept, tests written first, and chance
accuracy updated from 10 to 5 percent at every call site. Two defects were
found, both confirmed by running the code.

**`main.py --dataset tiny_imagenet` stopped working.** The default partition
path moved to the ten-group file for every caller, while `dataset_defaults`
still returned twenty task names. The run died with `The partition has no task
named ['10', ..., '19']`.

**The old split became unreadable even when its path was passed explicitly.**
This was wider than the first review said. `load_partition` built the split it
expected from a module constant of twenty classes per task, so handing it the
version-1 file produced a mismatch and an error. `scripts/norm_control.py` and
the paired-study notebooks pass that path and were broken too. The 12 November
commit message said the paper rows would keep building. They did not.

**A bare `Config()` gave Permuted MNIST twenty classes per task.** MNIST has
ten. Nothing failed when this happened: the generated head simply grew ten
outputs that no label could reach, and chance would have been reported at 5
percent instead of 10.

### The fix

The group size now travels with the partition path instead of living in a
separate constant. `load_partition` reads it off the file name, so both saved
splits load again, and `build_task` takes its task count from the partition it
loaded. `uncle/data.py` picks the file from `config.classes_per_task`, which is
the one value that already says which split a run wants.

The dataclass default returns to ten, matching the default dataset, and a task
can no longer ask for more classes than the dataset holds. Fewer is still
allowed: short CPU checks use four classes on purpose, and a strict allowlist
broke 97 tests before this was understood.

`dataset_defaults` now returns ten classes per task for Tiny ImageNet, which is
what twenty tasks means, and `study_defaults` carries the twenty of the
ten-group split. One test had asserted the broken pairing of twenty tasks of
twenty classes, which is 400 classes and more than the dataset has, so it was
locking the defect in. It was corrected and two tests were added.

Suite at the end: 318 pass, 1 skipped, and the pre-existing
`test_experiment_site.py` failure that predates this work.

## The three run notebooks

The revised study is three notebook jobs, not thirteen separate launches. Each
job builds the dataset once and loops inside itself, because indexing the
dataset costs more than the first optimizer steps do.

| Job | Notebook | Contains | Task-trainings | Where |
| --- | --- | --- | --- | --- |
| 1 | `43_learning_rate_sweep` | 5 learning rates, seed 0, first 3 tasks of `order_A` | 15 | Colab |
| 2 | `44_order_A_five_seeds` | `order_A` seeds 0 to 4, each ending with an unlearn of task 0 | 45 | RunPod |
| 3 | `45_orders_B_and_C` | `order_B` and `order_C`, seed 0 each, each ending with its own unlearn | 18 | RunPod |

Notebook 43 already existed. Notebooks 44 and 45 were written today.

The split between the two machines is by run length, not by arm. The sweep is
three tasks per run and finishes inside one Colab session. The two long jobs
go to RunPod, where the machine does not disconnect. Everything after them
trains nothing and returns to Colab: the rescaling control is four forward
passes per pair, and the tables and figures only read files.

### What jobs 2 and 3 record

Both notebooks attach the plasticity indicators at every boundary, through
hooks `uncle.experiment.run` already exposed, so no library change was needed.
`run_experiment` could not be used here: it keeps `on_start` and
`on_saved_checkpoint` for its own purposes and builds its own task dictionary.

Eleven boundaries per run: before any task, after each of the nine, and after
the unlearn. At each one, effective rank, srank and dormant units at all three
cut-offs, on all three image sets, plus the generator and target norms,
distance travelled and angle.

The indicator record and its baseline are uploaded to R2 after every boundary,
not at the end, and a run resumes from its own record. A dropped RunPod
instance therefore costs one task rather than one run. Three boundary models
per run are kept as files, at boundaries 4, 9 and 10, which is what the
rescaling control needs: the model before the unlearn and the model after it.

One gap is recorded in both notebooks. The within-task loss progress curve that
notebook 36 collected is not collected here, because this code path has no
per-update hook. Every boundary indicator the paper uses is collected.

### The unlearn now runs on all seven streams

The old study unlearned one model, which is why the weight-scale result of 8
November rests on a single pair of checkpoints. An unlearn is 100 update steps
against nine tasks of training, so appending one to every run costs 700 steps
in total and gives seven pairs across three orderings and five seeds. Each run
unlearns the task it learned first, matching what the earlier study did.

### The gate between job 2 and job 3

Notebook 44 ends by checking the methodology's target: five seeds, differing
only in random start and data order, within 3 percentage points of each other
on mean accuracy over the nine tasks. If the spread is wider, job 3 does not
start. The reason is a floor. If two seeds of the identical setting can differ
by ten points, a ten-point difference between two orderings means nothing.

Notebook 45 checks the rotation before it trains: tasks 0, 4 and 8 must each
visit positions 1, 5 and 9 across the three orderings, and the other six must
never move. If somebody edits the orderings and breaks the pattern, the run
stops rather than producing a comparison that cannot separate the order from
the task.

## Evidence

- `7ef57f0` the partition fix and its tests
- `add9c6d` notebooks 44 and 45
- `docs/plasticity/methodology.md`, `docs/plasticity/run-plan.md`

## Next decision

The compute budget. The projection is printed inside notebook 44 after its
first seed finishes, so the job can be stopped there rather than after all
five. Nothing else is blocked: notebook 43 can run on Colab now.
