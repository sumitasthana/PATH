# Experiment run plan

The brief for the code-writing fork. Point that chat at this file.

Companions: `docs/plasticity/methodology.md` is the specification this
implements, and `docs/plasticity/paper-brief.md` is the paper fork's brief.

## What this fork is for

Change the experiment code so it runs the revised design: all 200 Tiny ImageNet
classes, nine learned tasks of twenty classes, one held-out task as an
in-distribution probe, and a tuned learning rate. Then run the tuning sweep.

The written specification is `docs/plasticity/methodology.md`. Read it
first. It is the contract. If the code and that file disagree, one of them is
wrong and the user decides which.

## The design to implement

- 200 classes, ten disjoint groups of twenty. Nothing dropped.
- Nine groups form the learning stream.
- The tenth group is held out, never trained on, read as the `probe` image set.
- EuroSAT stays as the `eurosat` out-of-distribution image set.
- Readings at ten boundaries: before task 1, then after each of the nine.
- Three task orderings. Six tasks keep the same position in all three; three
  tasks rotate through positions 1, 5 and 9.
- Five seeds of the full stream, required to finish within 3 percentage points
  of each other on mean accuracy over the nine tasks.
- One unlearn arm: a tenth step on one finished run that deliberately destroys
  a single task, used to test whether the indicators detect certain damage.

Chance accuracy for an unlearned task is now 5 percent, not 10, because tasks
have twenty classes.

## Current state of the code

`uncle/tasks.py`, `load_partition()`. Builds **20 groups of 10**. It sorts the
WordNet IDs, permutes with NumPy PCG64 seeded at 42, and cuts into blocks of
ten. It writes `uncle/task_partition.json` and then **verifies every later run
against that saved file, raising if it differs**. That guard is deliberate and
should survive the change, not be removed.

`uncle/config.py`. `classes_per_task: int = 10`. The `tiny_imagenet` entry in
`DATASETS` has `task_count: 20`. `learning_rate: float = 1e-3`, `epochs: 5`,
`batch_size: 64`, `beta: 0.01` for Tiny ImageNet.

`uncle/streams.py`, `build_tasks()`. Turns the saved partition into the task
dict the trainer wants. It only indexes the classes the requested tasks name,
which is a real speed optimization, so keep that behaviour.

`uncle/plasticity_indicators.py`. The boundary measurement: reference code at
seed 20261030, fixed BatchNorm statistics, evaluation mode forced, features
hooked at the input to the final classifier, effective rank, srank, dormant
fraction, weight norm and angle. This is the part the paper's claims rest on.
It should need little or no change, but confirm the boundary count and the
image sets still line up.

`uncle/indicator_report.py`. Has `NEAR_TO_FAR = ("in_stream", "probe",
"eurosat")`, the three image sets.

## The trap

The task orderings in the old runs were **not** chosen for this study. They are
the learn-only prefix of the UnCLe paper's Table 4 sequence 1:

```
TINY_IMAGENET_SEQUENCES[1] = "L3 L0 U3 L9 L5 L17 L1 L7 L14 L15 ..."
```

Strip the unlearn requests and the first eight learns are 3, 0, 9, 5, 17, 1, 7,
14, which is exactly the draft's `order_01`.

Those sequences address task IDs 0 through 19. Under a ten-group split only IDs
0 through 9 exist, so **every one of the three Tiny ImageNet sequences in
`uncle/config.py` becomes invalid**. New orderings have to be defined for this
study, satisfying the six-fixed / three-rotating structure above. They are no
longer the paper's sequences and the methodology should say so.

Check `parse_sequence` callers and the tests in `tests/` before changing the
sequence constants, since the unlearning experiments also use them.

## Other things that will break

Changing the partition invalidates **every saved checkpoint**. A checkpoint
trained on groups of ten cannot be read by a model expecting groups of twenty:
the generated classifier layer changes width. Plan for a clean re-run rather
than resuming.

`classes_per_task` feeds the generated network's output width in
`uncle/hypernet.py` and the chance-accuracy figures in `uncle/baseline.py`,
`uncle/experiments.py` and `uncle/paired_diagnostic.py`. Chance accuracy moves
from 10 percent to 5 percent. Any hard-coded 10 percent is now wrong.

Search for the old numbers before declaring the change done. There is an
`ops-docs/why-ten-percent/` directory whose whole subject is the 10 percent
figure.

## Phase 1: code changes, before any run

1. **Failing test first.** Assert ten groups of twenty, all 200 classes
   covered, none repeated, none missing. Watch it fail, then fix it.
2. `load_partition`: cut into ten blocks of twenty. Keep the
   verify-against-saved-file guard. Write to a **new** path, do not overwrite
   `uncle/task_partition.json`, so the old runs stay reproducible.
3. `classes_per_task` to 20. `tiny_imagenet` `task_count` to 10.
4. Add the three orderings below as new sequence constants. Leave
   `TINY_IMAGENET_SEQUENCES` alone; the unlearning experiments still use it.
5. Fix the chance-accuracy call sites: 10 percent becomes 5 percent.
6. Per-boundary checkpoints: `uncle/experiment.py:138` currently overwrites one
   path. Make it write `t00.pt` through `t09.pt`. See Phase 2 for why this is
   the highest-value change in the whole brief.
7. `pytest tests/` green before any GPU time is spent.

## The three orderings

Task IDs 0 to 8 are the stream. **Task 9 is held out** and never trained on,
read as the `probe` image set.

Tasks 1, 2, 3, 5, 6, 7 hold the same position in all three orderings. Tasks 0,
4 and 8 rotate through positions 1, 5 and 9.

| Name | Sequence | Position 1 / 5 / 9 |
|---|---|---|
| `order_A` | `L0 L1 L2 L3 L4 L5 L6 L7 L8` | 0 / 4 / 8 |
| `order_B` | `L4 L1 L2 L3 L8 L5 L6 L7 L0` | 4 / 8 / 0 |
| `order_C` | `L8 L1 L2 L3 L0 L5 L6 L7 L4` | 8 / 0 / 4 |

A clean three-cycle: each of the three rotating tasks is learned first once,
middle once, last once. Six tasks never move.

## Phase 2: the runs

This is a collection phase, not an exploration phase. The framework is settled
and the job is to come away with every number the paper needs, in as few passes
over the data as possible. Two principles follow.

**Never pay for a training run twice.** Anything that can be read from a saved
checkpoint later should be, rather than re-run.

**Measure everything at every boundary, not just what the paper currently
needs.** A boundary reading costs milliseconds. A forgotten one costs a
nine-task re-run.

### The one change that matters most

`uncle/experiment.py:138` already saves a checkpoint after every request. It
saves to the same `checkpoint_path` every time, so each save overwrites the
last and only the final state survives.

Change it to write one file per boundary, `t00.pt` through `t09.pt`. That is a
small change and it converts every measurement you forgot to take from a
re-run into a re-read. The adapter study already does exactly this with its
`--ap_ckpt_dir` flag, so there is a working pattern in this repository to copy.

With per-boundary checkpoints you also get crash resilience for free, which a
nine-task run on a disconnecting Colab runtime needs.

Check the size of one checkpoint before launching everything. Seventy of them
may or may not fit wherever you are putting them.

### Record at every boundary

All of this, in the same forward pass, every boundary, every run:

- effective rank, srank, dormant fraction at all three cut-offs
- all three image sets: `in_stream`, `probe`, `eurosat`
- weight norm and angle
- per-task accuracy for every task seen so far

Do not prune this list to what the current draft reports. Pruning is what
turns a missing column into another week of GPU time.

### Three jobs, not thirteen launches

Building the task data indexes the dataset, and `uncle/streams.py` calls that
the largest cost before the first optimizer step. Build it once per job and
reuse it across every run inside the job.

| Job | Contains | Tasks trained | Purpose |
|---|---|---|---|
| 1 | 5 LR values, seed 0, first 3 tasks of `order_A` | 15 | pick the learning rate |
| 2 | `order_A` seeds 0-4, each ending with `U0` | 45 + 5 unlearns | Sections 4.1 and 4.3 |
| 3 | `order_B` and `order_C` seed 0, each ending with `U`(position-1 task) | 18 + 2 unlearns | Sections 4.2 and 4.3 |

Each job is one process, one dataset build, runs executed in sequence inside
it, checkpoints and indicator records written as it goes. If a job dies, it
resumes from the last boundary checkpoint rather than restarting.

### Unlearn on every run, not one

The unlearn is 100 update steps against a nine-task training run. It is close
to free.

The old study had one unlearned model, which is why Section 4.3 rests on a
single pair of checkpoints and says so. Appending an unlearn to all seven
nine-task runs costs 700 steps in total and turns that into seven pairs, across
three orderings and five seeds.

Unlearn the position-1 task of each ordering: task 0 for `order_A`, task 4 for
`order_B`, task 8 for `order_C`. That matches the old study, which unlearned
its own position-1 task.

The rescaling control stays four forward passes per pair, with nothing trained.

### Gate

Runs in job 1 pick the learning rate. Job 2 confirms it. If the five seeds do
not finish within 3 percentage points of each other on mean accuracy over the
nine tasks, **stop**. Do not start job 3 on a setting that failed its own
target. Tune further and repeat job 2.

## Where each job runs

Split by run length, not by arm. Long uninterrupted training goes to RunPod,
where the machine does not disconnect. Short work and anything that only reads
goes to Colab.

| Work | Where | Why |
|---|---|---|
| Job 1, LR sweep | Colab | 3 tasks per run, short enough to finish in one session |
| Job 2, 5 seeds of 9 tasks | RunPod | the long pole, 45 task-trainings |
| Job 3, order_B and order_C | RunPod | 18 task-trainings, same reasoning |
| The seven unlearns | wherever their run finished | 100 steps each, continue in place |
| Rescaling controls | Colab | four forward passes, no training |
| Any re-read of a saved boundary | Colab | no training at all |
| Analysis, tables, figures | Colab or local | reads result files only |

The gate between job 1 and job 2 is also the handover point between the two
machines. Do not start paying for a RunPod instance until the sweep has picked
a learning rate on Colab.

### Moving checkpoints between the two

`uncle/storage.py` already has a working R2 transfer layer, with verification
on read-back. Use it rather than inventing anything: both platforms push and
pull the same bucket, and `uncle/plasticity_indicators.py` already writes
boundary records to R2 under a key built by `indicator_key(study, order,
seed)`.

This is what makes the split safe. A job finishing on RunPod leaves its
per-boundary checkpoints and indicator records in R2, and every later re-read,
control and figure runs on Colab against the same files. Nothing has to be
recomputed just because it was measured on a different machine.

Two rules that keep it that way:

- Push per-boundary checkpoints as they are written, not at the end of a job.
  A RunPod instance that dies with nine tasks of unsaved work costs real money
  to repeat.
- Never let a result exist only on a running instance. If it is not in R2, it
  does not exist.

### RunPod sizing

The accelerator is a choice, not a given. Check what is actually available and
price it before launching, rather than assuming a particular card. Measure one
task's wall clock in the Colab sweep first, multiply by 45 for job 2, and use
that to pick between a cheaper card for longer and a faster card for less time.

Jobs 2 and 3 are sequential inside themselves but independent of each other, so
they can run on two instances at once if the budget allows, or one after the
other if not.

**Gate between phase 2a and 2b.** Runs 1 to 5 pick the learning rate. Runs 6 to
10 confirm it. If the five seeds do not finish within 3 percentage points of
each other on mean accuracy over the nine tasks, **stop**. Do not start runs 11
to 13 on a setting that failed its own target. Tune further and repeat runs 6
to 10.

Candidate learning rates for the sweep, centred on the current default of
1e-3: `3e-4, 5e-4, 1e-3, 2e-3, 4e-3`. Record every value including the
rejected ones.

**Unlearn settings for run 13.** Unlearn task 0, the task learned first in
`order_A`, matching the old study which unlearned its own position-1 task. 100
steps, the UnCLe prescription and the current `burn_in` default. Confirm the
unlearn worked before reading anything from it: task 0 at 5 percent, which is
chance for twenty classes, and the other eight tasks within a few points of
where they were.

## Cost

Per epoch, per task: 20 classes at 500 training images each is 10,000 images,
twice the old task size.

- Sweep: 5 runs of 3 tasks is 15 task-trainings.
- Main: 7 runs of 9 tasks is 63 task-trainings.
- Old study: 5 runs of 8 tasks is 40 task-trainings, at half the images each.

So the new study is about 3.1 times the old one on the nine-task runs alone,
and about 3.9 times with the sweep included. Measure one task's wall clock in
the first sweep run and multiply before committing to the rest.

The seven unlearns add 700 update steps, which is under one task's worth of
training. The rescaling controls add nothing. Every indicator reading rides on
a forward pass the run is already doing.

So the total is 78 task-trainings and seven short unlearns, in three jobs. The
only way to spend meaningfully more than that is to have to run something
twice, which is what the per-boundary checkpoints and the measure-everything
rule exist to prevent.

## Reporting back

Record results in `docs/wiki/Experiment-trajectory.md` and the dated page for
the day, as `hypernetwork/AGENTS.md` requires. Include failed and interrupted
runs, intended against actual settings, and the starting checkpoint. Do not
invent missing run dates.

If a notebook changes, commit it, push it, and give the user the Colab link in
the same reply. An unpushed notebook means a link that opens the old version.

## Constraints

- Use the user's configured Git identity. No assistant, tool or coauthor
  attribution in commits.
- Preserve unrelated working changes.
- `ops-docs/` is gitignored, so anything written there is not version
  controlled. Experiment records belong in `docs/wiki/`, and this plan and its
  companions are committed under `docs/plasticity/`.
- Keep this file current. If the plan changes during the work, edit it here
  rather than leaving the change only in a chat.
- The methodology file says the tuning runs are reported in full, including
  rejected values. Keep the rejected values.

## One correction to the framing

This is close to "the same experiments with extra seeds and an extra task", but
two parts are genuinely new rather than extended:

- The learning-rate sweep did not exist before. It is a prerequisite, not an
  extra, because every later comparison is measured against the tuned setting.
- The old orderings came from the UnCLe paper's Table 4 and addressed 20 task
  IDs. They cannot be carried over. The three above are new and chosen for this
  study.

Everything else is the old design at a larger size.

## Open question for the user

Confirm the compute budget before launching runs 6 to 10. At roughly 3.9 times
the old study, this is the decision worth making before the GPU time is spent,
not after.
