# UnCLe: reproduction and diagnostic implementation plan

This plan implements [PROPOSAL.md](../ops-docs/PROPOSAL.md). The proposal takes
precedence on experimental details, scope, and interpretation. This document
tracks the code, dependencies, checks, and runs needed to carry it out.
The proposal is currently in the ignored `ops-docs/` directory; the protocol
below records its requirements for readers of the tracked repository.

Last updated: 2026-10-05.

[Experiment trajectory](https://github.com/sumitasthana/PATH/wiki/Experiment-trajectory) records the reported GPU experiments
through E15, the 2026-09-29 30-step run, and the checkpoint-based diagnostics. No reported
trajectory has passed both the short-run forgetting and retention criteria.
These are existing observations, not measurements made during this plan update.

## Current priority after reassessment: test retention during learning

This priority supersedes the replay-first schedule below. The research asks
whether task information remains after unlearning. Its prerequisite is a model
that learns tasks and preserves useful accuracy on the tasks it keeps. The
full-sequence failure remains unresolved. That sequence contains both learn and
forget requests, so older-task collapse alone does not isolate learning as the
cause.

The supplied replay pair used different GPU models and first differed at update
1. It does not establish that hardware caused the historical spread. Exact
hash agreement is a diagnostic tool, not a requirement for proceeding with all
research. A tiny first-loss difference also does not, by itself, explain a large
final accuracy gap.

### Bound the remaining replay work

Verify the files from the completed A100 run A on CPU. If the files are complete,
allow at most one additional run B on the matching A100, using A's unchanged
settings. If the files are incomplete or the required hardware is unavailable,
retain the records and leave this pair unfinished. Do not restart the pair,
start a stricter deterministic pair, or continue kernel-level tracing now.
Record any remaining variance as an unresolved limitation and proceed to the
learning-only diagnostic. No same-GPU pair result has yet been supplied.

### Next research experiment

**Question.** Does learning new tasks damage older-task accuracy when no forget
requests occur?

1. Restore the existing E08 checkpoint before U3, after L3 and L0. Verify its
   saved history, evaluate tasks 3 and 0, and record their actual starting scores.
   This historical all-images checkpoint is diagnostic-only.
2. In one GPU session, learn tasks 9, 5, and 17 in that order, with no forget
   requests. Keep the checkpoint's learning rate, beta, epochs, partition, and
   other training settings unchanged. Save a distinct run configuration.
3. After each new task, evaluate every task learned so far. Save those accuracies,
   runtime provenance, and a checkpoint at each completed request boundary.
4. Produce one table showing each task's accuracy before and after the subsequent
   learn requests. Report changes in percentage points. Locate the first request
   where an older task drops more than five percentage points from its score
   before that request, if present. Use this as an exploratory warning, not a
   statistical decision rule. Do not claim this single-seed result generalizes
   across tasks, sequences, or seeds.

If older-task accuracy collapses, examine the learning preservation penalty and
its protected-task set at that request before choosing one intervention. If
accuracy stays useful over this short sequence, extend the learning-only test
before attributing the full-sequence collapse to unlearning. Do not resume gamma
sweeps or component attribution yet. Once the diagnostic identifies a candidate
setting, test it across multiple seeds to assess how consistently it works.

## Earlier target for 2026-10-05: explain cross-session L9 replay variance

**Target.** Locate the earliest recorded divergence when learning task 9 (L9)
from the same saved post-U3 checkpoint in two fresh Colab GPU sessions, and
use the evidence to narrow the cause.

[Stage 6 of the trajectory](https://github.com/sumitasthana/PATH/wiki/Experiment-trajectory#stage-6-paired-task-9-comparison-and-replay-variance)
reports an 11-percentage-point spread in final task 0 accuracy across sessions,
versus 0.4 points within one session. The paired effect remains inconclusive.
Earlier resumes overwrote some environment records. These were historical
observations when the initial replay target was written.

Workflow: [notebook 11](../notebooks/11_L9_session_replay.ipynb), with session A,
session B in a fresh GPU runtime, and comparison in a fresh CPU runtime.

### Follow-up: use the same GPU model in both runs

The user supplied a comparison for `20261005_L9_replay_01`. It reports task 0
accuracy of 29.2% in A and 36.4% in B, and task 9 accuracy of 43.8% and 42.2%.
The first recorded difference was at update 1 in loss, model parameters, and
task buffers. A used an RTX PRO 6000 Blackwell; B used an A100-SXM4-40GB.
These observations come from the supplied report, not a local GPU measurement.
Different GPU models are a hypothesis to test, not a confirmed cause of this
pair's final accuracy gap or the historical spread.

Next, run a new pair named `20261005_L9_same_gpu_01`, with an A100-SXM4-40GB in
both fresh sessions. Notebook 11 checks the GPU model before downloading data
or starting training, and checks the recorded training-software versions before
B trains. Keep the same checkpoint, seed, training settings, and images. Save the
new pair separately. Compare its first recorded divergence and final accuracy
gaps with the earlier pair. A closer match would support further hardware
investigation; a single matching pair does not establish a cause.

### Steps

1. Fix the checkpoint, source commit, resolved settings, task partition, data
   contents, preprocessing, and RNG restoration policy. Prepare identical
   trace logging for both runs. Hold gamma tuning today.
2. Run L9 in a fresh Colab GPU session. Save the starting checkpoint SHA-256,
   runtime versions, GPU, deterministic settings, RNG states at the training
   boundary, task-9 embedding hash, and starting task 0 and task 9 accuracies.
   Record batch order with sample identifiers and input-batch hashes,
   first-update loss, per-update losses and parameter hashes, and final task 0
   and task 9 accuracies. Logging must not consume training RNG or change
   model state.
3. Save artifacts to a unique Drive directory, verify persistence, then release
   the GPU. Preserve the original environment record when resuming a run.
4. Repeat in a second fresh GPU session with the same code, checkpoint, settings,
   and data. Save separately, verify persistence, and release the GPU.
5. Compare provenance and starting state, then batches, losses, and parameter
   hashes in update order. Report the earliest recorded mismatch with run IDs
   and update index. Separate observations from hypotheses about the cause.

### Completion criteria and next experiment

Deliver a saved comparison with final accuracies and either the first recorded
mismatch or agreement at all recorded points. A mismatch should identify the
next targeted check needed to explain it. If both runs agree, report that the
historical spread was not reproduced; two matching runs do not resolve its
cause. Finding the first divergence may narrow the cause without proving it.

After the replay difference is explained, run a learning-only sequence with no
unlearn requests, recording older-task accuracy after every learn request.
This will test the retention failure described in Stage 5.

## 1. Study question and scope

Does UnCLe retain information that helps it relearn a forgotten task? If so,
which model components contribute to that recovery?

| Component | Current behavior during forgetting | Planned test |
| --- | --- | --- |
| Task embedding | Unchanged | Adapt alone; replace with a random embedding |
| Shared layers | Updated | Adapt alone and jointly |
| Output heads | Updated | Adapt alone and jointly |
| Chunk embeddings | Frozen after the first task | Test jointly with shared layers and heads; no independent attribution |
| Per-task BatchNorm buffers | Unchanged in this implementation | Compare keeping and resetting statistics under an explicit adaptation policy |

The study evaluates unlearning and records implementation choices. Developing
an algorithm to remove information from identified components is outside scope.
Faster recovery is evidence to examine with the controls below. A negative
result is informative only when the pre-deletion positive control recovers.
Neither outcome alone establishes complete information deletion or storage in
a particular component.

## 2. Protocol from the proposal

### Dataset and model settings

- Tiny ImageNet: 20 tasks of 10 classes, ResNet50 target network, no pretrained
  weights, three request sequences, seeds 0, 1 and 2, class partition seed 42.
- Reserve 100 of the 500 training images per class before study training. Use
  the other 400 for initial learning and the reserved images for adaptation.
  Evaluate on the validation split. Apply the reservation to every study task.
- Use identical adaptation examples and evaluation examples for paired models.
  Save the split indices rather than reconstructing them from an unstored seed.
- Permuted MNIST with ResNet18 is the likely pilot and debugging setting.
  Results from that setting do not establish results on Tiny ImageNet.
- **[to discuss]** The 20-by-10 Tiny ImageNet partition uses all 200 classes.
  Resolve where the never-seen control task Y comes from before study training.
  Keep the existing partition for historical comparisons; record any diagnostic
  partition or sequence changes separately. Do not silently drop a task.
- **[to discuss]** Define the adaptation reservation for a Permuted MNIST pilot;
  the Tiny ImageNet counts do not specify that dataset's split.

The 400-image study training split differs from training on all 500 images per
class. State that difference when comparing against the published accuracy.

### Four model conditions

For each probed task X, match seed, architecture, data splits, and the order of
all common requests. Record the exact sequence for every condition.

| Condition | Training history | Purpose |
| --- | --- | --- |
| Unlearned | Learn X, then forget X in the full sequence | Model under test |
| Never-learned-X | Remove X's learn and forget requests | Paired learning reference |
| Pre-deletion | Stop the full run immediately before forgetting X | Positive control for probe sensitivity |
| Never-forgot-X | Keep X's learn request; remove only its forget request | End-of-sequence comparison |

The pre-deletion condition can reuse a saved checkpoint. Never-forgot-X is
specific to X; a sequence with every forget request removed is not equivalent.
Save aligned reference checkpoints for both immediate and endpoint probes.

Use separate recorded random streams for initialisation, task-code creation,
data order, forget noise, and adaptation where needed. Removing a request must
not silently change random draws for unrelated operations. The changed training
history still changes the model, protected-task sets, and potentially the
burn-in schedule. Record those differences rather than claiming identical
training. In particular, if X was the first task, removing it changes which
task trains the chunk embeddings; account for that when selecting probe tasks.

### Relearning and component interventions

Start every probe from a fresh checkpoint copy and a fresh optimiser. Disable
the preservation penalty during adaptation. Evaluate at fixed intervals,
including step zero, on adaptation images, held-out X images, and retained tasks.
The never-learned-X reference needs a newly initialised task embedding and
buffers. The unlearned model reuses its saved state unless the intervention
explicitly changes it. Apply the same initialisation policy to Y in both models.

| Intervention | Parameters allowed to adapt or state changed |
| --- | --- |
| Full adaptation | Task embedding, shared layers, and heads |
| Embedding only | X's task embedding |
| Shared layers only | Shared layers |
| Heads only | Output heads |
| Random embedding | Replace X's embedding; record whether it is then frozen or adapted |
| BatchNorm keep versus reset | Keep saved statistics or reset to the clean template; match other settings |
| Joint chunk test | Change or adapt chunk embeddings together with shared layers and heads |
| Combined interventions | Apply selected resets or swaps together and compare with their separate effects |

Every restricted arm needs its own never-learned-X reference with the same
training restriction. Chunk embeddings remain frozen in the ordinary arms;
the joint chunk test is the explicit exception. It does not isolate their
independent contribution.

**[to discuss]** Specify whether BatchNorm statistics update during adaptation,
separately from whether they are initially kept or reset. Define the random
embedding arm's trainable parameters, swap donors, and combined interventions
before the main runs. Recalibrating buffers without gradient updates can be a
supplementary diagnostic, but does not replace the keep-versus-reset comparison.

Earlier planning choices remain candidates where the proposal is silent:

| Setting | Candidate to confirm before the main runs |
| --- | --- |
| Sequence-1 probe tasks | 3, 17, 9, 0, 12; recheck against the final Y design and sequence |
| Main budgets | 1, 5, 10, 25, 50, 100 images per class, as nested subsets |
| Adaptation optimiser | Adam; set and record an adaptation learning rate, separately from any forgetting-only rate |
| Batch size | `min(64, adaptation pool size)` |
| Update budget | 200 steps per budget; measurements at 0, 20, 40, ..., 200 |
| Probe timing | Immediately after forgetting and at the sequence endpoint; pilot uses the immediate probe |

A fixed update count matches the number of optimiser steps across budgets.
It does not guarantee identical runtime when batch sizes or trainable
components differ. Measure cost separately. These candidate values are not
requirements stated by the proposal.

### Metrics

Compute paired metrics at matching adaptation budgets and measurement steps.
Keep raw values for each task, seed, sequence, condition, and intervention.

| Metric | Definition | Purpose |
| --- | --- | --- |
| Recovery advantage `A(b)` | Unlearned accuracy minus never-learned-X accuracy after adaptation on `b` images per class | Primary recovery comparison |
| Likelihood gain `G(b)` | Unlearned minus reference mean correct-class log-probability on held-out images | Detect differences when accuracy is near chance |
| Specificity ratio `S(b)` | Recovery advantage on X divided by the corresponding advantage on never-seen Y | Check generic transfer |
| Component share `p(c)` | Drop in recovery advantage after resetting or swapping component `c`, divided by the full advantage | Measure dependence on the intervention |
| Interaction residual `D` | Effect of a combined intervention minus the sum of separate effects | Check whether separate component effects explain the combined result |
| Maintenance cost `M(b)` | Change in mean retained-task accuracy during adaptation | Report recovery's effect on retained tasks |

Also report retained accuracy, forgotten-task accuracy, spill, and relapse.
For `M(b)`, record the retained-task set and use each probe's step-zero value
as the baseline. State the sign convention. For `p(c)` and `D`, record the exact
reset or swap and the paired reference treatment. Restricted-adaptation speed
alone is not the reset-or-swap measurement that defines `p(c)`.

Report numerator and denominator alongside `S(b)` and `p(c)`. Mark ratios
undefined when their denominator is zero. **[to discuss]** Set a policy for
near-zero denominators before analysis; do not silently clip them or interpret
unstable ratios as strong effects.

### Controls and interpretation

| Concern | Required control or measurement |
| --- | --- |
| Generic transfer | Apply the same probe to Y, which was absent from every training sequence |
| Memorisation of adaptation images | Report adaptation accuracy alongside held-out accuracy |
| Damage to retained tasks | Measure retained accuracy at every interval and compute `M(b)` |
| An insensitive probe | Include the pre-deletion positive control in every experiment; a failed control makes a null result inconclusive |
| Different optimisation difficulty | Match each restricted arm to a reference under the same restriction |
| Recovery attributed to the embedding | Replace the stored embedding with a freshly sampled vector |
| Indirect preservation of X | Repeat one forget request without the preservation penalty and report retained-task damage |
| Component interactions | Compare combined interventions with the sum of individual effects |

Removing protection during forgetting is a separate experiment from disabling
it during relearning. Implement a separate preservation switch or coefficient;
setting the current `gamma` to zero would remove the noise term instead.

Compare within seed and sequence. Five tasks across three seeds give fifteen
paired comparisons, but tasks within a run share a model and are not independent.
Report per-pair results and uncertainty that respects that dependence.
**[to discuss]** Fix the decision rule, uncertainty method, and smallest effect
of interest before the main runs. The earlier mean-gap-versus-seed-spread rule
is not a settled statistical criterion.

Membership inference is needed if the final paper makes the proposal's
conditional claim that recovery occurs while accuracy, relapse, and membership
inference indicate success. Specify and run that evaluation before making the
claim. Without it, report comparisons only against the metrics actually tested.

## 3. Current implementation status

Section 6 states, module by module, what the repository holds and what is
missing. This section states what is true of the research.

The method is implemented and the reproduction has not succeeded. The full
Tiny ImageNet run reported 10% retained accuracy against the paper's 55.24%,
with mean spill 30.72 against 0.722. The short diagnostics through E15 have not
met both screening criteria at any setting tried. The experiment log holds the
measured settings and their limits.

All 52 checks and the 19 guide code blocks passed on 2026-09-27, on CPU, with
the CUDA-specific check skipped. Passing checks establish software behavior on
the tested machine. They do not establish a reproduction, and no result in this
document was measured during this update.

Not implemented at all: the relearning entry point, the four-condition
reference builder, the reserved adaptation split, the Y control, component
interventions, Fisher screening, the six study metrics, and membership
inference. Out of scope unless the study changes: CIFAR-100, 5-Tasks, the
alternative noising strategies, and the seven comparison methods.

## 4. Implementation decisions to retain

- Cite the seven comparison methods without reimplementing them, as the proposal
  currently plans. **[to discuss]** Confirm the final comparison scope.
- Keep the seed-42 class partition for existing runs. Resolve Y explicitly for
  the diagnostic study and record any departure from the reproduction setting.
- Use Tiny ImageNet validation labels for evaluation; its public test split
  has no labels. Keep `uncle/data.py` routed through the existing task loaders.
- Retain the current `1/fan_in` classifier output scale as an implementation
  choice. The paper specifies Hyperfan, which is not implemented here. Record
  this difference; do not assume its impact is settled.
- Preserve the planned 100-image adaptation reservation before study training.
  That is 20% of the 500 training images per class, leaving 400 for learning.
  Older checkpoints trained on all images cannot serve as split-compliant study
  checkpoints; they can still support reproduction debugging.
- Add a dedicated adaptation entry point. Keep ordinary continual-learning
  checks against duplicate learn requests; the probe should not need to bypass
  them or call `add_task` on an existing embedding.
- Preserve explicit generator restrictions, no protection during adaptation,
  and equal update counts within each paired comparison. Use a fresh optimiser.
- Record unspecified choices such as epochs, batch size, parameter scaling,
  head allocation, and buffer policies. Separate code behavior from claims
  about what the original method intended.

## 5. Operational constraints

- Give every study run a distinct identifier covering its configuration,
  condition, split, checkpoint, and intervention. Existing sequence/backbone/seed
  filenames are insufficient for larger sweeps.
- Keep datasets on local storage in GPU runtimes and persist results separately.
  Pass the selected classes to the Tiny ImageNet reader to avoid indexing
  unused classes.
- Clear `UnCLe._reference` whenever the snapshot or protected task codes change.
  Reusing stale reference weights changes the preservation objective.
- The normal request runner reuses the previous request's accuracy as the next
  request's starting value. Intervention branches must evaluate their actual
  starting state after any reset or swap.
- `run_experiment` returns a dictionary. Keep notebook examples consistent with
  the callable API and run the guide checker when changing executable examples.
- Keep evaluation observational: it must not update stored buffers, parameters,
  or the random stream used by training.

## 6. Modules: what exists and what must be built

Every module below states what the repository holds today, then what is
missing. Status was checked against the code on 2026-09-27, not carried over
from an earlier plan. Diagnostic updates below were checked on 2026-10-01.
Module IDs name a responsibility, not a file; several
already live in one file.

| | Module | Status |
| --- | --- | --- |
| M01 | Study configuration | Run configuration built, study configuration missing |
| M02 | Dataset splits and control task | Task splits built, adaptation reserve and Y missing |
| M03 | Model component access | Semantic state inspection and before/after audit built for UnCLe; intervention controls missing |
| M04 | Learning and forgetting | Built, one switch missing, reproduction unresolved |
| M05 | Reproduction diagnostics | Built |
| M06 | Matched reference construction | Not built |
| M07 | Checkpoint collection and restoration | Built for resuming, stage retention missing |
| M08 | Relearning probe | Not built, and currently refused by validation |
| M09 | Component interventions | Not built |
| M10 | Experimental controls | Evaluation hygiene built, seven of eight controls missing |
| M11 | Fisher screening | Model-neutral diagonal calculation built; E08 checkpoint runner and study examples missing |
| M12 | Evaluation and metrics | Four reproduction metrics built; recovery metrics need the probe runner |
| M13 | Statistical analysis | Not built |
| M14 | Experiment orchestration | Reproduction matrix built, study matrix missing |
| M15 | Logging and report generation | Run logging and first diagnostic assessment built; recovery reports missing |
| M16 | Experiment verification | 52 checks built, study checks missing |
| Conditional | Membership inference | Not built |

### M01. Study configuration

**Built.** `uncle/config.py` holds every run knob as a frozen dataclass and
rejects invalid request lists, unknown datasets and unknown backbones in
`Config.__post_init__`. `dataset_defaults` supplies the paper's per-dataset
values, and `random_stream` gives each purpose and task its own generator so a
draw does not depend on how many requests came before it.

**Missing.** Nothing describes a study run: conditions, probed task X, control
task Y, adaptation budgets, checkpoint stages, interventions, evaluation
intervals and analysis choices. Add a separate probe configuration rather than
widening `Config`, which is the reproduction's contract with the paper.

**Accept** when an incomplete study configuration fails validation and a saved
one reproduces the run matrix. Depends on the decisions in section 9.

### M02. Dataset splits and control task

**Built.** `uncle/tasks.py` loads the seed-42 partition from
`uncle/task_partition.json` and verifies it is complete and disjoint on every
run. `uncle/streams.py` builds Tiny ImageNet tasks from it, sharing one base
dataset per split, with `include` and `max_images`. `uncle/data.py` routes
Permuted MNIST and Tiny ImageNet through one entry point.

**Missing.** The 100-image-per-class adaptation reserve does not exist: every
task currently trains on all 500. There are no persisted split indices, no
nested budget subsets, and no never-seen Y task. Write the indices to a file
beside the partition, in the same style. Do not regenerate them from a seed.

**Accept** when split counts and disjointness are checked, Y is absent from
every training sequence, and paired runs read identical adaptation examples.

### M03. Model component access

**Built.** `HyperNetwork` exposes `trunk`, `heads`, `chunk_codes`, `task_codes`
and `generator_parameters()`, and `UnCLe` holds `task_buffers` and
`buffer_template`. Every component the study addresses is reachable today.

**Partly built.** `uncle/research_diagnostic.py` maps UnCLe's task embedding,
chunk embeddings, shared layers, output heads, and task buffers to semantic
roles. The optional checkpoint diagnostic compares those roles before and after
forgetting. Live inspection reads weight and gradient summaries. The study's
trainable-role masks and state replacement still need to be built with the
intervention runner. `forget` uses its existing two fixed parameter choices.

**Accept** when groups match the live model, parameters and buffers are
distinguished, and a check proves an intervention moved nothing else.

### M04. Learning and forgetting

**Built.** `UnCLe.learn`, `UnCLe.forget` and `UnCLe.preserve` implement the
paper's equations, including per-task BatchNorm statistics, chunk codes frozen
after the first task, averaged noise draws and annealed burn-in.
`uncle/experiment.py` runs a request sequence, protecting every seen task
except the request's own.

**Missing.** A way to remove the preservation term during a forget request, for
the indirect-preservation control. Setting `gamma` to zero removes the noise
term instead, so this needs its own switch. The reproduction failure itself is
unresolved and blocks everything downstream.

**Accept** when the section 7 thresholds are measured, forgetting still uses no
data from the forgotten task, and departures from the paper are recorded.

### M05. Reproduction diagnostics

**Built.** `uncle/diagnostics.py` restores a checkpoint, runs the real
forgetting loop for a chosen number of steps, evaluates every seen task at step
zero and after each update, and writes a uniquely named report without touching
the source checkpoint. `UnCLe.forget(measure=True)` adds each loss term's
gradient norm and cosine per parameter group, actual Adam update norm, and the
forgotten task's raw output size. An optional before/after component audit uses
semantic roles. The model-neutral screen reads complete JSON reports and the
tracked historical traces without discarding failed steps.
That is what separates the two terms: they are computed from different
quantities, raw output against scaled weights, so their losses are not
comparable and their gradients are.

**Missing.** No GPU result yet uses the new cosine and component audit. The
recovery study requires the separate modules below.

**Accept** when reports name their starting checkpoint and settings, preserve
the source state, and separate observations from candidate explanations.

### M06. Matched reference construction

**Not built.** `uncle/experiment.py` runs one fixed request list, and
`run_sequences` in `uncle/experiments.py` loops over sequences and seeds.
Neither knows about conditions.

**To build.** A builder for all four conditions in section 2, producing aligned
checkpoint stages and a manifest of what differs between them: request lists,
protected-task sets, burn-in schedules, and whether removing X changed which
task trained the chunk codes.

**Accept** when request-list differences are exactly those specified, the
initial shared state matches, and the manifest records every other difference.
Depends on M01, M02, M04, M07.

### M07. Checkpoint collection and restoration

**Built.** `uncle/checkpoint.py` saves the hypernetwork, per-task buffers, the
request history, the seen and forgotten lists, previous accuracies, costs and
the CPU and CUDA random state. It writes to a temporary file and moves it into
place, refuses a checkpoint whose configuration differs anywhere, and restores
onto CPU before moving to the device. `load(config=None)` exists for diagnostic
callers, so a probe can open a checkpoint without the resume comparison
fighting it. A resumed run reproduces the records an uninterrupted one would
have had, which is tested.

**Missing.** One file per run, replaced after every request. The study needs
three surviving states per probed task: immediately before its forget,
immediately after, and at the sequence end. Optimiser and sampler state are not
saved, which is fine at a request boundary and not for resuming inside an
adaptation run.

**Accept** when restoration reproduces step-zero outputs and a probe cannot
mutate its source or another branch.

### M08. Relearning probe

**Not built, and currently refused.** `Config.__post_init__` rejects a task
learned twice, and `HyperNetwork.add_task` raises on a task that already has a
code. Both are correct for continual learning and both block the probe.

**To build.** A separate adaptation entry point beside `UnCLe.learn`: works on
an existing task state, creates state for reference models and Y, takes a fresh
optimiser, runs a fixed number of updates rather than epochs, and leaves the
preservation penalty off by default. Do not relax the existing validation to
make room for it.

**Accept** when paired runs use matching examples and update budgets, step zero
is recorded, a positive example actually learns, and frozen parameters do not
move. Depends on M02, M03, M07.

### M09. Component interventions

**Not built.** The component roles are named, but the arms in section 2 have
no execution path, trainable-role masks, or state replacement.

**To build.** Implement restricted adaptation per component, embedding
replacement, the BatchNorm keep-versus-reset comparison, the joint chunk test,
and combined interventions with matched probes.

**Accept** when every arm's trainable set and buffer policy are verified, each
arm has a reference under the same restriction, and swap donors and
incompatibility checks are recorded. Depends on M03, M07, M08.

### M10. Experimental controls

**Partly built.** `UnCLe.accuracy` evaluates without disturbing stored
statistics, cloning buffers first, and the forget callback isolates its random
draws and restores model modes. That is the evaluation hygiene every control
rests on.

**Missing.** Seven of the eight controls in section 2: Y, adaptation against
held-out accuracy, the pre-deletion positive control as a routine part of every
experiment, matched restrictions per arm, embedding replacement, the
no-preservation forget request, and the interaction comparison.

**Accept** when every reported result links to its required controls and a
failed control is visible rather than silent. Depends on M02, M04, M06, M08 and
M12, plus M09 for the component controls.

### M11. Fisher screening

**Partly built.** `diagonal_fisher` computes empirical and model-predicted
diagonal values from individual examples and averages per component parameter.
A small exact calculation is tested. It excludes buffers and leaves the model's
recorded gradients unchanged. No E08 checkpoint runner or study example set
exists yet. An older sanity script outside the repository is historical only.

**To build.** Run both Fisher forms at the pre-deletion checkpoint on a saved,
identified example set and record timing and memory use. This orders the
component tests and claims nothing about storage. Scores taken at the
pre-deletion checkpoint rank components of the
model that still holds the task, so using them to order tests on the unlearned
model is an assumption to state rather than assume.

**Accept** when a small exact calculation agrees with the implementation, input
examples are recorded, model state is unchanged, and BatchNorm buffers are
excluded from the ranking. Depends on M03, M07.

### M12. Evaluation and metrics

**Built.** `uncle/metrics.py` computes the paper's four numbers: retain
accuracy, forget accuracy, spill per forget request, and relapse per forgotten
task, summarised from the run history.

**Missing.** Recovery advantage, likelihood gain, maintenance cost,
specificity ratio, component share, and interaction residual need records from
the relearning probe. Pairing rules, held-out log-probabilities, adaptation
accuracy, and retained-task measurements belong with that runner. Keep the
four existing reproduction metrics unchanged; the study adds to them.

**Accept** when hand-calculated cases agree, paired records cannot be
mismatched, and zero-denominator ratios are reported as undefined. Depends on
M01 and M08, plus M09 for the attribution metrics.

### M13. Statistical analysis

**Not built.** Nothing aggregates across tasks, seeds or sequences beyond
`compare` in `uncle/experiments.py`, which prints a table.

**To build.** Paired analysis by task, seed and sequence, with uncertainty that
respects the fact that tasks inside one run share a model. Record the decision
rule and the smallest effect of interest before the main runs.

**Accept** when per-pair results stay available, shared-model dependence is
retained, and conclusions follow the rule recorded beforehand. Depends on M12
and the section 9 decisions.

### M14. Experiment orchestration

**Built for the reproduction.** `uncle/experiments.py` builds a config, runs a
sequence, writes `history_`, `summary_`, `costs_` and `environment_` files
named for sequence, backbone and seed, loops sequences and seeds through
`run_sequences`, and resumes from a checkpoint.

**Missing.** A study matrix: conditions, budgets, interventions and probe
timings, with the dependencies between them and a continuation rule after the
pilot. The existing file names cannot separate those runs, which is why
section 5 requires a run identifier.

**Accept** when a small run exercises the whole pilot matrix and an
interruption neither mixes configurations nor silently repeats completed
probes. Depends on M01, M06, M07, M08, M10 and M12, plus M09 and M11 for full
attribution.

### M15. Logging and report generation

**Built.** `uncle/telemetry.py` times each request, separates learning from
forgetting, records seconds per step, peak GPU memory and protected count, and
writes an environment record with the GPU, torch version and git commit, so a
number found later can be traced to the code that produced it.

**Partly built.** `scripts/research_diagnostic.py` evaluates every archived
forgetting step and can include original runtime JSON. It keeps the full
trajectory and source type in an optional JSON assessment. Recovery curves,
likelihood comparisons, X
against Y, component effects, interactions and retained-accuracy cost, plus
split and checkpoint identities in the record.

**Accept** when every plotted value traces to a raw run record and figures can
be rebuilt without retraining. Depends on M12 to M14; log support starts at M01.

### M16. Experiment verification

**Built.** Checks across `tests/test_uncle.py`, `tests/test_tasks.py`,
`tests/test_experiments.py`, `tests/test_diagnostics.py`, and
`tests/test_artifacts.py`. Notebook code cells are checked for Python syntax.
These checks cover
the paper's stated values, the partition, resume equivalence, evaluation
hygiene, and that the preserve term has no gradient at the first forget step.

**Missing.** Checks for everything the study adds: split separation, matched
histories, stage restoration, probe isolation, freeze masks, BatchNorm policy,
Fisher, the new metrics, and run resumption. The freeze-mask check matters
most: a leak there does not crash, it produces a clean-looking component
ranking that is wrong.

**Accept** when the checks include positive examples and deliberately invalid
inputs, and a small end-to-end run exercises the study path before any
expensive experiment.

### Conditional module: membership inference

**Not built.** Required only if the final paper keeps the proposal's
conditional claim.

**To build.** A separate evaluator with stated attack access, member and
nonmember sets, held-out attack evaluation and success criteria, recording any
difference from the original paper's protocol. Do not infer success from
chance-level accuracy on a forgotten task. Depends on M02, M06, M07, M15.

## 7. Execution order, schedule, and budget

### Weeks 1 and 2: reproduction and study foundations

The objective and scaling question **T0.2** remains open. The 27-update
candidate was checked in a full sequence before more objective diagnostics.
Pause further gamma and step sweeps. The existing
[diagnostic notebook](../notebooks/04_gradient_diagnostics.ipynb) can address
T0.2 later without changing the current checkpoint.

The existing short screen is a debugging check: before forgetting, both tasks
must have at least 25% accuracy; a candidate step must bring X to at most 12%
while changing retained-task accuracy by less than five percentage points.
Passing it still requires full-sequence validation.

For an exploratory check, the user accepted task X accuracy at or below 13%
with the same five-point retained-task limit. The reported step 27 (task 3 at
12.6%, task 0 at 41.0%) is the candidate. This does not replace the 12% short
screen or the paper reproduction target. The full-sequence run continued
sequence 1 from the E08 pre-forgetting checkpoint with 27 updates per forget
request, forgetting LR `1e-5`, and gamma `5e-6`. It saved a post-`U3`
checkpoint. The reported full-sequence retained accuracy was 22.4% and mean
spill was 18.98, so it did not pass the recovery-entry check. The run had one
seed and used the old all-images training split, so it remains a reproduction
check, not a study checkpoint. [Notebook 05](../notebooks/05_full_sequence_candidate.ipynb)
is the historical record of that specific run; use the
[notebook guide](../notebooks/README.md) to choose a current workflow.

In parallel with that code work, implement M01-M03 and M07 and resolve the data
split. Historical unsplit checkpoints remain diagnostic-only.

The proposal distinguishes two thresholds for the Tiny ImageNet setting:

- Recovery work may begin once retained accuracy exceeds 40% and spill is below 3.
- The reproduction target is retained accuracy within 10 percentage points of
  55.24%, forgotten-task accuracy within one point of 10%, and spill below 2.

Record forgotten-task accuracy at the recovery-entry check too. Do not report
that meeting the entry threshold completes the reproduction. Run the proposed
three request sequences and three seeds for the main reproduction. If using
Permuted MNIST as the fallback or pilot, define suitable criteria for that
setting before interpreting its results; Tiny ImageNet thresholds are not its
published targets.

### Week 3: probe implementation and pilot

Complete the minimum working M06, M08, M10, M12, M14-M16 path. The pilot uses
one task, three seeds, three small budgets, the immediate-post-deletion probe,
the never-learned reference, the pre-deletion positive control, and Y. The
never-forgot-X condition is required for the later endpoint study.

**[to discuss]** Choose the pilot dataset, task, budgets, and runtime estimate.
Proceed to component tests only if the pre-deletion control shows a recovery
advantage and the measured recovery advantage has a consistent sign across
seeds. Otherwise stop component work and report what the pilot establishes.
This continuation rule does not require an assumed positive finding and does
not replace the final statistical decision rule.

### Week 4: reference models and attribution support

After the pilot decision, build the main four-condition references for selected
X tasks and seeds, preserving both immediate and endpoint stages. Complete
M09, M11, and M13. Use Fisher scores to order the parameter-component tests;
BatchNorm tests do not depend on those scores. Finalise all analysis and
intervention choices before the main recovery comparisons.

### Week 5: recovery comparisons and controls

Run the main recovery budgets, component interventions, joint chunk tests,
combined interventions, and all eight controls. Include pre-deletion controls
in every experiment and matched restrictions in every arm. Run the conditional
membership-inference evaluation if making that comparison.

### Week 6: analysis and writing

Complete M13 and M15 outputs. Report uncertainty, failed controls, retained-task
costs, implementation choices, and the limits of attribution. Do not require
BatchNorm or embeddings to rank first for the experiment to be valid. A null
result with a successful positive control is a reportable outcome under the
proposal; it is not proof of complete deletion.

The proposal budgets about **50 A100-hours**, to be revised after the pilot.
Measure actual reference, adaptation, Fisher, and evaluation costs before
committing the remaining matrix. Earlier per-phase estimates and an assumed
2x H100 speedup do not establish the cost of this revised design.

Broad beta/gamma grids, alternative noising strategies, a learning-only
saturation study, extra datasets, and a buffers-only recalibration arm are
optional follow-ups. They do not precede or replace the proposal's pilot.

## 8. Verification and deliverables

Run checks appropriate to each code change and record their results. Existing
entry points are:

```bash
python tests/test_uncle.py
python tests/test_tasks.py
python tests/test_experiments.py
python -m unittest discover -s tests -p test_diagnostics.py
python -m unittest tests.test_artifacts
```

Some checks require the Tiny ImageNet data. Record missing prerequisites or
skipped checks explicitly. All applicable checks must
pass before committing implementation changes. Update this list as the new
study modules receive tests; avoid fixed test counts that become stale.

Before a main GPU run, verify the resolved configuration, split identities,
condition sequences, checkpoint stages, freeze masks, BatchNorm policy, and
required controls. Verify on small cases that evaluation does not affect the
next training update and that restoring a checkpoint reproduces its outputs.

The study deliverables are the reproduction record, relearning entry point,
four-condition checkpoint manifest, raw probe and control measurements, six
metric calculations, component and interaction results, analysis protocol,
rebuildable figures and tables, and runtime report. A successful experiment
produces interpretable evidence; it does not require the preferred hypothesis
to be confirmed.

## 9. Open decisions before dependent work

- **Y and splits:** choose the Tiny ImageNet control-task design and record its
  effect on task count, class partition, and request sequences before study
  training. Define the Permuted MNIST reservation if used.
- **Pilot:** choose dataset, task, three budgets, training and adaptation
  settings, and runtime estimate. State applicable reproduction criteria for
  a pilot outside Tiny ImageNet.
- **Main matrix:** confirm selected X tasks and sequences, adaptation budgets,
  step count, evaluation intervals, and learning rate. The candidate table in
  section 2 is not a substitute for these decisions.
- **Interventions:** fix BatchNorm update behavior, random-embedding policy,
  swap donors, joint chunk treatment, combined interventions, and the forget
  request used for the no-preservation control.
- **Analysis:** fix the decision rule, smallest effect of interest, uncertainty
  procedure, and handling of near-zero ratio denominators before main runs.
- **Comparison scope:** confirm whether membership inference is evaluated and
  retain only claims supported by the completed measurements. Confirm the plan
  to cite rather than implement the seven comparison methods.
- **Novelty:** complete the literature review before claiming this setting or
  attribution protocol has not been tested before.

Resolve each item before its dependent run. Continue independent implementation
and reproduction debugging while those choices remain open.
