# Experiment trajectory: 5 and 6 October 2026

[Back to the experiment index](Experiment-trajectory)

This page groups the reported 5 and 6 October runs with the implementation reviews that preceded their continuations. The original review notes do not record an exact date for every check. Historical plans are preserved; see the index for the current decision.

## On this page

- [Session replay, 5 October](#session-replay-5-october)
- [Learning without unlearning, 5 and 6 October](#learning-without-unlearning-5-and-6-october)
- [Hyperfan learning runs, 6 October](#hyperfan-learning-runs-6-october)
- [Task-17 continuation, beta 0.01](#task-17-continuation-beta-001)
- [Learning audit and its qualification](#learning-audit-and-its-qualification)
- [Hyperfan implementation and CPU checks](#hyperfan-implementation-and-cpu-checks)
- [Paper recheck and diagnostic validation](#paper-recheck-and-diagnostic-validation)

## Session replay, 5 October

The user reported two fresh-session L9 replays from the same selected post-U3 model:

| Run | Final task 0 (%) | Final task 9 (%) |
| --- | --- | --- |
| A | 29.2 | 43.8 |
| B | 36.4 | 42.2 |

The comparator's first recorded difference was at training step 1: model buffers, loss, and parameters differed. It did not list image order or input values among those first differences. Environment or setup metadata also differed. This narrows the first observed mismatch but does not identify its cause. The supplied output does not contain the full environment comparison or source fingerprint.

Reported comparison: `/content/drive/MyDrive/uncle/E08_forgetting_trace/L9_session_replay/20261005_L9_replay_01/comparison.json`.

For a later pair named `20261005_L9_same_gpu_01`, run A's session ended after reporting saved training results. Run B subsequently reported 395 saved training steps and Drive flushing. No final A/B comparison for that pair was supplied, so it is not recorded as a reproducibility success. Deterministic settings were discussed; no supplied controlled result establishes that they resolve the earlier mismatch.

## Learning without unlearning, 5 and 6 October

Two reported learning-only runs started from task 3 = 26.0% and task 0 = 44.6%, then learned 9, 5, and 17. The printed heading called these test scores; Tiny ImageNet evaluation in this project uses validation images.

| Beta | Stage | Task 3 (%) | Task 0 (%) | Task 9 (%) | Task 5 (%) | Task 17 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | After learning 9 | 19.0 | 40.6 | 41.6 | Not learned | Not learned |
| 0.01 | After learning 5 | 16.2 | 33.0 | 34.8 | 42.4 | Not learned |
| 0.01 | After learning 17 | 22.4 | 29.4 | 29.8 | 44.4 | 41.2 |
| 0.1 | After learning 9 | 15.4 | 40.0 | 39.0 | Not learned | Not learned |
| 0.1 | After learning 5 | 14.2 | 34.4 | 31.0 | 36.4 | Not learned |
| 0.1 | After learning 17 | 18.8 | 35.8 | 29.4 | 29.8 | 34.0 |

Both runs show old-task decline during learning. Higher beta helped task 0 in this pair but lowered several other final scores. This pair alone does not establish a reliable beta effect. These models use the historical initialization and are not the source for the later Hyperfan continuation.

## Hyperfan learning runs, 6 October

The revised initialization was tested with learning-only sequences. The following tables are separate runs; the four-task and five-task outputs have different scores from their first lessons. They cannot be combined into a single uninterrupted trajectory. Continuation from an explicitly selected checkpoint was subsequently implemented in notebook 14.

| Run | Stage | Task 3 (%) | Task 0 (%) | Task 9 (%) | Task 5 (%) | Task 17 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Three tasks | After learning 3 | 46.0 | Not learned | Not learned | Not learned | Not learned |
| Three tasks | After learning 0 | 42.2 | 40.6 | Not learned | Not learned | Not learned |
| Three tasks | After learning 9 | 39.0 | 44.9 | 51.2 | Not learned | Not learned |
| Four tasks | After learning 3 | 46.8 | Not learned | Not learned | Not learned | Not learned |
| Four tasks | After learning 0 | 43.4 | 47.0 | Not learned | Not learned | Not learned |
| Four tasks | After learning 9 | 34.8 | 34.0 | 47.8 | Not learned | Not learned |
| Four tasks | After learning 5 | 27.0 | 31.2 | 48.4 | 54.8 | Not learned |
| Five tasks | After learning 3 | 43.2 | Not learned | Not learned | Not learned | Not learned |
| Five tasks | After learning 0 | 43.8 | 37.0 | Not learned | Not learned | Not learned |
| Five tasks | After learning 9 | 39.0 | 20.4 | 46.8 | Not learned | Not learned |
| Five tasks | After learning 5 | 37.6 | 26.0 | 48.8 | 50.8 | Not learned |
| Five tasks | After learning 17 | 35.6 | 18.6 | 37.4 | 49.2 | 52.6 |

The four-task source was inspected in Colab. It contains completed tasks `['3', '0', '9', '5']`, scores `{'3': 27.0, '0': 31.2, '9': 48.4, '5': 54.8}`, initialization `hyperfan_in`, and learning rate `0.0001`.

Confirmed source: `/content/drive/MyDrive/uncle/learning_initialization/20261006_hyperfan_L3_L0_L9_seed0_02/checkpoint_seq1_resnet50_seed0.pt`.

Hyperfan learning scores do not establish that the initialization change alone improves retention. These are one-seed runs, and the comparison does not isolate each initialization component.

## Task-17 continuation, beta 0.01

The user supplied notebook 14's five-epoch output for teaching task 17 from the selected checkpoint after tasks 3, 0, 9, and 5. Source scores below are from the user's earlier checkpoint inspection. The Drive report, source hash, and gradient samples have not been inspected locally. The notebook's final file-check and GPU-release cell was still pending in the supplied output.

Source: `/content/drive/MyDrive/uncle/learning_initialization/20261006_hyperfan_L3_L0_L9_seed0_02/checkpoint_seq1_resnet50_seed0.pt`.

Result: `/content/drive/MyDrive/uncle/learning_loss_diagnostic/20261006_learning_loss_01/beta_0_01/checkpoint.pt`.

| Task | Saved starting score (%) | After learning 17 (%) | Change (percentage points) |
| --- | --- | --- | --- |
| 3 | 27.0 | 24.6 | -2.4 |
| 0 | 31.2 | 28.0 | -3.2 |
| 9 | 48.4 | 32.6 | -15.8 |
| 5 | 54.8 | 51.2 | -3.6 |
| 17 | Not supplied | 53.4 | Not calculated |

| Epoch | New-task loss | Beta times protection | Task 3 (%) | Task 0 (%) | Task 9 (%) | Task 5 (%) | Task 17 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.8112 | 0.8174 | 19.2 | 26.6 | 50.0 | 50.0 | 35.8 |
| 2 | 1.4661 | 0.4982 | 26.2 | 40.8 | 43.8 | 49.2 | 43.8 |
| 3 | 1.3163 | 0.4087 | 24.8 | 29.6 | 37.8 | 50.6 | 45.4 |
| 4 | 1.1408 | 0.3998 | 25.6 | 27.0 | 43.8 | 47.0 | 50.8 |
| 5 | 1.0235 | 0.4102 | 24.6 | 28.0 | 32.6 | 51.2 | 53.4 |

The mean old-task score fell from 40.35% to 34.10%, a loss of 6.25 percentage points. The notebook displayed this as -6.2 points. Task 9 had the largest final drop. Old-task scores fluctuated across epochs, so the decline was not monotonic.

The positive protection term shows that protection contributed to the objective. Its scalar value does not establish the strength or direction of its gradient, or prove that increasing beta will improve retention. This run confirms older-task degradation during the new lesson under these settings; it does not identify the full cause or establish a result across seeds.

For the batch comparison, finish notebook 14's final cell, then use a fresh A100 session with `MODE = "RUN"`, `EXPERIMENT_ID = "20261006_learning_loss_01"`, and `BETAS = (0.0, 0.001, 0.01, 0.1, 1.0)`. The saved selection is reused; no new selection or retraining of the source tasks is needed. Compare new-task accuracy, mean old-task change, and the largest old-task drop. The beta batch changes only the weight of protection during task 17, keeping the checkpoint, code, seed, images, learning rate, and training budget fixed.

## Learning audit and its qualification

These notes describe the review and software checks preceding the reported continuation. Proposed steps in this historical section are superseded by the completed results on the [7 October page](Experiment-trajectory-2026-10-07).

The protection penalty passed a direct equation and gradient check. The clearest implementation difference is initialization: our code does not implement the paper's Hyperfan method. This is a candidate explanation for poor learning and retention, not a demonstrated cause.

This audit checked training revision `83e1c32773fc59886c8b43a0ed159ccf6424637a`, which notebook 12 selects. The local pooling and determinism edits are not part of that revision. No GPU training was run and no training code was changed.

### What the two completed experiments show

Both runs started from the saved model after learning tasks 3 and 0, then learned tasks 9, 5, and 17 without any unlearning. The user supplied these final test scores:

| Task | Beta 0.01 | Beta 0.1 | Difference in percentage points |
| --- | ---: | ---: | ---: |
| 3 | 22.4% | 18.8% | -3.6 |
| 0 | 29.4% | 35.8% | +6.4 |
| 9 | 29.8% | 29.4% | -0.4 |
| 5 | 44.4% | 29.8% | -14.6 |
| 17 | 41.2% | 34.0% | -7.2 |

Increasing beta helped task 0 in this pair, but did not improve every older task and produced lower scores on newly learned tasks. One run per setting does not establish a reliable beta effect. Learning alone can reduce older task scores, so unlearning is not required for the decline observed here.

### Protection penalty

Equation 2 compares the generated model weights with weights from a frozen copy of the hypernetwork taken before learning the next task. It sums squared differences over model parameters, then averages across previous tasks. Our `UnCLe.preserve()` follows that formula for the supplied protected tasks. It uses the same scaled weights as prediction. Dividing again by the number of model parameters would change this objective. [Paper, Learning](https://arxiv.org/html/2509.17530#S3.SS1.SSS1)

A CPU check used a small two-layer target, two protected tasks, and the pinned hypernetwork code. The checked trainer file was identical to the pinned trainer file. Results with PyTorch `2.12.0+cpu`:

- Penalty before changing the generator: `0.0`.
- Penalty after adding `0.03` to the output-head weights: `0.1256491243839264`.
- Independently calculated squared-distance penalty: `0.1256491243839264`.
- Largest gradient difference between the two calculations: `0.0`.
- The snapshot had no trainable parameters.

This checks the penalty's arithmetic and gradient on a small model. It does not prove that protection is strong enough during ResNet50 training.

### Initialization

Initialization means choosing the model's numbers before its first lesson. The paper specifies Hyperfan initialization to produce Kaiming He initialized main-network parameters. Our code instead initializes the generator's hidden layers with Kaiming normal, its output heads with another normal distribution, then applies hand-selected output scales. Those steps are not an implementation of Hyperfan. [Paper, Learning](https://arxiv.org/html/2509.17530#S3.SS1.SSS1)

A separate CPU measurement used the actual Tiny ImageNet ResNet50 configuration, hidden widths 128, 256, and 512, 200 chunks, seed 0, and task code 3. It measured a fresh model, not the saved E08 checkpoint:

| Generated parameter | Mean | Standard deviation | Applied scale |
| --- | ---: | ---: | ---: |
| `conv1.weight` | -0.004057 | 0.194999 | 0.272166 |
| `bn1.weight` | -0.000770 | 0.007965 | 0.01 |
| `bn1.bias` | -0.000326 | 0.008961 | 0.01 |
| `fc.weight` | 0.000000589 | 0.000448 | 0.000488 |

The final classifier uses a scale of `1 / 2048`. A fan-in He reference for that shape has standard deviation `sqrt(2 / 2048) = 0.03125`. The measured classifier spread is approximately 70 times smaller than that reference. The paper does not specify every classifier and BatchNorm initialization detail, so this comparison does not establish the authors' exact tensors.

`bn1.weight` controls the size of normalized activations. Our generated values start close to zero; the target template's BatchNorm weights start at one. These differences justify checking initialization before another beta sweep. Their effect on trained accuracy remains untested.

### Other confirmed difference

The paper states Adam with learning rate `0.001` and a scheduler. Our learning loop has Adam but no scheduler. E08 used the diagnostic learning rate `0.0001`; notebook 12 inherits the starting checkpoint's configuration. The cited appendix does not supply an identifiable scheduler recipe, so an exact schedule should not be invented. [Paper, Implementation](https://arxiv.org/html/2509.17530#S4)

### Next target and steps

**Target: make the initialization match a documented Hyperfan method, and verify the generated starting weights before spending more GPU time.**

1. Prepare the initialization change from the primary Hyperfan reference. Document choices the UnCLe paper leaves unspecified, especially BatchNorm and the final classifier.
2. Check generated weight statistics and a finite forward/backward pass on CPU. Keep the protection equation unchanged.
3. After those checks pass, run a short fresh sequence: learn 3, then 0, then 9. Use beta `0.01`, keep the diagnostic learning rate fixed for this pilot, and record each task's score after each lesson. Judge whether learning and retention warrant a longer experiment.

A fresh start is necessary to test initialization. Loading the old checkpoint overwrites newly initialized generator weights. Also, the output scales are not saved in the state dictionary: changing those scales would change the old checkpoint's generated model even if loading succeeds. Preserve the old code and checkpoint together.

The next pilot would assess the revised implementation. It would not establish generalization across seeds or reproduce the full paper.

### Qualification after reading the Hyperfan reference

The original convolution-weight recipe is already algebraically consistent with the weight-only Hyperfan-in variance formula: a raw head with weight variance `1 / hidden_width`, followed by multiplication by `sqrt(2 / fan_in)`, has the required effective variance when the last hidden activations have unit second moment. The audit's claim that the entire initialization recipe differs from Hyperfan was too broad. The classifier and one-dimensional parameter policies are the substantive differences. The measured spread of one freshly generated tensor is not, by itself, evidence that this variance formula is wrong.

The new option and its choices are documented in [Hyperfan pilot](#hyperfan-implementation-and-cpu-checks). The historical measurements above remain measurements of the original recipe.

## Hyperfan implementation and CPU checks

These notes describe the review and software checks preceding the reported continuation. Proposed steps in this historical section are superseded by the completed results on the [7 October page](Experiment-trajectory-2026-10-07).

The new `Config(initialization="hyperfan_in")` option changes the starting recipe for a fresh model. The default remains `legacy` so historical runs and checkpoints keep their original generated weights. The protection objective is unchanged.

### Formula and implementation choices

Hyperfan-in sets the effective output-head weight variance to `gain_squared / (bias_factor * target_fan_in * hidden_width * embedding_variance)`. Here embedding variance is one, `bias_factor` is two when generating an affine layer's bias, and gain squared is two for ReLU layers or one for linear layers. Generated bias variance is `gain_squared / 2`. See Table 1 and section 4.1 of [Chang, Flokas, and Lipson](https://arxiv.org/html/2312.08399#S4.SS1).

Our shared chunk heads generate parameters with different fan-in values. A head initialized with variance `1 / hidden_width`, followed by a separate scale for each generated tensor, implements the effective variance formula. The hidden ReLU trunk keeps its Kaiming fan-in initialization. Neither embeddings nor chunks are normalized using observed samples.

The original bias-free convolution recipe was already consistent with this formula. The new recipe changes these policies:

- Convolution weights use gain squared two. If a convolution has a generated bias, half the variance budget goes to that bias.
- The final classifier is linear, with gain squared one. Its weight variance is `1 / (2 * fan_in)` and its generated bias variance is `1 / 2`.
- BatchNorm starts with gamma one and beta zero. Its output head starts at zero and remains trainable. An additive offset supplies gamma one. Output scale is one for both parameters.

The UnCLe paper does not specify these classifier and BatchNorm details or which Hyperfan variant it used. These are documented reproduction choices, not recovered author code. The new option has been checked for learning; this pilot does not assess its forgetting behavior.

### CPU evidence

[Saved check](https://github.com/sumitasthana/PATH/blob/main/docs/experiments/hyperfan_cpu_check_20261006.json): full Tiny ImageNet ResNet50 shapes, 200 chunks, hidden widths 128/256/512, seed 0, PyTorch 2.12.0+cpu. Original adaptive average pooling was used, matching the published pilot.

- Generated classifier weight standard deviation: `0.0143448` (formula reference `0.015625`). The old recipe measured `0.0004483` for the same seed and architecture.
- BatchNorm gamma was exactly one and beta exactly zero.
- A synthetic two-image forward/backward pass had finite generated values, predictions, loss, and gradients. Cross-entropy was `2.8180091`.
- BatchNorm-head gradients were nonzero. The smaller automated check also verified that an optimizer update moves gamma.
- Across 24 independent initializations, generated affine weight and bias second moments were within 20% of the formula references.
- Legacy generation matched the pinned historical implementation exactly.
- Protection values and gradients matched a direct squared-distance calculation with the new offsets.

Synthetic images do not measure dataset accuracy. A single tensor's empirical spread need not equal the formula's expectation across random initializations.

### Checkpoints and notebook

Checkpoint files record the initialization option. Files predating this field are interpreted as `legacy`. Restoring a checkpoint into a model with a different initialization option is refused before loading model state. The offsets and scales are derived from that option.

Notebook [13](https://github.com/sumitasthana/PATH/blob/main/notebooks/13_hyperfan_learning_pilot.ipynb) starts a new model and learns tasks 3, 0, and 9. Settings: seed 0, beta 0.01, learning rate 0.0001, five epochs per task, batch size 64, ResNet50, and the existing fixed class partition. No scheduler or deterministic-mode change is introduced. Learning rate and epoch count remain diagnostic settings, so this is not a complete paper reproduction.

The notebook saves the code version, settings, class-partition hash, task scores, environment, and a checkpoint after each completed task. It can resume its own checkpoint after a disconnected session. It does not load E08 or any other historical starting model. Its final cell checks the files, flushes Drive, then releases the GPU.

Judge the pilot by each task's learning score and the change in earlier scores after the next lesson. Do not use this one run to claim a seed-independent improvement or identify one of the changed initialization components as the cause. Stop after task 9 and review the saved table before extending the sequence.

### Validation and code version

Training code is pinned to `b1797e0a93644f501508dfb856d38ad921bb6db8`. The published version retains the original ResNet pooling.

Forty-one automated checks passed across Hyperfan statistics, historical output compatibility, experiments, replay, and notebook helpers. The additional pilot notebook check passed using real CPU training and checkpoint files with a small stand-in target. It simulated a disconnect after saving task 3, resumed tasks 0 and 9, reopened the completed run without training again, and verified that Drive flushing precedes runtime release. Colab and Drive services were mocked; no remote GPU run was performed during validation.

## Paper recheck and diagnostic validation

These notes describe the review and software checks preceding the reported continuation. Proposed steps in this historical section are superseded by the completed results on the [7 October page](Experiment-trajectory-2026-10-07).

The new-task objective and protected-task implementation agree with the paper's stated learning equation. The review did not find a missing replay buffer, a missing old-task loss on images, or a missing extra coefficient. It did find remaining reproduction gaps and a confirmed difference in the forgetting path. The current learning-only results cannot establish complete agreement with the paper.

### Paper and code comparison

Reviewed sources: the [arXiv HTML](https://arxiv.org/html/2509.17530) and [17-page PDF](https://arxiv.org/pdf/2509.17530), version 1. A search also surfaced an OpenReview submission, but access returned a browser-verification page; its full contents were not verified. No author implementation was established from the searches performed.

| Item | Finding in our code |
| --- | --- |
| Learning objective, Equation 2 | Cross-entropy plus beta times generated-weight preservation. Squared differences are summed over target parameters and averaged across protected tasks. No additional averaging across parameters is applied. |
| Frozen reference | Snapshot before adding the new task; cached reference weights come from that snapshot. Cache is reset for each lesson. |
| Trainable parameters | Shared generator and new task code. Old task codes are frozen. Chunk codes are trained only for the first task and frozen afterwards. |
| Protected tasks | All previously learned tasks except the task being taught. No forgetting occurs in the pilot or new diagnostic. |
| Architecture | ResNet50, hidden widths 128/256/512, task and chunk codes of size 32, 200 chunks, separate parameter-type heads. |
| Hyperfan | Opt-in Hyperfan-in variance recipe is available. The original convolution scaling was already compatible with its weight-only formula. Final-classifier, generated-bias, and BatchNorm choices are documented implementation choices. |
| Optimizer | Adam, newly constructed for each lesson. The paper does not state whether Adam state persists across lessons. |
| Learning rate and schedule | Pilot uses constant 0.0001. Paper states 0.001 and a scheduler, but an identifiable schedule was not supplied in the reviewed appendix. This gap remains open. |
| Training length and batch size | Five epochs and batch size 64 are our diagnostic settings. The reviewed appendix does not identify the promised values. |
| Target stem and data | Our ResNet uses a 3x3 stride-one stem. Images use ToTensor without augmentation or normalization, with a fixed class partition. These details are not fully specified by the paper. |
| BatchNorm running statistics | Stored separately for each task; old buffers are not updated during later learning. This handling is not specified by the paper. |
| Forgetting representation | Noise alignment uses raw generator outputs, while preservation uses scaled target weights. Equation 3 uses the same H for both terms. The two code paths therefore require resolution before claiming a faithful unlearning implementation. |
| Benchmark scope | The five-task learning-only pilot is a diagnostic. It is not the paper's mixed 30-request Tiny ImageNet sequence or a multi-seed result. |

The learning equation, architecture, freezing rule, optimizer statement, and beta search are given in the [Learning, Implementation, and appendices](https://arxiv.org/html/2509.17530). Unknown details above are not assumed to be implementation bugs. The noise-representation difference cannot explain accuracy decline in a run with no forgetting.

### Confirmed CPU checks

The new instrumentation passed a comparison against training without instrumentation. Generated parameters, BatchNorm buffers, and returned epoch losses were identical. The test evaluated validation scores and consumed random numbers inside callbacks, confirming that callback mode and PyTorch RNG isolation work.

The same check verified that old task codes and chunk codes are frozen and unchanged, and that old BatchNorm buffers stay unchanged. Protection was zero before the first update and positive after the generator moved. A zero initial protection gradient is expected for squared distance to an identical snapshot; it is not evidence of a disabled term.

Continuation tests restored actual CPU checkpoint files, trained exactly one additional task, preserved the source file hash and earlier history, and saved a new model. Two different beta branches had the same source hash, starting scores, and first-batch task loss. Completed runs were not retrained. Interrupted runs kept their partial epoch report and required a separate attempt folder.

The notebook test used its actual selection, training, and finishing cells with a small CPU target and mocked Colab services. SELECT mode did not train. RUN mode added exactly one task. File checks preceded Drive flushing and GPU release. These are machinery checks, not new dataset accuracy results.

### What to run next

Notebook [14](https://github.com/sumitasthana/PATH/blob/main/notebooks/14_learning_loss_diagnostic.ipynb) makes source selection explicit:

1. Use a CPU session in SELECT mode. It lists saved model files on Drive. Select a file and inspect its actual saved tasks and scores.
2. Prefer a source after L3, L0, L9, and L5, so the next lesson is L17. If that checkpoint no longer exists, use a retained source after L3, L0, and L9, then diagnose L5. Do not use the final model after L17 to recreate a model before L17.
3. Save the selection. In a fresh A100 session, use RUN mode with beta 0.01. Source scores must match before training starts. Only the next task is taught.
4. Record cross-entropy, unweighted protection, beta times protection, and combined loss for every update. Save validation scores and batch-mean loss components after every epoch.
5. Sample shared-generator gradient norms and their cosine similarity on the first batch of each epoch. The gradient of beta times protection is compared with the task gradient. The new task code is excluded from this comparison because protection does not act on it.
6. Verify the final history and save the new checkpoint in a separate folder. Keep the source file unchanged. Flush Drive and release the GPU.

Loss and gradient samples describe the model before an update. Epoch accuracies describe it after that epoch. Raw gradient norms are useful diagnostics, but Adam's adaptive scaling means they are not a direct measurement of the resulting parameter movement. Negative gradient cosine indicates conflict on that sampled group; sparse samples do not prove behavior at every update.

The new notebook preserves the source model's initialization, epochs, batch size, seed, and learning rate. It does not silently introduce a scheduler or change the learning objective. For an interrupted lesson, epoch measurements survive, but the current optimizer is not saved mid-lesson; a new attempt restarts that lesson from the original source.

### Search decision

The paper searches beta in `{0.001, 0.01, 0.1, 1}` and selects 0.01 for Tiny ImageNet. [Appendix C, Table 5](https://arxiv.org/html/2509.17530)

There is one relative loss weight in our learning objective: beta. A grid over two independent loss coefficients is not needed for this objective. A beta-by-learning-rate grid is a separate, larger experiment and is not scheduled now.

The immediate plan is one measured beta-0.01 lesson. If its losses and gradients show that protection is active and correct, an optional four-value beta sweep is already supported by changing BETAS in notebook 14. Every value starts from the same selected checkpoint, with the same source RNG state, code, seed, task images, and training settings. Each value has its own result folder. The completed baseline can be reused.

Compare both new-task accuracy and old-task score changes. The comparison report includes the mean old-task change, worst old-task drop, and individual changes; it does not automatically pick a winner by retention alone. A setting that prevents all old-task loss but fails to learn the new task is not sufficient.

This continuation sweep tests beta only for the new lesson. It cannot reproduce the paper's beta search throughout an entire sequence or establish generalization across seeds. If beta does not improve the learning/retention tradeoff, the next controlled question is the documented optimizer/schedule gap, not an unlimited expansion of the search.

### Validation and publication

Training revision: `6a59648a5512724f94917ed9b97a12807c68809b`. The targeted CPU regression set passed 29 tests; one CUDA RNG test was skipped because this machine has no CUDA runtime. Notebook code cells compile, and the written prose was checked for the repository style constraint. No Colab GPU training or inspection of the user's actual Drive checkpoint files was performed locally.
