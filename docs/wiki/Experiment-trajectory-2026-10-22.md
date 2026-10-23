# Experiment trajectory: 22 October 2026

[Back to the experiment index](Experiment-trajectory)

## EuroSAT probe plan and rationale

Status: planned. No EuroSAT probe has been implemented or run. This plan extends the completed Tiny ImageNet plasticity measurement; it does not replace its results.

A review on 23 October proposed deferring this plan because its t = 0 reference is an untrained generator. That reasoning was wrong and has been withdrawn. The measured t = 0 full probe reaches about 52 percent on ten-class tasks, so the untrained generator learns perfectly well inside the probe budget. The [23 October record](Experiment-trajectory-2026-10-23) holds the correction. What survives from that review is a set of additions, listed under "What the review did change" below.

### Question

After learning eight Tiny ImageNet tasks, can our generator learn satellite-image classification as well as a fresh generator can?

Plasticity means the ability to learn a new task. Our existing probes use unseen classes from Tiny ImageNet. The 20 and 21 October record reports improved learning on those probes, but earlier Tiny ImageNet training may help because the images come from the same dataset. An external dataset tests whether that improvement also appears on a different kind of image. It does not remove every possible source of transfer.

### What the t = 0 comparison measures, and what it does not

The comparison measures learnability relative to initialization. It does not separate transfer from plasticity changes, and it does not rule out deterioration relative to an earlier trained checkpoint.

Two consequences follow. Useful transfer can hide deterioration, because a model that has become harder to train may still score higher on a related probe. And a rise early in the sequence followed by a fall later is invisible when only the two endpoints are measured.

Neither is a reason to drop the t = 0 measurement. Both are reasons to add measurements at trained checkpoints in between.

### Why EuroSAT RGB

EuroSAT contains 27,000 satellite images in ten land-use and land-cover classes. Use its RGB version, which has the ordinary red, green, and blue image channels.

The ten classes matter. The generator's output layout is fixed when it is built, from a classifier with `classes_per_task` outputs, and that value is 10. Of the adapter study's three natural probe tasks, EuroSAT at ten classes is the only one that fits. DTD has 47 classes and Flowers102 has 102, so neither can be generated at all.

Satellite images give a different classification problem from the object photographs used in Tiny ImageNet. Dataset source: [official EuroSAT repository](https://github.com/phelber/EuroSAT), with downloads linked through Zenodo.

Use all ten classes together as one probe task. Do not split EuroSAT into another continual-learning sequence. Keep its images out of the Tiny ImageNet training stream.

### Models and parameters

Use the same order-01 source models as notebook 29, for seeds 0, 1, and 2. Order 01 learns tasks 3, 0, 9, 5, 17, 1, 7, and 14.

| Starting state | What it represents | Available now |
| --- | --- | --- |
| t = 0 | Generator initialized with the source configuration and seed, before learning any task | Yes |
| t = 1, 2, 4, 6 | Generator partway through the sequence | No. Notebook 29 records that no mid-stream checkpoints were kept |
| t = 8 | Saved generator after the eight Tiny ImageNet tasks | Yes |

Run t = 0 and t = 8 now. Add the intermediate states when the sequence is next trained. `uncle/checkpoint.save` writes to whichever path it is given, so saving one file per task needs only a per-task path, not a change to the function.

For each measurement, make a temporary copy, add a fresh 32-number EuroSAT task code, train, measure, and discard the changes. Train the generator's network parameters and the new task code. Keep chunk embeddings and existing task codes fixed. The classifier weights change through the generator; they are not independent optimizer parameters. Disable the old-task protection penalty during this discarded probe, because the probe is thrown away and there is nothing to protect.

Initialize fresh probe-specific running statistics and do not reuse statistics from a Tiny ImageNet task. Restore and verify model parameters, task codes, and buffers after every probe. Match the new task-code initialization and training batch order between model states for each paired repetition.

Rationale: the comparison should change the generator's training history while keeping the probe training procedure the same. Use the full generator-plus-code arm as the primary experiment. Run the code-only arm alongside it, as the adapter study runs its head-only control. Read the two separately. A code-only arm near chance says that 32 numbers on a frozen generator cannot learn the task. It says nothing about whether the full probe, which also trains the generator, can.

The generator-only arm is optional; the 21 October record shows that training a task's 32 numbers buys nothing on Tiny ImageNet probes.

### Data and training settings

Proposed initial settings:

| Setting | Plan | Rationale |
| --- | --- | --- |
| Data split | Fixed, disjoint, class-balanced subsets: 800 training, 200 validation, 2,000 test images | Matches the adapter study's EuroSAT probe sizes and controls class proportions |
| Split identity | Save file identifiers, labels, split seed, and a manifest hash | Makes every model see exactly the same probe |
| Preprocessing | Deterministic RGB conversion and resize to our classifier's configured input size; use one fixed normalization for both model states | Avoids changing the input procedure between comparisons |
| Training budget | Initially 250 steps, batch size 64 | Starts from notebook 29's natural-probe budget |
| Learning rates | 0.0001, 0.0003, 0.001 | Allows each model state to learn at a suitable rate |
| Optimizer | Fresh optimizer for every probe, using notebook 29's configuration | Removes inherited optimizer history from the comparison |
| Repetitions | Five at t = 0, two at each later state, per training seed | The adapter study spends more repetitions on the point every other number is compared against |

EuroSAT RGB images and Tiny ImageNet images are both 64 by 64 pixels. Check whether the resize step does anything before writing a resize path.

With unequal repetition counts, state which repetitions are paired. Pair by repetition index up to the smaller count, and treat the extra repetitions at t = 0 as contributing to the noise floor only, not to a paired difference. `uncle/probe.noise_floor` already takes separate counts for the two points.

Select the learning rate separately for each model state, seed, and repetition using final validation accuracy. Break ties with a fixed rule, such as choosing the lower rate. Evaluate the selected run on the test split only after selection. Do not choose rates or training budgets from test results. This corrects notebook 29's practice of selecting natural-probe rates using test accuracy, so earlier natural-probe numbers rest on a different rule and should not be compared directly with numbers selected the new way.

### Pilot before the full experiment

Prepare the data and verify split separation, image shapes, ten-class labels, and checkpoint provenance on CPU. Then run a seed-0 pilot at both model states, with one paired repetition and all three learning rates: six short GPU runs.

The pilot decides whether both starting states learn meaningfully inside the budget. Do not assume either answer. On Tiny ImageNet the t = 0 full probe reached about 52 percent against 10 percent chance, so an untrained generator can learn a ten-class task in 250 steps. Whether that carries to satellite images is the question the pilot answers.

Use training loss and validation curves. If both models remain near chance, this is not evidence of equal plasticity. If both saturate very early, final accuracy may conceal differences in learning speed. In either case, revise the budget using pilot validation results, record the reason, and freeze the final settings before the main experiment. Treat pilot runs as calibration, not as independent confirmation.

### Measurements and interpretation

Primary measurement: test accuracy after the fixed training budget, at the validation-selected learning rate.

For each seed and repetition, calculate:

```text
accuracy change = accuracy after eight tasks - fresh-model accuracy
```

A negative value means poorer learning on EuroSAT after the Tiny ImageNet sequence. A positive value means better learning on EuroSAT. Neither result alone identifies a mechanism.

Also save training loss and validation accuracy at fixed step intervals. These curves show whether one model learns more slowly even when final accuracy is similar. Any learning-speed summary must use the same evaluation intervals and training budget at both states.

Report each seed's paired changes, their mean, and variation across repetitions. Then report the mean and sample standard deviation across the three training seeds. Estimate measurement noise from repeated probes of the unchanged fresh model. Use paired uncertainty estimates that retain the repetition matching; do not count all repetitions as independent training seeds. With only three training seeds, broad population claims remain provisional even if a bootstrap interval is narrow.

Use a deterioration verdict only when the paired change is negative, its uncertainty interval excludes zero, and its magnitude exceeds the prespecified noise floor. Write down the exact noise-floor and interval calculations before the main run. `uncle/probe.noise_floor` holds the formula. Otherwise report no clear detected deterioration, not proof of unchanged plasticity.

| Possible observation | Supported interpretation |
| --- | --- |
| No clear EuroSAT decline | No detected loss on this external task, at these settings |
| EuroSAT declines while Tiny ImageNet probes improve | Learning history has different effects on the two probe domains; the Tiny ImageNet gain does not generalize to EuroSAT |
| Both model states fail to learn EuroSAT | Probe budget or setup is inadequate for answering the question |
| EuroSAT improves | Prior training also helps this external task; transfer remains a possible explanation |

Keep the existing Tiny ImageNet findings alongside these results. They used a different rate-selection rule, so do not combine the scores into one averaged endpoint. A direct numerical comparison across datasets would require the same validation-based protocol on both.

### Diagnostics worth adding

The adapter study records adapter geometry after every one of its twenty tasks, not only at its six probe points. That record is what lets it tie a change in learning to a change in the adapter's shape. We record nothing between probe points.

These are exploratory diagnostics. They can help explain a change in learnability. They cannot replace the probe, and a small reading on any of them does not establish that probing is pointless.

| Quantity | Cost | What it needs |
| --- | --- | --- |
| Relative weight change for a task, against that task's weights at an earlier point | Weights only | The earlier model state, or generated weights saved at the time. An endpoint checkpoint cannot reconstruct movement across the sequence |
| Condition number of each generator head's output layer | Weights only | Nothing extra. Treat as exploratory; it is not an established equivalent of the adapter study's LoRA factor conditioning |
| Task code norms, and the smallest angle between any two learned codes | Weights only | Nothing extra |
| Fraction of silent channels | Forward pass | A fixed input set, the generated weights, the normalization statistics, and a stated evaluation mode. CPU is enough, but it is not a weight-only measurement |
| Feature effective rank | Forward pass | A fixed input set. `uncle/probe.effective_rank` computes it |

The weight-only quantities can run on checkpoints we already hold, for the states those checkpoints represent. Movement across the sequence needs per-task saves from the next training run.

### Compute, saving, and stopping

The original grid was 3 training seeds x 2 model states x 5 repetitions x 3 learning rates = 90 probe runs.

The revised grid changes the repetition counts and adds the code-only arm:

```text
3 seeds x (5 repetitions at t = 0 + 2 at t = 8) x 3 learning rates x 2 arms
= 126 probe runs
```

Each intermediate state added later costs a further 3 x 2 x 3 x 2 = 36 runs. Each run is initially 250 steps. The t = 0 and t = 8 grid needs GPU probe training but no retraining of the eight-task stream; the intermediate states need the stream trained again with per-task saves.

Measure pilot runtime and peak GPU memory before estimating the full cost; do not reuse notebook 29's runtime estimate without checking.

Run one training seed per session if needed. Save each completed probe, selected-rate metadata, and learning curves to R2, and verify the saved files before releasing the GPU. Record source checkpoint hashes, code version, dataset manifest hash, preprocessing, optimizer settings, rate, seed, repetition, and model state in the run identity. Skip only completed runs with the same identity. Restart an interrupted probe unless optimizer state, progress, and random-number state have also been saved; completed-run reuse is not mid-probe resume.

### Scope and next decision

This first extension tests one external task, one training order, three seeds, and two measurement points, with more points to follow. It does not identify a mechanism, and it does not show that unlearning restores plasticity.

We have no defined and validated equivalent of the adapter study's re-anchoring intervention. There it means folding the adapter into the frozen backbone and restarting the adapter at identity, so the model computes the same function while its coordinates reset. We have no frozen backbone to fold into. That rules out the specific operation. It does not establish that no function-preserving reparameterization of a hypernetwork exists, and finding one is open work rather than a closed question.

After reviewing the result, decide whether to add intermediate points, other task orders, or a second external dataset. An unlearning comparison would require matched branches from the same trained checkpoint, one unchanged and one unlearned, followed by identical probes. Call any gain improved learnability; use restored plasticity only if prior deterioration has been established.

Implementation should reuse `uncle/probe.py` and notebook 29's checkpoint and saving conventions, add the EuroSAT loader and validation-based rate selection, and keep results under a separate experiment identity. No training is started by this plan.
