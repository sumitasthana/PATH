# Experiment trajectory: 22 October 2026

[Back to the experiment index](Experiment-trajectory)

## EuroSAT probe plan and rationale

Status: planned. No EuroSAT probe has been implemented or run. This plan extends the completed Tiny ImageNet plasticity measurement; it does not replace its results.

### Question

After learning eight Tiny ImageNet tasks, can our generator learn satellite-image classification as well as a fresh generator can?

Plasticity means the ability to learn a new task. Our existing probes use unseen classes from Tiny ImageNet. The 20 and 21 October record reports improved learning on those probes, but earlier Tiny ImageNet training may help because the images come from the same dataset. An external dataset tests whether that improvement also appears on a different kind of image. It does not remove every possible source of transfer.

### Why EuroSAT RGB

EuroSAT contains 27,000 satellite images in ten land-use and land-cover classes. Use its RGB version, which has the ordinary red, green, and blue image channels. The ten classes fit our generated classifier's ten outputs. Satellite images provide a different classification problem from the object photographs used in Tiny ImageNet. EuroSAT also appears among the natural probes in the adapter study we are adapting. Dataset source: [official EuroSAT repository](https://github.com/phelber/EuroSAT), with downloads linked through Zenodo.

Use all ten classes together as one probe task. Do not split EuroSAT into another continual-learning sequence. Keep its images out of the Tiny ImageNet training stream.

### Models and parameters

Use the same order-01 source models as notebook 29, for seeds 0, 1, and 2. Order 01 learns tasks 3, 0, 9, 5, 17, 1, 7, and 14.

| Starting state | What it represents |
| --- | --- |
| t = 0 | Generator initialized with the source configuration and seed, before learning any task |
| t = 8 | Saved generator after the eight Tiny ImageNet tasks |

For each measurement, make a temporary copy, add a fresh 32-number EuroSAT task code, train, measure, and discard the changes. Train the generator's network parameters and the new task code. Keep chunk embeddings and existing task codes fixed. The classifier weights change through the generator; they are not independent optimizer parameters. Disable the old-task protection penalty during this discarded probe.

Initialize fresh probe-specific running statistics and do not reuse statistics from a Tiny ImageNet task. Restore and verify model parameters, task codes, and buffers after every probe. Match the new task-code initialization and training batch order between t = 0 and t = 8 for each paired repetition.

Rationale: the comparison should change the generator's training history while keeping the probe training procedure the same. Use the full generator-plus-code arm as the primary experiment. Code-only and generator-only arms are optional follow-ups; the existing Tiny ImageNet controls do not establish their behavior on EuroSAT.

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
| Repetitions | Five matched repetitions at each model state, per training seed | Estimates variability and gives equally paired measurements |

Select the learning rate separately for each model state, seed, and repetition using final validation accuracy. Break ties with a fixed rule, such as choosing the lower rate. Evaluate the selected run on the test split only after selection. Do not choose rates or training budgets from test results. This corrects notebook 29's practice of selecting natural-probe rates using test accuracy.

### Pilot before the full experiment

Prepare the data and verify split separation, image shapes, ten-class labels, and checkpoint provenance on CPU. Then run a seed-0 pilot at both model states, with one paired repetition and all three learning rates: six short GPU runs.

Use training loss and validation curves to check whether the budget permits learning. If both models remain near chance, this is not evidence of equal plasticity. If both saturate very early, final accuracy may conceal differences in learning speed. In either case, revise the budget using pilot validation results, record the reason, and freeze the final settings before the main experiment. Treat pilot runs as calibration, not as independent confirmation.

### Measurements and interpretation

Primary measurement: test accuracy after the fixed training budget, at the validation-selected learning rate.

For each seed and repetition, calculate:

```text
accuracy change = accuracy after eight tasks - fresh-model accuracy
```

A negative value means poorer learning on EuroSAT after the Tiny ImageNet sequence. A positive value means better learning on EuroSAT. Neither result alone identifies a mechanism.

Also save training loss and validation accuracy at fixed step intervals. These curves show whether one model learns more slowly even when final accuracy is similar. Any learning-speed summary must use the same evaluation intervals and training budget at both states.

Report each seed's paired changes, their mean, and variation across repetitions. Then report the mean and sample standard deviation across the three training seeds. Estimate measurement noise from repeated probes of the unchanged fresh model. Use paired uncertainty estimates that retain the repetition matching; do not count all repetitions as independent training seeds. With only three training seeds, broad population claims remain provisional even if a bootstrap interval is narrow.

Use a deterioration verdict only when the paired change is negative, its uncertainty interval excludes zero, and its magnitude exceeds the prespecified noise floor. Write down the exact noise-floor and interval calculations before the main run. Otherwise report no clear detected deterioration, not proof of unchanged plasticity.

| Possible observation | Supported interpretation |
| --- | --- |
| No clear EuroSAT decline | No detected loss on this external task, at these settings |
| EuroSAT declines while Tiny ImageNet probes improve | Learning history has different effects on the two probe domains; the Tiny ImageNet gain does not generalize to EuroSAT |
| Both model states fail to learn EuroSAT | Probe budget or setup is inadequate for answering the question |
| EuroSAT improves | Prior training also helps this external task; transfer remains a possible explanation |

Keep the existing Tiny ImageNet findings alongside these results. They used a different rate-selection rule, so do not combine the scores into one averaged endpoint. A direct numerical comparison across datasets would require the same validation-based protocol on both.

### Compute, saving, and stopping

The full primary grid is 3 training seeds x 2 model states x 5 repetitions x 3 learning rates = 90 probe runs, each initially 250 steps. It needs GPU probe training but no retraining of the eight-task stream. Measure pilot runtime and peak GPU memory before estimating the full cost; do not reuse notebook 29's runtime estimate without checking.

Run one training seed per session if needed. Save each completed probe, selected-rate metadata, and learning curves to R2, and verify the saved files before releasing the GPU. Record source checkpoint hashes, code version, dataset manifest hash, preprocessing, optimizer settings, rate, seed, repetition, and model state in the run identity. Skip only completed runs with the same identity. Restart an interrupted probe unless optimizer state, progress, and random-number state have also been saved; completed-run reuse is not mid-probe resume.

### Scope and next decision

This first extension tests one external task, one training order, three seeds, and two measurement points. It does not test re-anchoring, identify a chart mechanism, or show that unlearning restores plasticity.

After reviewing the result, decide whether to add intermediate points, other task orders, or a second external dataset. An unlearning comparison would require matched branches from the same trained checkpoint, one unchanged and one unlearned, followed by identical probes. Call any gain improved learnability; use restored plasticity only if prior deterioration has been established.

Implementation should reuse `uncle/probe.py` and notebook 29's checkpoint and saving conventions, add the EuroSAT loader and validation-based rate selection, and keep results under a separate experiment identity. No training is started by this plan.
