# Experiment trajectory: 22 October 2026

[Back to the experiment index](Experiment-trajectory)

## EuroSAT probe plan and rationale

Status: deferred. No EuroSAT probe has been implemented or run. This page was written on 22 October as a ready-to-run plan. A review on 23 October found a fault in its baseline, so the plan is kept here with that fault marked and with the settings that survive it. The [23 October record](Experiment-trajectory-2026-10-23) holds the review and the decision.

### The fault in the baseline

The plan compares two model states: t = 0 and t = 8. It defines t = 0 as a generator built from the source configuration and seed, before it has learned any task. That is a generator with random weights.

A random generator cannot do much with 800 images in 250 steps. It has never produced a working set of convolution filters. The t = 8 generator has. So the t = 8 model should win, and winning would say almost nothing about plasticity. It would say that a trained generator is a better starting point than an untrained one, which nobody doubts.

The adapter study this plan copies does not have this problem. Its t = 0 is a frozen ViT-B/16 that was pretrained on ImageNet-21k, with the adapter set to zero. That model is already strong, so comparing against it is safe.

`uncle/probe.py` already carries this warning in the note on `relative_contribution`: at t = 0 our control sits near chance, so the ratio divides by almost nothing.

This affects more than the EuroSAT plan. The 20 and 21 October record measures a gain of 4.74 points from t = 0 to t = 8 on Tiny ImageNet probes. Those numbers are correct as measured. What they cannot support is the reading that plasticity was preserved, because the reference model had never learned anything. The same applies to the random-label loss drop recorded there.

### What would fix it

Use t = 1 as the reference point and report t = 0 as a floor only. After one task the generator has a working visual system, so a later comparison is about how long the history is, not about whether there is a history at all.

This cannot be done with the models we have. Notebook 29 states that the study kept no mid-stream checkpoints, and `uncle/checkpoint.save` replaces any earlier checkpoint rather than keeping it. A sound reference point needs the stream trained again, saving the generator after every task.

That is the same retraining job the 21 October record already listed for the mid-stream probe points and for orders 02 and 03. It is one job, not three.

### Question, as it would be asked after the fix

After learning eight Tiny ImageNet tasks, does our generator learn satellite images as well as a generator that has learned one task?

Plasticity means the ability to learn a new task. Our existing probes use unseen classes from Tiny ImageNet. Earlier Tiny ImageNet training may help on those, because the images come from the same dataset. An external dataset tests whether the effect also appears on a different kind of image. It does not remove every possible source of transfer.

### Why EuroSAT RGB

EuroSAT contains 27,000 satellite images in ten land-use and land-cover classes. Use its RGB version, which has the ordinary red, green, and blue image channels.

The ten classes matter. Our generator's output layout is fixed when it is built, from a classifier with `classes_per_task` outputs, and that value is 10. Of the adapter study's three natural probe tasks, EuroSAT at ten classes is the only one that fits. DTD has 47 classes and Flowers102 has 102, so neither can be generated at all.

Satellite images give a different classification problem from the object photographs in Tiny ImageNet. Dataset source: [official EuroSAT repository](https://github.com/phelber/EuroSAT), with downloads linked through Zenodo.

Use all ten classes together as one probe task. Do not split EuroSAT into another continual-learning sequence. Keep its images out of the Tiny ImageNet training stream.

### Models and parameters

Use the same order-01 source models as notebook 29, for seeds 0, 1, and 2. Order 01 learns tasks 3, 0, 9, 5, 17, 1, 7, and 14.

| Starting state | What it represents | Available now |
| --- | --- | --- |
| t = 0 | Generator initialized with the source configuration and seed, before learning any task | Yes. Report as a floor, not as the reference |
| t = 1 | Generator after the first task | No. Needs the stream trained again |
| t = 8 | Saved generator after the eight Tiny ImageNet tasks | Yes |

For each measurement, make a temporary copy, add a fresh 32-number EuroSAT task code, train, measure, and discard the changes. Train the generator's network parameters and the new task code. Keep chunk embeddings and existing task codes fixed. The classifier weights change through the generator; they are not independent optimizer parameters. Disable the old-task protection penalty during this discarded probe, because the probe is thrown away and there is nothing to protect.

Initialize fresh probe-specific running statistics and do not reuse statistics from a Tiny ImageNet task. Restore and verify model parameters, task codes, and buffers after every probe. Match the new task-code initialization and training batch order between model states for each paired repetition.

Rationale: the comparison should change the generator's training history while keeping the probe training procedure the same. Use the full generator-plus-code arm as the primary experiment. Also run the code-only arm, because the pair of arms is the closest thing this architecture has to the adapter study's split between the part that holds the history and the part that is new to the probe. The generator-only arm is optional; the 21 October record already shows that training a task's 32 numbers buys nothing on Tiny ImageNet probes.

### Data and training settings

These settings are sound and survive the review. Proposed initial values:

| Setting | Plan | Rationale |
| --- | --- | --- |
| Data split | Fixed, disjoint, class-balanced subsets: 800 training, 200 validation, 2,000 test images | Matches the adapter study's EuroSAT probe sizes and controls class proportions |
| Split identity | Save file identifiers, labels, split seed, and a manifest hash | Makes every model see exactly the same probe |
| Preprocessing | Deterministic RGB conversion and resize to our classifier's configured input size; use one fixed normalization for both model states | Avoids changing the input procedure between comparisons |
| Training budget | Initially 250 steps, batch size 64 | Starts from notebook 29's natural-probe budget |
| Learning rates | 0.0001, 0.0003, 0.001 | Allows each model state to learn at a suitable rate |
| Optimizer | Fresh optimizer for every probe, using notebook 29's configuration | Removes inherited optimizer history from the comparison |
| Repetitions | Five at the reference point, two elsewhere, per training seed | The adapter study spends more repetitions on the reference point, because every other number is compared against it |

EuroSAT RGB images and Tiny ImageNet images are both 64 by 64 pixels. Check whether the resize step does anything before writing a resize path.

Select the learning rate separately for each model state, seed, and repetition using final validation accuracy. Break ties with a fixed rule, such as choosing the lower rate. Evaluate the selected run on the test split only after selection. Do not choose rates or training budgets from test results. This corrects notebook 29's practice of selecting natural-probe rates using test accuracy, so earlier natural-probe numbers are affected by that practice and should not be compared directly with numbers selected the new way.

### Pilot before the full experiment

Prepare the data and verify split separation, image shapes, ten-class labels, and checkpoint provenance on CPU. Then run a seed-0 pilot at both model states, with one paired repetition and all three learning rates: six short GPU runs.

Use training loss and validation curves to check whether the budget permits learning. If both models remain near chance, this is not evidence of equal plasticity. If both saturate very early, final accuracy may conceal differences in learning speed. In either case, revise the budget using pilot validation results, record the reason, and freeze the final settings before the main experiment. Treat pilot runs as calibration, not as independent confirmation.

### Measurements and interpretation

Primary measurement: test accuracy after the fixed training budget, at the validation-selected learning rate.

For each seed and repetition, calculate:

```text
accuracy change = accuracy at the later state - accuracy at the reference state
```

A negative value means poorer learning on EuroSAT after the longer history. A positive value means better learning. Neither result alone identifies a mechanism.

Also save training loss and validation accuracy at fixed step intervals. These curves show whether one model learns more slowly even when final accuracy is similar. Any learning-speed summary must use the same evaluation intervals and training budget at both states.

Report each seed's paired changes, their mean, and variation across repetitions. Then report the mean and sample standard deviation across the three training seeds. Estimate measurement noise from repeated probes of the unchanged reference model. Use paired uncertainty estimates that retain the repetition matching; do not count all repetitions as independent training seeds. With only three training seeds, broad population claims remain provisional even if a bootstrap interval is narrow.

Use a deterioration verdict only when the paired change is negative, its uncertainty interval excludes zero, and its magnitude exceeds the prespecified noise floor. Write down the exact noise-floor and interval calculations before the main run. `uncle/probe.noise_floor` holds the formula. Otherwise report no clear detected deterioration, not proof of unchanged plasticity.

| Possible observation | Supported interpretation |
| --- | --- |
| No clear EuroSAT decline | No detected loss on this external task, at these settings |
| EuroSAT declines while Tiny ImageNet probes improve | Learning history has different effects on the two probe domains; the Tiny ImageNet gain does not generalize to EuroSAT |
| Both model states fail to learn EuroSAT | Probe budget or setup is inadequate for answering the question |
| EuroSAT improves | Prior training also helps this external task; transfer remains a possible explanation |

Against an untrained reference, the last row is the expected result and it answers nothing. That is the fault described at the top of this page.

Keep the existing Tiny ImageNet findings alongside these results. They used a different rate-selection rule, so do not combine the scores into one averaged endpoint. A direct numerical comparison across datasets would require the same validation-based protocol on both.

### A cheaper measurement that should come first

The adapter study records adapter geometry after every task, not only at its six probe points. Rotation angles and condition numbers come from the weights alone, so there is no data pass and almost no cost. That record is what lets the study say what the adapter looked like at the moment learning changed.

We log nothing equivalent. The following all come from `weights_from_code` and `raw_from_code` with no forward pass, and can be computed on checkpoints we already hold:

| Quantity | Why it is worth logging |
| --- | --- |
| Relative weight change for each learned task, against its own earlier weights | The adapter study's common size measure. Says how far the generated network has drifted |
| Condition number of each generator head's output layer | The nearest thing we have to the study's LoRA conditioning diagnostic |
| Task code norms, and the smallest angle between any two learned codes | Codes crowding together is a mechanism this architecture can have and theirs cannot |
| Fraction of silent channels | Measured once on 21 October. Make it routine |

Feature effective rank needs a forward pass, so it belongs at probe points only. `uncle/probe.effective_rank` already computes it.

This costs no GPU time and no retraining. It also answers a prior question: if the generator barely moves across the stream, then no plasticity experiment on this stream can show anything, and the retraining job can be planned accordingly.

### Compute, saving, and stopping

The full primary grid, as originally written, is 3 training seeds x 2 model states x 5 repetitions x 3 learning rates = 90 probe runs, each initially 250 steps. It needs GPU probe training but no retraining of the eight-task stream. A corrected version that uses t = 1 as the reference needs the stream trained again first, so its cost is higher and should be estimated together with the retraining job.

Measure pilot runtime and peak GPU memory before estimating the full cost; do not reuse notebook 29's runtime estimate without checking.

Run one training seed per session if needed. Save each completed probe, selected-rate metadata, and learning curves to R2, and verify the saved files before releasing the GPU. Record source checkpoint hashes, code version, dataset manifest hash, preprocessing, optimizer settings, rate, seed, repetition, and model state in the run identity. Skip only completed runs with the same identity. Restart an interrupted probe unless optimizer state, progress, and random-number state have also been saved; completed-run reuse is not mid-probe resume.

### Scope and next decision

This plan tests one external task, one training order, three seeds, and two measurement points. It does not test re-anchoring, identify a mechanism, or show that unlearning restores plasticity.

Re-anchoring cannot be ported to this architecture at all. In the adapter study it means folding the adapter into the frozen backbone and restarting the adapter at identity, which leaves the model unchanged but resets its coordinates. We have no frozen backbone to fold into, because the generator is the model. Any future record should say this rather than leave it implied.

An unlearning comparison would require matched branches from the same trained checkpoint, one unchanged and one unlearned, followed by identical probes. Call any gain improved learnability; use restored plasticity only if prior deterioration has been established.

Implementation should reuse `uncle/probe.py` and notebook 29's checkpoint and saving conventions, add the EuroSAT loader and validation-based rate selection, and keep results under a separate experiment identity. No training is started by this plan.
