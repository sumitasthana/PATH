# Experiment trajectory: 20 and 21 October 2026

[Back to the experiment index](Experiment-trajectory)

The plasticity probe finished. It ran on three seeds. This page gives the result, one defect found during the run, and what comes next.

## On this page

- [What we asked](#what-we-asked)
- [Result: the generator does not get worse at learning](#result-the-generator-does-not-get-worse-at-learning)
- [Second result: training the task code buys nothing](#second-result-training-the-task-code-buys-nothing)
- [Third result: the task code alone holds nothing](#third-result-the-task-code-alone-holds-nothing)
- [Two readings that showed nothing](#two-readings-that-showed-nothing)
- [Two observations from comparing the three seeds](#two-observations-from-comparing-the-three-seeds)
- [What we changed before this run](#what-we-changed-before-this-run)
- [Defect: the probe repeated work it had already done](#defect-the-probe-repeated-work-it-had-already-done)
- [What this settles](#what-this-settles)
- [Limits](#limits)
- [Next](#next)

## What we asked

Plasticity means the ability to learn a new task. Loss of plasticity means that ability gets worse as the model learns more tasks.

This is not the same as forgetting. Forgetting means old tasks get worse. Plasticity is about new tasks. A model can forget everything and still learn new things well.

We asked one question. After the generator learns eight tasks, is it worse at learning a ninth?

The word **generator** means the hypernetwork. It is the part that produces the weights of a classifier. It is the only part that carries anything from one task to the next. So it is the only part whose plasticity can change.

## Result: the generator does not get worse at learning

It gets slightly better.

We measured this in two ways. Both say the same thing.

The first way is **useful learning**. We teach the model three tasks it has never seen. We record the test accuracy. Higher is better.

The second way is **capacity to change**. We give images random wrong labels. A dog might be labelled "truck". No pattern is left to find. The only way to fit the data is to memorise each image one by one. We record the final training loss. Lower is better.

| Measurement | Mean change over 3 seeds | Spread between seeds |
| --- | ---: | ---: |
| Useful learning | +4.74 points | 1.68 |
| Capacity to change | -0.70 | 0.06 |

Accuracy went up. Loss went down. Both mean better learning.

### The numbers for each seed

A **seed** is one random starting point for training. We ran three.

A change only counts when it is bigger than the measurement's own wobble. We call that wobble the **noise floor**. We measure it by running the probe five times on the same unchanged model and looking at how much those five differ.

| Measurement | Seed | Before | After 8 tasks | Change | Noise floor | Bigger than the floor |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Useful learning | 0 | 51.87 | 58.23 | +6.37 | 5.53 | Yes |
| Useful learning | 1 | 53.09 | 56.10 | +3.01 | 3.23 | No |
| Useful learning | 2 | 52.51 | 57.37 | +4.86 | 2.28 | Yes |
| Capacity to change | 0 | 1.37 | 0.67 | -0.70 | 0.29 | Yes |
| Capacity to change | 1 | 1.54 | 0.78 | -0.76 | 0.42 | Yes |
| Capacity to change | 2 | 1.31 | 0.67 | -0.64 | 0.18 | Yes |

Five of the six readings pass. Seed 1's accuracy reading is 3.01 against a floor of 3.23. That is just inside the floor. So we record it as no finding. It still points the same way as the other two.

The second measurement is the stronger one. All three seeds pass. And the three values are 0.70, 0.76 and 0.64. They differ from each other by only 0.06. That is very consistent.

This measurement also worked for the first time. On 17 October it was void. The model never memorised anything, so the number measured the budget and not the model. We then made the probe use fewer images and more passes over them. This run shows it worked: the loss ends at 1.37 and 0.67, well below the chance level of 2.303.

These numbers replace the 17 October reading of +0.8 points. That reading used one new task, one learning rate, and a noise floor borrowed from other work. This run uses three new tasks, three learning rates with the best kept, and a floor measured for each seed.

## Second result: training the task code buys nothing

Each task has its own **task code**: 32 numbers that tell the generator which classifier to make. Normally these 32 numbers are trained together with the generator.

We added a new arm that leaves them at their random starting value. An **arm** means one setting of which parts are allowed to train. Only the generator trains in this arm.

| Seed | Code trained | Code left random | Difference | Noise floor |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 51.87 | 53.35 | -1.48 | 5.53 |
| 1 | 53.09 | 52.08 | +1.01 | 3.23 |
| 2 | 52.51 | 51.24 | +1.27 | 2.28 |

These are test accuracies in percent, before any task is learned.

The difference is negative for one seed and positive for two. So it has no consistent direction. It is also much smaller than the noise floor every time.

The same holds after eight tasks. It also holds for the second measurement, where the differences are 0.16, 0.07 and -0.19 against floors of 0.29, 0.42 and 0.18.

So training the 32 numbers makes no measurable difference here. The generator reaches the same result either way.

## Third result: the task code alone holds nothing

Here we do the opposite. We freeze the generator and train only the 32 numbers.

The result is 10.84, 10.25 and 10.91 percent before any task. It is 10.07, 10.00 and 10.60 after eight tasks. Chance accuracy for ten classes is 10 percent.

So 32 numbers cannot hold a task. We knew this from one seed. Now it holds on three seeds, at both measurement points.

This also tells us how much the generator itself contributes. We subtract the frozen-generator result from the full result. The answer is 41.03, 42.84 and 41.60 points before any task. It rises to 48.17, 46.10 and 46.77 after eight tasks.

## Two readings that showed nothing

**Feature effective rank** says how many independent directions the model's internal features use. A fall is a known warning sign of plasticity loss.

It reads 591.0, 616.7 and 628.7 before any task. It reads 606.3, 612.1 and 604.9 after eight tasks. But the spread inside each reading is 38 to 70. That is larger than any of the changes. So there is no finding.

This replaces the 17 October values of 467.9 and 396.6. Those averaged over every learning rate instead of only the chosen ones, which we have now fixed.

**The relative contribution ratio** is 1.174, 1.076 and 1.124. We can now compute it. But it tells us nothing new. Its control arm sits at chance at every point, so the ratio just follows the accuracy. We report it as a number and do not treat it as separate evidence.

## What we changed before this run

Three changes went in between 20 and 21 October. None of them changes the method the probe follows.

**A third arm**, commit `c9f87d5`. The probe already had a full arm and a frozen-generator arm. The new arm freezes the task code instead. This is what gives the second result above. It uses the same grid and the same budget. It costs half again as much time.

**All seeds in one pass, with resume**, commit `b2136a3`. Three seeds with three arms takes about 10 hours. That is longer than one session on our hardware. The notebook now works through a list of seeds. It writes each finished run to a file in R2. When you start it again, it reads that file, skips what is done, and continues.

**One spelling per arm**, commit `2f13b55`. See the defect below.

The random-label budget was already fixed on 17 October. It now uses 2,000 images and about 19 passes over them. This is the first run to use it.

## Defect: the probe repeated work it had already done

The resume check compares two names. One is the arm's name in the notebook settings. The other is the name saved inside each result.

Two arms were spelled with a hyphen in one place and an underscore in the other. So those two names never matched. Those two arms were never skipped. They ran again in every session. Only the third arm, spelled the same in both places, was skipped correctly.

The finished file had 1,032 lines for 756 real runs. The 276 extra runs are about 3.6 hours of wasted computation.

The extra runs all fall in the two affected arms. This matters, because the probe keeps the best result out of several learning rates. Two arms would have had more results to choose from than the third. That makes those two look better than they are. The main result uses the third arm only, so it was not affected.

The fix uses one spelling per arm. The loading step also drops the repeats, so we did not have to run anything again. A test now compares the two names and fails if they differ.

The repeats are themselves an accidental measurement. They are 186 identical settings run two or three times on the same model. Of those, 174 disagree. The largest disagreement is 24.4 points. That is bigger than the 8 to 11 points we recorded before for session-to-session variation. It is accidental evidence, not a planned measurement, and we have not analysed it.

## Two observations from comparing the three seeds

These came from putting the plasticity numbers beside the paired-study numbers from 10 October. Both rest on three data points. They are observations, not claims. Each says what would be needed to turn it into a claim.

### One task is far noisier than three tasks averaged

Task 15 after the eight tasks reads 50.8, 44.0 and 34.4 percent across the three seeds. That is a range of 16.4 points.

The plasticity probe averages three tasks. It reads 58.23, 56.10 and 57.37. That is a range of 2.13 points.

The same gap appears before any task is learned. Task 15 alone gives 44.5, 43.2 and 39.0, a range of 5.5. The three averaged tasks give 51.87, 53.09 and 52.51, a range of 1.22.

So most of the disagreement between seeds came from using one task, not from the seeds. This explains why the 17 October reading had a spread of 5.5 and this run has 1.68. It also means a single-task measurement should not be trusted at this sample size.

**To make this a claim:** repeat the probe with a different set of three tasks and check that the range stays small.

### The seed with the weakest control gained the most from unlearning

The task-15 controls are 50.8, 44.0 and 34.4. The gains from unlearning task 3 are 0.0, +3.2 and +13.0. The order matches exactly. The lower the control, the larger the gain.

This matters because the three-seed mean for unlearning task 3 is +5.40, and seed 2 supplies most of it.

Seed 2's control of 34.4 is also below its own fresh-model baseline of 39.0. The other two seeds sit above theirs. So the control looks low rather than the branch looking high.

Seed 2 is not unusual on the probe. Its change there is +4.86, between seed 0's +6.37 and seed 1's +3.01. Its memorising change is -0.64, close to the other two. So "seed 2 is a weak seed" is not supported by the evidence. It gave one unusual reading on one single-task measurement, which the first observation explains.

**To make this a claim:** rerun the seed-2 control. If it returns near 44, the pattern is a low reading rather than an effect of unlearning.

A fuller version of this comparison, with the full sequence and all four measurements per seed, is at `ops-docs/guides/seeds-and-sequence.html`. It stays local, because `ops-docs` is not committed.

## What this settles

The paper says that unlearning restores the hypernetwork's plasticity. Our measurements say there is no plasticity loss to restore.

We also measured unlearning directly, on 10 October. We compared learning a new task with and without unlearning first. The difference was +5.40 and -0.20 points across three seeds. Both are at or inside the noise floor. So unlearning showed no effect on later learning either.

The paper's claim is not supported in this framework.

We also planned a control arm with the protection term switched off. The **protection term** is the part of the loss that holds the generator near its current weights, so old tasks are not damaged. That arm only handicaps the eight-task model, because a fresh model has nothing to protect. The eight-task model already scored higher while carrying that handicap. Removing it would widen the gap, not explain it. So that arm is no longer needed. We keep it available if a later seed reverses the sign.

A separate study reaches the same conclusion on a different architecture. It trains small parts called adapters on a frozen vision transformer, over twenty tasks. It also finds no plasticity loss at the settings that give the best accuracy. The two projects agree.

This result says nothing about erasure, recovery or privacy. We have not measured any of those.

## Limits

**Three seeds is thin.** The spread between seeds on the accuracy measurement is 1.68. The mean change is 4.74. So the spread is about a third of the effect. Three seeds give us a range, but not a narrow one. These numbers are provisional. Always say how many seeds they rest on.

**One task order.** All three seeds learn the tasks in the same order. So we cannot separate "after eight tasks" from "after these particular eight tasks". Orders 02 and 03 are in the study plan and would separate them.

**Two measurement points only.** We measure before the stream and after all eight tasks. Nothing in between, because we kept no mid-stream checkpoints. A change that appeared and then went away would be invisible to us.

**The new tasks come from the same dataset.** Tasks 2, 4, 6 and 8 are Tiny ImageNet groups, like the tasks the model already learned. A generator that suits this one dataset would look healthy here and could still fail on a different kind of image. The adapter study avoids this by using unrelated datasets. We cannot copy that: three of its four probe tasks need more than ten classes, and our generator produces exactly ten.

**We do not know why it got better.** The measurements show the direction. They do not give a cause. Transfer between similar classes is one possible reason. We have not separated it from the others.

## Which neurons still fire after unlearning

Notebook 27 now reads every ReLU in the generated ResNet50, before the unlearning request and after it, on the same images. It reads a kept task through the same hooks as a control. The user supplied one run: order 01, seed 0, task 3 removed.

The table gives each ResNet stage's mean response after the request, as a share of its response before.

| ResNet stage | Forgotten task 3 | Kept task 0 |
| --- | ---: | ---: |
| stem | 29% | 100% |
| layer1 | 14% | 99% |
| layer2 | 13% | 98% |
| layer3 | 16% | 97% |
| layer4 | 19% | 97% |

A channel counts as silent when its strongest response falls below 1 percent of its own earlier peak. For task 3, 58.9 percent of the 22,720 channels went silent. For task 0, 0.0 percent did.

Three readings.

**The damage is specific.** The kept task lost nothing measurable. No channel went silent and every stage holds within 3 percent of where it was. This rules out damage to the generator itself, at least for this request.

**There is no single point of failure.** Every stage falls by a similar amount. The whole generated network is turned down at once. One layer blocking the signal would have shown as a single stage collapsing while earlier ones stayed normal.

**It is not fully dead.** About 41 percent of channels still respond, at roughly a sixth of their earlier strength. That fits the weights collapsing toward zero rather than being randomised, which is what the objective's averaging does.

The stem falls least, keeping 29 percent against 13 to 19 elsewhere. One candidate cause is already in the code comments: the unlearning objective compares raw generator outputs, before the per-layer scale factors, while the protection term compares scaled weights, and gamma does not remove that difference in layer weighting. This is a candidate, not a tested explanation.

Limits. One seed, one forget task, one control task, one batch of 256 images. The reading says what the network does after the request. It does not say whether the task can be recovered.

## Next

This experiment is closed. Plasticity is not where our problem lies, on this probe, at this sample size.

Two measurements are still unrun. Neither follows from this result.

The first is the mid-stream probe points. The second is orders 02 and 03. Both need the stream trained again with the probe called during it. So they are one job, not two. Doing them would also settle the task-and-position question that the 17 October record left open.

The largest open question in the study is untouched by this work. It is whether a removed task can be brought back under a fixed training budget with proper controls. It is still not specified.

A longer write-up is at `ops-docs/guides/plasticity-findings-2026-10-21.html`. It stays local, because `ops-docs` is not committed.
