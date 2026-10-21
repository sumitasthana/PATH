# Experiment trajectory: 20 and 21 October 2026

[Back to the experiment index](Experiment-trajectory)

The plasticity probe was run to completion on three seeds. This page records the probe's final form, the result, one defect found during the run, and what follows.

## On this page

- [Result: no plasticity loss, a small gain instead](#result-no-plasticity-loss-a-small-gain-instead)
- [Second result: training the task code buys nothing](#second-result-training-the-task-code-buys-nothing)
- [Third result: the task code alone holds nothing, on three seeds](#third-result-the-task-code-alone-holds-nothing-on-three-seeds)
- [Readings that produced no finding](#readings-that-produced-no-finding)
- [Changes made to the probe before this run](#changes-made-to-the-probe-before-this-run)
- [Defect: the resume check skipped only one of three arms](#defect-the-resume-check-skipped-only-one-of-three-arms)
- [What this settles and what it does not](#what-this-settles-and-what-it-does-not)
- [Limits](#limits)
- [Next](#next)

## Result: no plasticity loss, a small gain instead

Plasticity means the ability to learn a new task. Loss of plasticity means that ability gets worse as more tasks are learned. It is a separate question from forgetting, which is about old tasks getting worse.

The generator does not lose that ability. After learning the eight tasks of order 01 it learns an unseen task slightly better than a generator that has learned nothing. Two measurements agree, using different kinds of evidence.

| Measurement | What it measures | Mean change | Spread across seeds |
| --- | --- | ---: | ---: |
| Useful learning | Test accuracy on tasks 2, 4 and 6, averaged | +4.74 points | 1.68 |
| Capacity to change | Final training loss on randomly labelled images | -0.70 | 0.06 |

Higher accuracy is better and lower loss is better, so both changes are in the direction of better learning.

Per seed, against each seed's own measured noise floor. The floor comes from the spread of five repetitions of the probe on the same unchanged model, so it is this measurement's own wobble rather than a number borrowed from the replay work.

| Measurement | Seed | Before | After 8 tasks | Change | Noise floor | Exceeds the floor |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Useful learning | 0 | 51.87 | 58.23 | +6.37 | 5.53 | Yes |
| Useful learning | 1 | 53.09 | 56.10 | +3.01 | 3.23 | No |
| Useful learning | 2 | 52.51 | 57.37 | +4.86 | 2.28 | Yes |
| Capacity to change | 0 | 1.37 | 0.67 | -0.70 | 0.29 | Yes |
| Capacity to change | 1 | 1.54 | 0.78 | -0.76 | 0.42 | Yes |
| Capacity to change | 2 | 1.31 | 0.67 | -0.64 | 0.18 | Yes |

Five of six readings pass the decision rule, which was fixed before the runs. Seed 1's accuracy reading falls just inside its floor, 3.01 against 3.23, and is recorded as no finding rather than as a gain. It points the same way as the other two.

The capacity measurement is the stronger evidence here. All three seeds pass, and the three values differ from each other by 0.06. The random-label probe also worked this time: the full arm ends at 1.37 and 0.67, well below the chance loss of ln(10) = 2.303, so nothing is void. The 17 October run of this same probe was void on this endpoint because its budget was too small.

This supersedes the 17 October reading of +0.8 points from notebook 26. That used one probe task, one learning rate, and a borrowed noise floor. The present numbers use three probe tasks, a three-rate grid with the best kept, and a floor measured per seed.

## Second result: training the task code buys nothing

Each task has its own task code: 32 numbers that select which classifier the generator produces. Normally the code and the generator train together. A new arm holds the code at its random starting value and trains only the generator.

| Seed | Code trained | Code frozen | Difference | Noise floor |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 51.87 | 53.35 | -1.48 | 5.53 |
| 1 | 53.09 | 52.08 | +1.01 | 3.23 |
| 2 | 52.51 | 51.24 | +1.27 | 2.28 |

Test accuracy in percent, before any task. The difference changes sign between seeds and is far smaller than the floor in every case. The same holds after eight tasks, and on the capacity measurement, where the differences are 0.16, 0.07 and -0.19 against floors of 0.29, 0.42 and 0.18.

No measurable effect in either direction. In this setup the generator reaches the same result whether its task code is trained or left random.

## Third result: the task code alone holds nothing, on three seeds

A frozen generator with a fresh task code reads 10.84, 10.25 and 10.91 percent before any task, and 10.07, 10.00 and 10.60 after eight tasks. Chance accuracy for ten classes is 10 percent. This was known from one seed and now holds on three, at both measurement points.

The generator's own contribution, measured as the full arm minus the frozen-generator arm, is 41.03, 42.84 and 41.60 points before any task, rising to 48.17, 46.10 and 46.77 after eight tasks.

## Readings that produced no finding

Feature effective rank says how many independent directions the features really use. It reads 591.0, 616.7 and 628.7 before any task, and 606.3, 612.1 and 604.9 after eight tasks, with a spread of 38 to 70 within each reading. The spread is larger than any change, so there is no finding. The 17 October values of 467.9 and 396.6 are superseded: they averaged over every learning rate rather than only the selected runs, which is now corrected.

The relative contribution ratio R(8) is 1.174, 1.076 and 1.124. It is computable but carries no information beyond the accuracy result, because its control arm sits at chance at every point, so the ratio tracks accuracy. It is reported as a number and is not independent evidence.

## Changes made to the probe before this run

Three changes were committed between 20 and 21 October. None of them changes the protocol the probe follows.

- **A third arm**, committed as `c9f87d5`. The probe already ran a full arm and a frozen-generator arm. The new arm freezes the task code instead and trains only the generator, which answers the second result above. It uses the same grid and budget and costs half again as much time.
- **All seeds in one pass, with resume**, committed as `b2136a3`. Three seeds with three arms is about 10 hours, longer than one session on the available hardware. The notebook now works through a list of seeds and appends each finished probe run to a file in R2. Starting the run again reads that file, skips what is done, and continues.
- **One spelling per arm**, committed as `2f13b55`. See the defect below.

The random-label budget was already corrected on 17 October to 2,000 images for about 19 passes, matching the adapter study's own correction. This run is the first to use it, and it is why the capacity endpoint is valid for the first time.

## Defect: the resume check skipped only one of three arms

The resume check compared the arm's name as written in the notebook's settings against the name written into each saved result. Two arms were spelled with a hyphen in one place and an underscore in the other, so they never matched and were run again in every session. Only the arm whose name was spelled identically in both places was skipped correctly.

The finished file held 1,032 lines for 756 distinct runs. The 276 repeats are about 3.6 hours of wasted computation. They fall entirely in the two affected arms, so the best-of-grid selection would have chosen from more samples for those arms than for the first, which biases them. The main accuracy and capacity verdicts use the first arm only and were not affected.

The fix uses one spelling per arm. The loading step also drops repeats already saved, so the existing file stayed usable and no run had to be repeated. A test compares the two sources of the name and fails if they differ again.

The repeats are themselves an accidental measurement: 186 identical configurations run twice or three times on the same model, of which 174 disagree, by up to 24.4 points. That is larger than the 8 to 11 points previously recorded for session-to-session variation. It is accidental evidence rather than a designed measurement, and it has not been analysed.

## What this settles and what it does not

The paper states that unlearning restores the hypernetwork's plasticity. These measurements say there is no plasticity loss for unlearning to restore. The separate branch-minus-control comparison recorded on 10 October gave +5.40 and -0.20 points across three seeds, both at or inside the noise floor, so unlearning showed no effect on later learning either. The claim remains unsupported in this framework.

The planned beta-zero control arm is not needed. It would remove a handicap that only the eight-task model carries, and that model already scored higher while carrying it. It stays available if a later seed reverses the sign.

The adapter study reaches the same conclusion on a different architecture over twenty tasks: no plasticity loss at the settings that give the best accuracy. The two projects now agree.

This result says nothing about erasure, recovery or privacy. None of those has been measured.

## Limits

Three seeds, two probe points, one learning order, four probe tasks from one dataset family.

The spread between seeds on the accuracy measurement is 1.68, about a third of the mean change. Three seeds give an interval but not a narrow one. These numbers are provisional and the seed count should be stated wherever they appear.

All three seeds use order 01, so "after eight tasks" and "after these particular eight tasks" cannot be separated. Orders 02 and 03 exist in the study plan and would separate them.

The probe runs before the stream and after all eight tasks, with nothing in between, because no mid-stream checkpoints were kept. A change that appeared and then reversed would not be visible.

Probe tasks 2, 4, 6 and 8 are Tiny ImageNet groups, like the tasks the model already learned. A generator suited to this one dataset family would appear healthy here and could still fail on a different kind of image. The adapter study avoids this with unrelated datasets; three of its four probe tasks cannot be used here, because the generator's output is fixed at ten classes.

The measurements show the direction of change. They do not explain why learning eight tasks makes the generator better at a ninth. Transfer between similar classes is one possible cause and has not been separated from the rest.

## Next

Experiment A is closed. Plasticity is not where the difficulty lies, on this probe, at this sample size.

Two measurements from the local guide remain unrun and neither follows from this result: the mid-stream probe points, and orders 02 and 03. Both need the stream trained again with probing inline, so they are one job rather than two. Running them would also settle the task-and-position confound that the 17 October record left open for the in-stream curve.

The largest open question in the study is unchanged and untouched by this work: whether a removed task can be recovered under a fixed training budget with suitable controls. It has still not been specified.

A full write-up of these findings is at `ops-docs/guides/plasticity-findings-2026-10-21.html`, which stays local because `ops-docs` is not committed.
