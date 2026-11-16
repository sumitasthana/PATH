# Experiment trajectory: 16 November 2026

Notebook 48, `48_what_the_indicators_read.ipynb`, ran four tests on the saved
revised-study models. The tests ask what the published rank numbers are
measuring, and whether the two endpoints at boundary 0 and boundary 9 were
hiding the middle of the stream. Three of the four results change what the
paper can claim.

## Protocol check

Before any test, the notebook recomputed effective rank through the fixed
reference code on ten saved checkpoints and compared each value with the
number the original run wrote to its indicator log.

| Run | Boundary | Notebook | Record | Gap |
| --- | --- | --- | --- | --- |
| order_A seed 0 | 4 | 3.9835 | 3.9828 | 0.0007 |
| order_A seed 0 | 9 | 5.0206 | 5.0134 | 0.0072 |
| order_A seed 1 | 4 | 2.0322 | 2.0320 | 0.0002 |
| order_A seed 1 | 9 | 2.1308 | 2.1313 | 0.0005 |
| order_A seed 2 | 4 | 1.6619 | 1.6616 | 0.0003 |
| order_A seed 2 | 9 | 3.2487 | 3.2495 | 0.0007 |
| order_A seed 3 | 4 | 1.4035 | 1.4036 | 0.0001 |
| order_A seed 3 | 9 | 1.3197 | 1.3195 | 0.0001 |
| order_A seed 4 | 4 | 2.2002 | 2.1999 | 0.0003 |
| order_A seed 4 | 9 | 2.2231 | 2.2240 | 0.0009 |

Largest gap 0.0072. The notebook is reading the same models through the same
protocol the study used, so the four tests below are comparable with the
published numbers.

## Test 1: the reading depends on which task code is used

Each saved model was read twice. Once through the fixed reference code, which
is the protocol the study published. Once through every task code the model
had actually trained, with the results averaged.

| Boundary | Reading route | Effective rank | srank | Readings |
| --- | --- | --- | --- | --- |
| 4 | reference code | 1.835 | 69.7 | 15 |
| 4 | trained codes, mean | 2.492 | 112.6 | 60 |
| 9 | reference code | 2.095 | 81.1 | 15 |
| 9 | trained codes, mean | 2.792 | 119.7 | 135 |

A trained code reads 1.4 times higher on effective rank and 1.5 times higher
on srank than the reference code, on the same model and the same images. The
published indicator value is therefore a property of the model and the code
together, not of the model alone. Any paper reporting a rank level for a
hypernetwork has to state which code the weights were generated from.

## Test 1b: nothing moves between boundary 4 and boundary 9

Only codes present at both boundaries were compared, so each code is compared
with itself rather than with a different code.

| Reading route | Boundary 4 | Boundary 9 | Rose |
| --- | --- | --- | --- |
| reference code | 1.835 | 2.095 | 6 of 15 |
| trained task codes | 2.492 | 2.545 | 33 of 60 |

The change over the last five tasks is small, and its direction is close to a
coin toss on both routes. Combined with the 15 November finding that the drop
is concentrated at the first task, the picture is that effective rank moves
once, when the generator leaves its random start, and then holds.

## Test 2: the average feature direction dominates the uncentred reading

Each feature matrix was decomposed twice, once as the study computes it and
once after subtracting the average row. The average row is the feature vector
that every image shares. The column `mean share` is the fraction of the
matrix explained by that one shared row, from 0 to 1.

| Code | Images | Effective rank, as published | Effective rank, centred | srank, as published | srank, centred | Mean share |
| --- | --- | --- | --- | --- | --- | --- |
| reference | in-stream | 1.965 | 9.2 | 75.4 | 149.4 | 0.981 |
| reference | probe | 1.924 | 8.2 | 71.7 | 141.8 | 0.977 |
| trained | in-stream | 2.699 | 15.5 | 117.5 | 187.0 | 0.982 |
| trained | probe | 2.605 | 13.7 | 113.1 | 180.8 | 0.979 |

About 98 percent of the feature matrix is the single direction every image
shares. Removing it raises effective rank by a factor of about 4.7 and
roughly doubles srank. The published effective rank near 2 is largely
reporting that all images produce nearly the same features, not that the
model uses only two directions to tell them apart.

This also corrects the 15 November chat estimate. A simulation there
suggested a pure mean offset would leave srank near 175 while the study
reported about 66, and so that centring would not explain the published
numbers. The measured result goes the other way: the mean share is 0.98 and
centring changes both measures substantially. The simulation used a single
offset with otherwise isotropic noise, which is not the shape of these
features.

Note that the uncentred form is what Kumar et al. (2021) and the plasticity
literature compute, so the published numbers follow the standard definition.
The finding is about what that standard definition reads in this model, not
an error in applying it.

## Test 3: the probe gain is finished by boundary 4

The probe was run at boundary 4 for all five order A seeds, using the same
settings as the published boundary 0 and boundary 9 runs: three learning
rates with the best chosen per repetition on a validation split, 250 steps,
eleven repetitions, 4,800 training images, the `full` arm. Chance is 5
percent.

| Run | t = 0 | t = 4 | t = 9 | 0 to 4 | 4 to 9 |
| --- | --- | --- | --- | --- | --- |
| order_A seed 0 | 26.13 | 43.56 | 43.72 | +17.44 | +0.15 |
| order_A seed 1 | 27.25 | 43.58 | 43.65 | +16.33 | +0.07 |
| order_A seed 2 | 26.06 | 44.40 | 43.75 | +18.34 | -0.65 |
| order_A seed 3 | 26.55 | 42.94 | 42.76 | +16.39 | -0.17 |
| order_A seed 4 | 26.78 | 45.78 | 43.90 | +19.00 | -1.88 |
| mean | | | | +17.50 | -0.50 |

The whole measured gain happens in the first four tasks. The change over the
last five tasks is -0.50 points, with a standard deviation across the five
runs of 0.83 and a standard error of 0.37, which is 1.3 standard errors from
zero on four degrees of freedom. It is not distinguishable from no change.
Accuracy fell in 3 of 5 runs.

Both sides of the study are therefore flat across the middle and late stream.
The indicators move once at task 1 and hold. The probe moves once by task 4
and holds. The disagreement reported on 14 November between a falling
indicator and a rising probe was measured between boundary 0 and boundary 9
only, and both endpoints of that span are dominated by the same event, which
is the exit from random initialisation.

## Test 4: the frozen-generator arm cannot measure anything

| Run | full, t = 0 | full, t = 9 | frozen, t = 0 | frozen, t = 9 |
| --- | --- | --- | --- | --- |
| order_A seed 0 | 26.13 | 43.72 | 5.04 | 4.72 |
| order_A seed 1 | 27.25 | 43.65 | 5.26 | 5.84 |
| order_A seed 2 | 26.06 | 43.75 | 4.52 | 4.14 |
| order_A seed 3 | 26.55 | 42.76 | 4.58 | 5.06 |
| order_A seed 4 | 26.78 | 43.90 | 5.02 | 5.18 |

Chance is 5 percent. With the generator frozen and only the task code
trained, every reading sits between 4.14 and 5.84 percent at both boundaries.
The frozen arm is at chance. Its change over the stream is +0.10 points
against the full arm's +17.00.

Two consequences. First, the paper must report the full arm and describe it
correctly, because the frozen arm has no measurable signal to report. Second,
this is a result in its own right: a 32-number task code, on its own, cannot
steer this generator to a held-out group of twenty classes. Whatever lets the
model learn a new task lives in the generator weights, not in the code space.

## Interpretation and next decision

The claim the draft was built on does not survive. The draft said the rank
indicators fall while directly measured learning ability rises, and that the
two therefore disagree. Both of those movements are now located at the same
place, the first few tasks after random initialisation, and neither measure
moves after that. With nine tasks in this stream there is no loss of
plasticity for an indicator to detect, so the study cannot test whether the
indicators detect one.

What the four tests do establish is a set of measurement facts about rank
indicators in a hypernetwork:

1. The reading depends on the task code it is taken through, by a factor of
   about 1.4 on these models.
2. The uncentred reading is about 98 percent the shared mean feature
   direction, and centring moves effective rank by a factor near 4.7.
3. The large fall over the stream is the exit from random initialisation,
   not accumulation across tasks. Measured from boundary 1 or boundary 4 the
   indicators are flat.
4. The task code alone is not a usable handle on a new task, so a
   frozen-generator plasticity probe reads chance.

Two honest directions follow, and the choice has not been made.

**A measurement paper.** The contribution is the four facts above, which
apply to anyone reading rank measures off a hypernetwork or off any model
with a strong shared feature direction. This is supported by the evidence
already collected and needs no new training.

**A stream that actually loses plasticity.** Nine tasks of twenty Tiny
ImageNet classes with this learning rate does not degrade the model. A harder
or longer stream would be needed before the original question can be asked at
all. This needs new runs and a new design decision about what makes the
stream hard enough.

## Evidence

Reads from Test 1 and Test 2 are in
`uncle/indicator_protocol/plasticity_revised_v1/code_sweep_and_centering.jsonl`.
Probe runs from Test 3 and Test 4 are in
`uncle/matched_probe/plasticity_revised_v1/revised_20class_train4800_extended.jsonl`.
The published probe file was read and not written. Tests 3 and 4 cover order
A only, five seeds. Orders B and C were not run at boundary 4 and were not run
on the frozen arm.

## Independent recomputation of the four tests

All four tables above were recomputed from the saved records in a separate
session, using the same learning-rate selection rule the study uses: the best
of the three rates chosen per repetition on that repetition's validation
accuracy, ties going to the lower rate. Every value reproduced. The Test 3
per-seed rows, the mean 0 to 4 gain of +17.50, the 4 to 9 change of -0.50 with
standard deviation 0.84 and standard error 0.37, and the frozen-arm range of
4.14 to 5.84 percent all match to the second decimal.

One alternative explanation for Test 3 was checked and does not hold. If 44
percent were the most this probe can read, flatness after boundary 4 would be
the measurement running out of room rather than the model holding still. At the
selected rate the probe reaches 78 to 87 percent on its own training images
while scoring about 44 percent on held-out ones, at both boundaries. There is
headroom in the measure, so the flatness is a property of the model.

Two wording changes follow. First, the reason the boundary 0 to 9 comparison
proves nothing is not that both measures move at the same time. At that one
point they move in opposite senses. The reason is that boundary 0 is a randomly
started network, so neither movement carries information about losing
plasticity. Second, "flat" is stronger than five runs support. A standard error
of 0.37 rules out a large change, not a change of about one point. Write it as
no detectable change.

## Constraint found: boundary 1 cannot be re-read

Only boundaries 4, 9 and 10 are saved as checkpoints, three files per stream,
45 files and 22.56 GB in total. Boundary 1 is not saved. The claim that the
indicator movement happens at the first task therefore rests on the reference
code route alone, from the values written live into the stream logs during
training. The trained-code route cannot be extended back to boundary 1 without
retraining the streams. This is a limitation to state, not a run to schedule.

## Planned: the centring check on an ordinary network

Notebook `50_centering_on_an_ordinary_network.ipynb` is written. It has not
been run.

**Question.** Test 2 found that about 98 percent of the feature matrix is the
single row every picture shares, and that removing it raises effective rank by
a factor near 4.7. The obvious objection is that this is a property of a
hypernetwork, whose weights are written by a generator from a 32-number code,
rather than a property of the measure. Does an ordinary network read the same
way?

**What changes.** The network only. Four are read: the study's own target
architecture holding ordinary random weights rather than generated ones; a
stock ResNet50 with random weights; the same with ImageNet weights at 64 by 64;
and the same at its own 224 by 224 size with ImageNet scaling. The first is the
sharpest, because it differs from the study's models in exactly one way, which
is where the weights come from.

**What is held fixed.** The same three 256-picture sets drawn by
`fixed_image_sets` with the same seed, the same reading site at the 2048 inputs
to the final classifier layer, evaluation mode forced, and the same effective
rank, srank and mean share arithmetic copied from notebook 48.

**Cost.** Twelve readings, forward passes only, no training. It runs on CPU and
does not compete with the probe run in flight.

**Decision.** All four near 0.98: the finding is about the measure and applies
to anyone reporting effective rank on ReLU features. Only the hypernetwork near
0.98: the finding is about hypernetworks and the paper must narrow the claim.
In between: report both numbers.

**Early signal, not a result.** The two functions were run locally on CPU
before the notebook was committed. A randomly started stock ResNet50, fed
uniform noise pictures rather than the study's picture sets, read a mean share
of 0.9994 with effective rank 3.906 rising to 59.52 when centred. That is the
same pattern the hypernetwork shows. It is one network on the wrong pictures
and settles nothing, but it is the direction the notebook is expected to
confirm.

**Results go to**
`uncle/indicator_protocol/plasticity_revised_v1/plain_network_centering.jsonl`.

## Compute plan for the CIFAR probe

Decided 16 November, before running. Notebook 49 runs order A only at first.
Orders B and C are not included, because the fifteen streams share five seeds,
so B and C vary the ordering rather than adding independent starting points,
and the question being asked is whether a 17.5 point gain survives a different
dataset. The order A per-seed spread on that gain is about one point, so five
streams settle it.

Boundary 4 is added to notebook 49 after the current run finishes rather than
by restarting it. The notebook's skip key includes the boundary, so a second
run repeats none of the 330 runs at boundaries 0 and 9 and performs only the
165 new runs at boundary 4, about 1.5 hours instead of 4.5.

Orders B and C are run only if the order A result lands between about 5 and 14
points, which is the ambiguous range where which groups were seen by boundary 4
could plausibly matter. A result near 17 or near zero does not need them.

## Result: the centring check on an ordinary network

Notebook 50 ran on 16 November. Twelve readings, forward passes only, no
training. The same three 256-picture sets and the same arithmetic notebook 48
used. Notebook 48's 450 hypernetwork readings are shown beside them.

Mean share is how much of the feature matrix is the single row every picture
shares, from 0 to 1.

| Network | in_stream | probe | eurosat |
| --- | ---: | ---: | ---: |
| hypernetwork, reference code | 0.9811 | 0.9770 | not read |
| hypernetwork, trained code | 0.9820 | 0.9793 | not read |
| study target, random weights, 64px | 0.9812 | 0.9789 | 0.9410 |
| stock ResNet50, random weights, 64px | 0.9826 | 0.9814 | 0.9294 |
| stock ResNet50, ImageNet weights, 64px | 0.3181 | 0.3101 | 0.3823 |
| stock ResNet50, ImageNet weights, 224px scaled | 0.5299 | 0.5230 | 0.5961 |

Effective rank on in-stream pictures, as published and after removing the
shared row:

| Network | Published | Centred | Times |
| --- | ---: | ---: | ---: |
| hypernetwork, reference code | 1.965 | 9.25 | 4.7 |
| hypernetwork, trained code | 2.699 | 15.46 | 5.7 |
| study target, random weights, 64px | 5.827 | 45.15 | 7.7 |
| stock ResNet50, random weights, 64px | 16.793 | 107.86 | 6.4 |
| stock ResNet50, ImageNet weights, 64px | 224.564 | 228.14 | 1.0 |
| stock ResNet50, ImageNet weights, 224px scaled | 208.233 | 220.10 | 1.1 |

**The split is training, not architecture.** Every network with untrained
weights reads a mean share near 0.98, the unmodified stock ResNet50 included at
0.9826, which is slightly higher than the hypernetwork's 0.9811. So the
generator is not the cause, and neither is the changed stem. Both trained
networks read far lower, 0.32 at 64 pixels and 0.53 at their proper 224 pixels
with their proper scaling.

The notebook's own printed verdict on this run was wrong. It averaged the three
stock readings into one number of 0.6170 and concluded the cause was the
changed stem. The table it printed shows otherwise. The verdict code has been
corrected to group by trained against untrained.

**The finding that survives is narrower and sharper.** The study's models have
learned nine tasks and still read like networks that were never trained. An
ordinary network that has really been trained does not: centring changes its
effective rank by a factor of 1.0, against 4.7 for the hypernetwork. So
centring is not a general correction to the plasticity literature's rank
numbers. It is a statement about these models.

**Effective rank falls in three steps, each smaller than the last.** Stock
ResNet50 at random initialisation reads 16.793. Swapping in the study's
architecture, still at random initialisation, drops it to 5.827. Letting the
generator write the weights drops it to 1.965. So the changed stem accounts for
part of the gap and the generator for more. None of these is the trained value
of about 224.

**The open confound, and the run that closes it.** The trained comparison was
trained on ImageNet: 1.28 million pictures and a thousand classes. The study's
models saw nine groups of twenty classes. "Trained" is therefore not matched,
and the difference could come from how much data was seen rather than from how
the weights are made. An ordinary ResNet50 trained on this study's own nine
groups, read the same way, would settle it. That run was listed as optional
before this result and is now the one that decides what the paper can claim.

Readings are in
`uncle/indicator_protocol/plasticity_revised_v1/plain_network_centering.jsonl`.
Notebook 49 was still running on order A when this was recorded.

## Result: the direct probe on CIFAR-100

Notebook 49 ran on 16 November, order A, five streams, boundaries 0 and 9. The
probe task is twenty CIFAR-100 classes chosen from seed 7, renumbered 0 to 19
and enlarged to 64 by 64. Everything else matches notebook 47: the same saved
checkpoints, the same three learning rates chosen per repetition on a
validation split, 250 steps, eleven repetitions, 4,800 training images, the
`full` arm. Chance is 5 percent for both probes.

| Run | CIFAR start | CIFAR end | CIFAR gain | Floor | Clears |
| --- | ---: | ---: | ---: | ---: | --- |
| order_A seed 0 | 34.45 | 55.03 | +20.59 | 2.21 | yes |
| order_A seed 1 | 33.20 | 55.89 | +22.69 | 1.33 | yes |
| order_A seed 2 | 33.09 | 53.93 | +20.84 | 1.86 | yes |
| order_A seed 3 | 33.31 | 52.15 | +18.85 | 1.85 | yes |
| order_A seed 4 | 33.55 | 53.61 | +20.07 | 1.55 | yes |
| mean | | | +20.61 | | 5 of 5 |

The same five streams on the Tiny ImageNet probe gain +17.00 points. The
different-dataset probe keeps 121 percent of the gain.

**The transfer objection does not hold.** The probe task shares no classes, no
pictures and no source with the stream, and the gain is larger rather than
smaller. Every run clears its own noise floor. Section 5.2's measurement is not
neighbouring-class transfer within Tiny ImageNet.

CIFAR is simply an easier twenty-class task at this budget. Both ends are
higher: 33 against 26 at boundary 0, and 54 against 44 at boundary 9. What
matters is that the distance between them survives.

**What this does not remove.** Boundary 0 is a randomly started network, so the
comparison is still untrained against trained. General visual features help on
any natural-image task, and a result of the same size on an unrelated dataset
is what general features would predict. So the narrow transfer explanation is
ruled out and the broad one is not. The paper must say which of the two it is
ruling out.

**Where the gain happens is not yet known for CIFAR.** On the Tiny ImageNet
probe the whole gain is finished by boundary 4 and the change from 4 to 9 is
-0.50 points, 1.3 standard errors from zero. The CIFAR run covers boundaries 0
and 9 only, so it cannot say whether it behaves the same way.

**Decision, and a conflict with the earlier rule.** The 15 November plan said a
clearly positive gain makes orders B and C worth five further hours. The 16
November compute plan said to run them only on an ambiguous result between
about 5 and 14 points. This result is +20.61, far outside that range. The two
rules disagree, and the 16 November one is being followed, because B and C vary
the ordering rather than the five starting points and would not change a call
with this margin.

Boundary 4 is added to notebook 49 instead. That is 165 runs and about 1.5
hours, and it tests a question that is still open on this dataset. The notebook
now reports the 0 to 4 and 4 to 9 split with a standard error, and that
reporting code was checked against the Tiny ImageNet records, where it
reproduces notebook 48's +17.50, -0.50 and 1.3 standard errors exactly.

## Cross-reading: training moves rank the other way in an ordinary network

Two results from 16 November fit together and were not planned to.

| Model | Effective rank, in-stream pictures |
| --- | ---: |
| stock ResNet50, random weights | 16.793 |
| study target architecture, random weights | 5.827 |
| hypernetwork at boundary 0, untrained generator | 5.563 |
| hypernetwork at boundary 9, after nine tasks | 2.094 |
| stock ResNet50, trained on ImageNet | 224.564 |

The hypernetwork's boundary 0 value of 5.563 and the study architecture's
random-weight value of 5.827 are close, which is a useful check: notebook 50's
first network really is standing in for the study's starting point.

Reading down the table, training an ordinary network raises effective rank by
more than a factor of ten. Training this hypernetwork lowers it by more than
half. So the fall at the first task is not a general consequence of training
beginning. It runs opposite to what an ordinary network does.

The same confound applies as before. The ordinary trained network saw 1.28
million pictures and a thousand classes. Until an ordinary ResNet50 is trained
on this study's own nine groups and read the same way, the cause cannot be
named.

## Result: where the CIFAR gain happens

Notebook 49 was run again on 16 November with boundary 4 added. The skip key
includes the boundary, so this pass performed only the 165 new runs. Settings
are unchanged from the boundaries 0 and 9 pass.

| Run | t = 0 | t = 4 | t = 9 | 0 to 4 | 4 to 9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| order_A seed 0 | 34.45 | 54.85 | 55.03 | +20.40 | +0.19 |
| order_A seed 1 | 33.20 | 52.80 | 55.89 | +19.60 | +3.09 |
| order_A seed 2 | 33.09 | 53.71 | 53.93 | +20.62 | +0.22 |
| order_A seed 3 | 33.31 | 51.69 | 52.15 | +18.38 | +0.46 |
| order_A seed 4 | 33.55 | 55.39 | 53.61 | +21.84 | -1.77 |
| mean | | | | +20.17 | +0.44 |

The change from boundary 4 to boundary 9 is +0.44 points, which is 0.6
standard errors from zero on four degrees of freedom. It is not
distinguishable from no change. The whole gain of +20.61 is +20.17 by task 4.

**The two probes agree on shape as well as size.** Tiny ImageNet gains +17.50
by task 4 and then changes -0.50. CIFAR gains +20.17 by task 4 and then changes
+0.44. The late changes have opposite signs and neither is separable from zero,
so the sign carries no information. On two probe tasks that share nothing, the
model stops changing in the same place.

This closes the question left open when the boundaries 0 and 9 pass was
recorded. It also removes the last way the probe side could have been read as
a slow accumulation across the stream. Both the indicators and the probe move
once, early, and then hold.

What it does not change is the boundary 0 confound. The large early gain is
still measured against a randomly started network, on both datasets.
