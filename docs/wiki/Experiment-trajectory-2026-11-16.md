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

## Planned: an ordinary network on the same stream

Notebook `51_an_ordinary_network_on_the_same_stream.ipynb` is written and
pushed. It has not been run.

**Question.** Notebook 50 left one hole, and it sits under both of that
notebook's findings. Its trained comparison was trained on ImageNet: 1.28
million pictures and a thousand classes, all at once. The study's models saw
nine groups of twenty classes, one group at a time. So a difference between the
two sides could come from how much data was seen, or from learning one task
after another, rather than from where the weights come from. What does an
ordinary network do on this study's own stream?

**What changes.** The weights only. They are ordinary parameters trained by
gradient descent instead of being written by a generator from a 32 number code.

**What is held fixed.** The nine groups in order A from the same ten-group
split, five epochs per group, batch 64, Adam at 1e-3, a fresh optimizer per
group, the same target architecture with its 3 by 3 stride 1 stem and spatial
mean, the same three fixed 256 picture sets from the same seed, the same
reading site at the 2048 inputs to the classifier, evaluation mode forced, and
the same effective rank, srank and mean share arithmetic. The training settings
are read off the study's own `Config` rather than retyped.

**The network.** One shared trunk and one 20 way classifier per group. The
shared trunk mirrors the shared generator and the per-group head mirrors the
per-task code. Every reading is taken on the trunk, so the heads never enter a
number. There is no preservation term, so the two sides differ in that way as
well, and the write-up has to say so.

**A second question, for free.** Every boundary is read twice. Once through the
statistics the network built up while training, which is what an ordinary user
would do. Once through the statistics it was born with, which is the route the
study uses, since the study reads through `trainer.buffer_template` so that only
the weights move between boundaries. If the two disagree, the study's protocol
is part of what it measures. A local check on a barely trained network already
reads 3.128 one way and 2.472 the other, so the two routes do separate.

**Boundary 1 comes for free.** The study cannot re-read its own boundary 1,
because only boundaries 4, 9 and 10 were saved. Here the readings are taken
while the run goes, so all ten boundaries are recorded. Boundary 1 is where the
study's whole rank fall happens, so this is the sharpest single comparison in
the notebook.

**Cost.** Three seeds, 7,065 training steps each, about half an hour per seed
on an A100. Results are written and uploaded after each seed, and a seed
already in the file is skipped, so a stopped run can be restarted.

**Decision.** Effective rank rises over the stream: the fall is about the
hypernetwork and the opposite direction finding is a real result with the
confound closed. Effective rank falls in both buffer modes: the fall is about
sequential training on this much data, and the opposite direction claim comes
out of the paper. The two buffer modes disagree: the reading route is part of
the measurement and must be stated before any rank number is reported. Mean
share stays near 0.98 after training: the untrained signature is about this data
and this budget rather than about the generator.

**Checks run before pushing.** The notebook's three function cells were executed
on CPU against small fake data. Six checks passed: the trunk returns 2048
features with gradients on, 159 BatchNorm statistics move during training and
all are restored, a boundary read leaves the model's mode and statistics as it
found them, the two buffer modes give different numbers, training lowers the
loss, and mean share reads 1.0 on identical rows.

**Results go to**
`uncle/indicator_protocol/plasticity_revised_v1/plain_network_stream.jsonl`.

## Result: an ordinary network on the same stream

Three seeds finished, 4.7 minutes each, 180 readings in the file. All ten
boundaries were read, both buffer modes, on all three picture sets.

**The training worked.** Each group reached 57.7 percent on average right after
it was learned, against 5 percent chance on twenty classes. By the end of the
stream the same groups read 28.0 percent through their own heads. So the
ordinary network learns every group and then forgets a large part of it. The
study's models do not forget, because they pay a preservation term that this
network does not have. That difference is a property of the training objective,
not a plasticity measurement, and it should not be read as one.

**Effective rank on the in-stream pictures, three seeds.**

| boundary | own statistics | start statistics | study's models |
|---|---|---|---|
| 0 | 7.30 | 7.30 | 5.56 |
| 1 | 59.21 | 1.16 | 2.07 |
| 9 | 115.66 | 1.18 | 2.79 |
| change | +108.37 | -6.12 | -2.78 |

The study's column is the five order A streams read through
`trainer.buffer_template`.

**The study's route is the start statistics route, exactly.**
`Trainer.__init__` clones the buffers of the target as `build_target` returns
it, and `build_target` calls `resnet50(weights=None)`. So the template holds a
running mean of 0 and a running variance of 1 at every BatchNorm layer, and
those values never change. Notebook 51's start statistics are a snapshot of the
same freshly built network. The two routes hold the same numbers, so the
comparison is matched and not approximate.

**Matched route to route, the two sides agree.** Read the way the study reads,
the ordinary network falls from 7.30 to 1.18 and the study's models fall from
5.56 to 2.79. Both lose almost all of it by boundary 1. Both end close to a
rank of one. The hypernetwork is not doing anything the ordinary network does
not do.

**Sequential training on these nine groups raises effective rank.** Read
through statistics measured from its own weights, the ordinary network goes
from 7.30 to 115.66, a factor of about sixteen. So the fall is not what
sequential training on a small stream does to a network. It is what reading
does. The data and the budget are not the cause, and the record should not say
they are.

**The route decides the direction.** At boundary 9 the same three networks, the
same pictures and the same arithmetic read 115.66 through the statistics the
networks built and 1.18 through the statistics they were born with. That is a
gap of 114.48, or 99 percent of the larger number. One route says the features
grew much richer over the stream. The other says they collapsed onto a single
direction. The paper cannot report a rank number without first saying which
route produced it.

**Mean share on the in-stream pictures, three seeds.**

| boundary | own statistics | start statistics | study's models |
|---|---|---|---|
| 0 | 0.9821 | 0.9821 | about 0.98 |
| 9 | 0.7400 | 0.9542 | 0.9811 |

**Correction to the notebook's own printed verdict.** Part 3 of the final cell
compared the own statistics reading at boundary 9, 0.7400, against the study's
0.9811, and concluded that the untrained signature does not survive training.
Those two numbers come from different routes, so the comparison is not valid.
On the matched route the ordinary network reads 0.9542 and the study's models
read 0.9811. The signature does survive. The notebook's printed text is wrong
on this point and the result file is not.

**What this takes out of the paper.** The claim that effective rank moves in
the opposite direction for hypernetworks has to go. On the matched route an
ordinary network falls the same way, by the same boundary, to the same place.

**What this puts into the paper, and how to frame it.** The claim is about
hypernetworks, not about how the continual learning field measures ordinary
networks. Nobody evaluates a standard ResNet50 by overwriting its learned
BatchNorm running averages with initialisation values, so a finding written as
"effective rank fails on an ordinary ResNet50 under fixed statistics" reads as a
bug in our own evaluation script and will be dismissed as one.

The finding belongs to stateless weight generation. A hypernetwork writes fresh
weights from a task code every time it is asked, so it carries no per-task
running BatchNorm statistics of its own. That absence pushes anyone measuring
one towards holding the statistics fixed at initialisation, which is what this
study did and what `trainer.buffer_template` is. Notebook 51 is the control
experiment for that choice. It shows that holding BatchNorm statistics at
initialisation moves effective rank down by 99 percent, from 115.66 to 1.18,
and locks the shared row share between 0.95 and 0.98, in a ResNet50 that is
demonstrably healthy.

**The strongest single line.** The control network reads 1.18 under the
protocol, a value that would normally be reported as a near total collapse,
while at that same moment it is learning each new group of twenty classes to
57.7 percent from 5 percent chance, and while its own statistics say its
features have grown sixteen times richer. The protocol says the model cannot
learn. The model is learning. So a fixed-statistics effective rank does not
measure the ability to learn a new task, and any plasticity claim resting on
one has to be rechecked under measured statistics first.

**Not settled by this run.** The hypernetwork has only one column. Notebook 51
read the ordinary network both ways, but the study's models have never been read
through statistics measured from their own generated weights, so the table is
half empty on the side that matters. Notebook 52 fills it.

This run also does not say what a hypernetwork should use instead. It writes
fresh weights for every code and has no single set of running statistics that
belongs to it, which is the honest reason the fixed template was chosen. The run
shows the choice is not free. It does not name a replacement.

## Planned: both buffer routes on the saved hypernetwork models

**Notebook 52.** Forward passes only, no training.

**Question.** Notebook 51 gave the ordinary network two columns, one per reading
route. The study's models have only ever been read one way. What do they read
when the BatchNorm statistics are measured from the weights the generator
actually writes?

**What changes.** The buffers only. For each code, the generated ResNet50 is run
over 2048 pictures from the nine learned groups' training split with the
BatchNorm layers in training mode and `momentum` set to None, so each layer
keeps a plain running average over every batch rather than an exponentially
decaying one and the result does not depend on batch order. The indicators are
then read through those measured values.

**What is held fixed.** The same 256 picture reading sets from the same seed,
the same reading site at the 2048 inputs to the classifier, evaluation mode
forced for every reading, and the same effective rank, srank and mean share
arithmetic. The adaptation pictures come from the training split and the reading
pictures from the held-out split, so the statistics are never fitted to the
pictures the indicator is computed on.

**Scope.** Order A, five seeds, boundaries 0 and 9. Boundary 0 needs no download
because the untrained generator is rebuilt from the seed, and without it the
table has no starting value and no change column. Boundary 9 is five 540 MB
files, deleted one at a time. Both the fixed reference code and the nine trained
codes are read, kept apart in the table, because notebook 48 already showed the
reading moves by about 1.4 times between them. A trained-code row averages a
seed's nine codes first, so a seed counts once rather than nine times.

**Decision.** If the hypernetwork rises under measured statistics the way the
ordinary network does, the fall in every published figure of this study is the
reading route and nothing else, and the rank section becomes a measurement
result. If it stays low under both routes, the two networks genuinely differ and
the fixed template is not the whole story. The mean row share decides the second
question the same way: the ordinary network holds 0.954 under the fixed route
and drops to 0.740 under measured statistics, so if the hypernetwork holds near
0.98 under both, the shared row belongs to the generator.

**A limit to state either way.** A hypernetwork carries no running statistics of
its own, so the measured route is a quantity this notebook constructs rather
than one the model holds. It answers what these weights would do if their
statistics matched them, which is the right question for a comparison against an
ordinary network. It is not a claim that the study should have used it.

**Checks run before pushing.** The notebook's measurement cell was executed on
CPU against a real ResNet18 hypernetwork. Seven checks passed: the fixed route
really is a running mean of 0 and a running variance of 1 at all 20 layers, the
measured route moves every buffer and every layer counts every batch, the
module's momentum and training mode are restored and the study's own template is
never written into, the two routes give different features, both readings give
sane rank and share values, mean share reads 1.0 on identical rows, and two
different codes get different measured statistics.

**Results go to**
`uncle/indicator_protocol/plasticity_revised_v1/buffer_route_on_hypernetwork.jsonl`.

## Correction: notebook 51's printed verdicts 1 and 3

Both mixed the two reading routes, and both are now fixed in the notebook. The
result file was always correct; only the printed text was wrong.

Verdict 1 compared the ordinary network's fixed-statistics change against the
study's models and concluded that sequential training on this much data is what
lowers effective rank. Under measured statistics the same network rises from
7.30 to 115.66. Sequential training on these nine groups raises effective rank
by a factor of about sixteen. The fall is caused by reading evolved weights
through statistics that were never updated, and the printed verdict now says so
and reports both columns.

Verdict 3 compared the ordinary network's measured-statistics mean share at
boundary 9, 0.7400, against the study's 0.9811, which is a fixed-statistics
number, and called the gap an architectural difference. On the matched route the
ordinary network reads 0.9542, and 0.9736 at boundary 7. The printed verdict now
reports both routes and compares only the matched one.

## Result: both buffer routes on the saved hypernetwork models

Notebook 52 finished. Order A, five seeds, boundaries 0 and 9, both the fixed
reference code and the nine trained codes, no training. The ordinary network's
columns are notebook 51's, read back from its own file.

**Effective rank on the in-stream pictures.**

| what | measured statistics | fixed statistics |
|---|---|---|
| hypernetwork t=0, reference code | 106.272 | 5.563 |
| hypernetwork t=9, reference code | 143.001 | 2.789 |
| hypernetwork t=9, trained codes | 137.542 | 3.837 |
| ordinary ResNet50 t=0 | 7.300 | 7.300 |
| ordinary ResNet50 t=9 | 115.664 | 1.181 |

**Mean row share on the in-stream pictures.**

| what | measured statistics | fixed statistics |
|---|---|---|
| hypernetwork t=0, reference code | 0.805 | 0.981 |
| hypernetwork t=9, reference code | 0.865 | 0.983 |
| hypernetwork t=9, trained codes | 0.814 | 0.983 |
| ordinary ResNet50 t=0 | 0.982 | 0.982 |
| ordinary ResNet50 t=9 | 0.740 | 0.954 |

**Centred effective rank on the in-stream pictures.**

| what | measured statistics | fixed statistics |
|---|---|---|
| hypernetwork t=0, reference code | 156.874 | 42.211 |
| hypernetwork t=9, reference code | 193.857 | 17.173 |
| hypernetwork t=9, trained codes | 176.627 | 26.909 |
| ordinary ResNet50 t=0 | 57.836 | 57.836 |
| ordinary ResNet50 t=9 | 141.625 | 1.566 |

**The route decides the sign, not just the size.** Read through statistics
measured from the generated weights, the hypernetwork rises from 106.272 to
143.001 over the nine groups. Read through the fixed template, the same five
models fall from 5.563 to 2.789. The same models, the same 256 pictures, the
same arithmetic. One route says the features grew richer, the other says they
collapsed onto one direction. At boundary 9 the two readings differ by a factor
of 51.

Every effective rank figure in the draft is the falling number. The claim that
this study's models lose rank over the stream is a statement about the reading
route, and the opposite statement is equally available from the same weights.

**The control network behaves the same way, which is the point.** The ordinary
ResNet50 reads 115.664 and 1.181 at the same boundary. Two networks that share
nothing but a backbone both collapse by about two orders of magnitude when read
through statistics that never moved. The effect is the protocol.

**The code effect is mostly the same effect.** At boundary 9 the trained codes
read 3.837 and the reference code 2.789 under the fixed template, a ratio of
1.38, which is the 1.4 times swing notebook 48 reported. Under measured
statistics the same comparison is 137.542 against 143.001, a ratio of 0.96. So
when each generated network is read through statistics that match it, which code
you read through barely changes the answer. The code sensitivity is not a
separate trap. It is the fixed-template trap seen from another angle, and the
write-up should say so rather than list it as independent evidence.

**The shared row is mostly the route, and what is left is a real difference.**
The fixed template holds both networks between 0.95 and 0.98 at every point. So
most of the 0.98 signature is the route.

Under measured statistics the two networks separate, and they move in opposite
directions. The hypernetwork starts at 0.805 and rises to 0.865. The ordinary
network starts at 0.982 and falls to 0.740. Training pushes an ordinary
network's features apart and does not push the hypernetwork's apart. That is the
clearest route-matched architectural difference in the table, and unlike the
code effect it survives the correction. Both end far above the 0.318 a stock
ImageNet trained ResNet50 reads at 64 pixels.

**With both confounds removed the hypernetwork is not the degenerate one.**
Centred and read through measured statistics, the hypernetwork reads 193.857 at
boundary 9 and the ordinary network 141.625. After taking out the reading route
and the shared row, the generated features are more varied than the ordinary
network's, not less. Nothing in this study supports a claim that these models
are representationally impoverished.

**Both networks grow, the ordinary one faster.** Under measured statistics
the hypernetwork's centred rank goes from 156.874 to 193.857, a rise of 36.983,
and the ordinary network's from 57.836 to 141.625, a rise of 83.789. Both gain
feature variety over the stream. The ordinary network gains more, and it starts
from much lower. Under the fixed template both fall instead, the hypernetwork
from 42.211 to 17.173 and the ordinary network from 57.836 to 1.566.

### Open: the rebuilt boundary 0 does not match the logged boundary 0

Notebook 52 rebuilds boundary 0 from the seed rather than from a file, because
no boundary 0 checkpoint exists. Under the fixed template that rebuild reads
11.790, while the live indicator logs record 5.563 at t=0 for order A on the
same route. Boundary 9 matches to three decimals, 2.789 here against 2.788 in
the logs, so the loading path and the picture sets are not the problem. Only the
rebuild differs.

Reading the code did not settle it. The study records boundary 0 inside
`run_request` after `h.add_task(task)`, but `add_task` only creates a task code,
and `weights_from_code` for the reference code depends on the trunk, the heads,
the chunk codes, the scales and the offsets, none of which `add_task` touches.

Until this is explained, no boundary 0 number from notebook 52 goes in the
paper, and that includes the change columns for the hypernetwork. The decisive
comparison does not need it: the 143.001 against 2.789 gap is measured at
boundary 9 alone, on five loaded checkpoints.

The next check is cheap and needs no GPU. Read the order A indicator logs
directly and find what the t=0 rows actually contain: how many there are per
seed, which code and which buffers produced them, and whether `t` indexes the
boundary the way notebook 51's verdict cell assumed.

One thing the table already explains. For the ordinary network the two columns
are identical at t=0, which they must be, because an untrained network's own
statistics are the ones it was born with. For the hypernetwork they differ,
119.517 against 11.790, because at boundary 0 the generator already writes
weights that are not the target's initialisation weights. So the two routes
separate for these models before any task is learned.

## Resolved: the boundary 0 mismatch was an unseeded rebuild

**Cause.** `HyperNetwork.__init__` builds the generator with
`nn.init.kaiming_normal_`, `nn.init.normal_` and `torch.randn`. All three read
the global random number generator. None of them reads `config.seed`. The study
seeds that global generator before it builds: `uncle/paired_study.py` calls
`random.seed`, `np.random.seed` and `torch.manual_seed` with `config.seed`, and
`torch.cuda.manual_seed_all` on a GPU, immediately before `build_target` and
`HyperNetwork`.

Notebooks 52 and 48 rebuilt a model without those four calls. So the generator
they built was a fresh random draw rather than the study's. At boundary 9 the
checkpoint overwrites every weight, which is why that boundary matched the logs
to three decimals. Boundary 0 has no checkpoint, so nothing overwrote the wrong
draw.

**Shown on CPU.** Two `HyperNetwork` builds with the same `Config` in one
process give different chunk codes. With the three seed calls first they are
identical. So the construction is reproducible only when the global generator is
seeded.

**The logged values were right.** The order A logs hold one indicator row per
boundary, `t` running 0 to 10, exactly as notebook 51's verdict cell assumed.
Effective rank on the in-stream pictures at boundary 0 reads 7.142, 6.374,
4.758, 3.131 and 6.410 across the five seeds, a mean of 5.563. Boundary 9 reads
a mean of 2.788 against notebook 52's 2.789. Nothing in the published records
needs changing.

**What was wrong.** Notebook 52's boundary 0 row, which read 11.790 under the
fixed template and 119.517 under measured statistics. Both are readings of
models that were never part of this study.

**Fixed.** Both notebooks now seed the global generator inside `build` before
constructing anything, with a comment saying why. Notebook 52's results key
moves to `buffer_route_on_hypernetwork_v2.jsonl`, so the rows already saved are
not resumed from. The old file stays in place as evidence and is not deleted.

**What this does not touch.** Every boundary 9 number in the previous section
stands: those five models came from checkpoints. The decisive comparison,
143.001 against 2.789, is unaffected. Notebook 48's test 4 arm also rebuilt
boundary 0 without seeding, but its result is that the frozen generator reads
chance at both ends, and chance does not depend on which random generator the
model started from. Notebook 48's indicator readings used boundaries 4 and 9
only, both loaded from files.

**Confirmed by the re-run.** Notebook 52 was run again under the fixed build.
Boundary 0 now reads 5.563 through the fixed template, matching the logged mean
of the five order A seeds to three decimals. The rebuild reproduces the study's
own boundary 0, so every number in the table above is comparable with the
published records. The tables in the previous section carry the corrected
values.

## Result: learning speed, from the probe records already on disk

No new run. `scripts/learning_speed.py` reads the published probe files from R2
and uses the four checkpoints every probe run already recorded, at steps 62,
125, 188 and 250. 1155 full-arm runs. Boundary 4 exists for order A only.

Runs are matched. A boundary 0 run and a boundary 9 run sharing an order, a
seed, a learning rate and a repetition started from the same fresh code and saw
the same batches in the same order, so they differ only in the model being
probed. Pairing removes the starting-code spread, which is the largest source of
noise in this measurement.

**Test accuracy on the held-out group, percent.**

| boundary | step 62 | step 125 | step 188 | step 250 | runs |
|---|---|---|---|---|---|
| t=0 | 9.78 | 15.07 | 18.47 | 21.27 | 495 |
| t=4 | 30.58 | 36.33 | 36.44 | 38.84 | 165 |
| t=9 | 32.85 | 36.96 | 38.25 | 39.57 | 495 |
| paired gap, t=9 minus t=0 | +23.07 | +21.89 | +19.78 | +18.29 | 495 |

**Training loss on the held-out group.**

| boundary | step 62 | step 125 | step 188 | step 250 | runs |
|---|---|---|---|---|---|
| t=0 | 3.00 | 2.69 | 2.47 | 2.31 | 495 |
| t=4 | 2.01 | 1.45 | 1.12 | 0.89 | 165 |
| t=9 | 1.91 | 1.28 | 0.93 | 0.70 | 495 |
| paired gap, t=9 minus t=0 | -1.09 | -1.41 | -1.54 | -1.61 | 495 |

Every gap in both tables is larger than its own 95 percent interval.

**The gain is speed, not ceiling.** The accuracy gap is widest at the first
checkpoint and narrows from there, +23.07 at step 62 down to +18.29 at step 250.
A model that merely ends higher at the same pace would show the opposite shape,
a gap that grows as the budget is spent. This one shrinks, which is what a
faster learner looks like when both runs are given the same fixed budget.

**A measure that a higher ceiling cannot inflate.** For each run, divide its
accuracy at step 62 by its own accuracy at step 250. That share says how much of
this run's own final result was already reached early, and because each run is
divided by its own ceiling, a ceiling difference cannot move it.

| boundary | share of its own final accuracy reached at step 62 |
|---|---|
| t=0 | 0.476 |
| t=4 | 0.787 |
| t=9 | 0.837 |

The paired gap from boundary 0 to boundary 9 is +0.361 with a 95 percent
interval of 0.028, over 495 pairs. The untrained generator needs the whole
budget. The model at the end of the stream is 84 percent finished at step 62.

**The cleanest single sentence this study has.** At step 62 the end-of-stream
model reads 32.85 percent on a group of twenty classes it has never seen. The
untrained generator does not reach that in the whole 250-step budget; it ends at
21.27 percent. So the trained model is not slightly better at learning something
new. It passes the untrained model's best result in a quarter of the budget.

**Why this matters more than the endpoint.** Loss of plasticity is defined as a
loss of the ability to learn, and the literature measures it as slower fitting
of a new task. The study's headline number, accuracy after a fixed 250 steps,
mixes speed and ceiling and does not answer that definition directly. The speed
share does. It moves the wrong way for a plasticity claim, by a large margin,
on a statistic a ceiling cannot inflate.

**Where this disagrees with an existing finding, weakly.** The accuracy endpoint
is finished by task 4: the change from boundary 4 to boundary 9 is +0.44 points
on CIFAR and -0.50 on Tiny ImageNet, both inside the noise. The speed share is
not. On order A, over 165 matched pairs from five seeds, the share rises a
further +0.076 from boundary 4 to boundary 9.

The interval on that number depends on what is counted as an independent
observation. Counting every repetition gives 0.044. The eleven repetitions of
one seed probe the same two models, so they are not eleven independent readings
of the difference between those models. Counting each seed once, with the
correct small-sample multiplier for four degrees of freedom, gives 0.063. The
gap clears that, about 3.3 standard errors, roughly p = 0.03.

So this is real but thin: order A alone, five seeds, and one comparison among
several this study has made. It should be reported as weak evidence that the
speed gain continues past task 4 while the accuracy endpoint has stopped, not as
a settled disagreement. Orders B and C of boundary 4 would settle it, and that
is a GPU run the study has so far declined.

**One correction to how intervals were being computed.** The script first used
1.96 for every interval. On five seeds the right multiplier is 2.776. Using 1.96
there makes the interval about 40 percent too narrow, which is exactly the
difference between the boundary 4 to 9 result standing and not standing. The
script now picks the multiplier from the number of observations and carries a
runnable check on that function.

## Result: srank under both routes, and the one indicator still uncovered

Read back from notebook 52's own file. No new run.

**srank on the in-stream pictures.**

| what | measured statistics | fixed statistics |
|---|---|---|
| hypernetwork t=0, reference code | 248.20 | 215.60 |
| hypernetwork t=9, reference code | 248.40 | 120.40 |
| hypernetwork t=9, trained codes | 248.36 | 160.60 |

**srank, shared row removed.**

| what | measured statistics | fixed statistics |
|---|---|---|
| hypernetwork t=0, reference code | 248.40 | 238.80 |
| hypernetwork t=9, reference code | 248.60 | 182.20 |
| hypernetwork t=9, trained codes | 248.16 | 212.51 |

**srank does not move at all under measured statistics.** It reads 248.20 at
boundary 0 and 248.40 at boundary 9. Through the fixed template the same models
fall from 215.60 to 120.40, a drop of 44 percent. The published srank fall is
the reading route in full. This is a cleaner result than effective rank, where
the measured route gives a rise rather than a flat line.

**One caveat on that flatness.** The feature matrix has 256 rows, so srank
cannot exceed 256. A reading of 248 is close to that ceiling. "Flat at 248" and
"saturated at the ceiling" cannot be told apart from this measurement alone, so
the write-up should say srank shows no fall under measured statistics rather
than claiming it is unchanged. Reading on more pictures would separate the two
and costs one forward pass per model.

**The code effect is the same effect here too.** Under the fixed template the
trained codes read 160.60 and the reference code 120.40, a ratio of 1.33. Under
measured statistics the same pair reads 248.36 and 248.40, a ratio of 1.00.

### Still uncovered: dormant units

The draft reports three indicators and the opening table carries all three:
effective rank, srank and dormant units. The buffer route now explains the first
two. Dormant units have never been measured under the two routes, because
notebook 52 records effective rank, srank, the centred versions and the mean row
share, and nothing about dormant units.

This matters because the claim is that the indicators read the protocol. A
reviewer who sees two of three indicators explained will ask about the third,
and it is in the paper's own first table. The dormant unit count is also the
most likely of the three to be a direct consequence of the mechanism, since a
unit counted as dormant is one whose activation is small, and reading through
statistics that do not match the weights is exactly what would make activations
small.

**What it needs.** One field added to notebook 52's reading loop and a re-run.
The dormant count is already implemented in `uncle/plasticity_indicators.py` and
is computed on the same features the other indicators use, so this is a small
change rather than a new measurement. The cost is the same 2.7 GB of downloads
and a few minutes of arithmetic. No training.
