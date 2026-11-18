# Rank-based plasticity indicators and directly measured learning ability in a hypernetwork continual learner

## Abstract

A network trained on tasks in sequence can lose the ability to learn new ones.
Measuring that loss directly means training the network on a further task,
which is costly enough that it is rarely repeated, so practitioners read it off
the weights instead, using effective rank, srank, or the fraction of dormant
units. Each is normalised so that overall scale cannot change it, which is
taken to mean that two readings may be compared.

We test the three against the quantity they stand for, on fifteen streams of
nine 20-class Tiny ImageNet tasks in a hypernetwork continual learner, with the
direct measurement repeated 990 times so that every model carries its own noise
floor.

The reading protocol decides the result. Every indicator value of this kind is
read through one set of BatchNorm statistics, and which set is almost never
reported. Read through the fixed statistics of a freshly built network the
indicators fall on every run; read through statistics measured from the same
weights they rise, and at the end of the stream the two readings differ by a
factor of 51. An ordinary ResNet50 reproduces the fall more strongly, while
learning every group it is given. Set the protocol aside and the measures still
fail: every indicator reports a loss while measured fit to an unseen task
improves on all fifteen runs, the largest correlation between the two is 0.212
and carries the wrong sign, and a model with one task destroyed to chance moves
both rank measures in the direction they report as healthy. In this setting a
reading of these measures is not evidence about the ability to learn.

# 1 Introduction

A model that has learned many tasks may fit the next one worse than a freshly
started model would. The direct way to find out is to train it on that next
task and look, which costs one training run per reading. That is why the
measurement is usually run once, reported without an error bar, and rarely
repeated.

Three measures are used instead, all read with no training: effective rank (Roy
and Vetterli, 2007), srank (Kumar et al., 2021), and the fraction of dormant
units (Sokar et al., 2023). All three divide away the overall size of what they
measure, and that normalisation is what licenses a comparison between two
models.

This paper puts the three and the direct measurement side by side on the same
models, in a setting where the direct measurement is cheap enough to repeat
eleven times at every reading point. We report four results: a reading choice
that nobody reports decides the sign of the indicator result; the reported fall
reproduces more strongly on an ordinary ResNet50 that learns every group it is
given; the indicators and the direct measurement disagree in direction and the
size of the indicator change does not rank the runs; and repeating the direct
measurement gives each model a noise floor, which turns a reported change into
a claim that can be checked.

# 2 Method

**Model and stream.** A generator writes the weights of a ResNet50 from a
32-number task code; the target network is never trained directly. The stream
is 200 Tiny ImageNet classes in 10 disjoint groups of 20. Groups 0 to 8 are
learned in sequence and group 9 is held out entirely, so chance accuracy on the
held-out group is 5 percent. Each group is trained for 5 epochs at batch size
64, at a learning rate chosen by a sweep run beforehand. Fifteen streams were
run: three orderings, five seeds each. A reading is taken at every boundary,
numbered 0 before any task and 9 after the ninth.

**Reading protocol.** The weights are generated from one fixed reference code,
never trained, so that a change between boundaries is a change in the generator
and not in the code. The BatchNorm layers are given the statistics a freshly
built ResNet50 carries, a running mean of zero and a running variance of one at
every layer, never updated, because the generator writes weights but no running
statistics. The model is read in evaluation mode at the 2048 features entering
the final classifier, on 256 fixed images. Section 3.1 reads the same models a
second way, through statistics measured from their own weights over 2048
training images from the nine learned groups.

**Measures.** Effective rank and srank with delta = 0.01 are computed from the
singular values of the 256 by 2048 feature matrix, and the dormant-unit
fraction at cut-offs 0.01, 0.025 and 0.1.

**The direct probe.** A copy of the model is taken, a fresh task code is drawn
for the held-out group, the generator and that code are trained together for
250 steps, the accuracy is recorded, and the copy is discarded, so the
measurement cannot change the model it measures. The probe runs eleven times at
each point, and each run's noise floor is 1.96 s sqrt(1/11 + 1/11), where s is
the standard deviation of its eleven boundary-0 repetitions.

**The damaged model and the control.** After the ninth group each stream
destroys one group it has already learned; that group reads 5.00 percent
afterwards in all fifteen streams, which is exactly chance. Separately, an
ordinary ResNet50 with one classifier head per group was trained on the same
nine groups in the same order with the same settings, three seeds. It shares no
component with the hypernetwork, so behaviour common to both belongs to the
reading and not to the architecture.

# 3 Results

## 3.1 The reading protocol decides the direction

Table 1 reads the same checkpoints twice: through the fixed statistics of the
protocol, and through statistics measured from the models' own weights.

Table 1. Each measure at the start and the end of the stream, read twice from
the same models. The hypernetwork rows are the mean over five seeds of ordering
A, the control rows the mean over three seeds.

| Measure | Network | Fixed, b0 | Fixed, b9 | Matched, b0 | Matched, b9 |
| --- | --- | --- | --- | --- | --- |
| Effective rank | Hypernetwork | 5.56 | 2.79 | 106.27 | 143.00 |
| Effective rank | Ordinary ResNet50 | 7.30 | 1.14 | 7.30 | 110.51 |
| srank | Hypernetwork | 215.6 | 120.4 | 248.2 | 248.4 |
| srank | Ordinary ResNet50 | 226.0 | 2.7 | 226.0 | 245.3 |
| Dormant units, cut-off 0.01 | Hypernetwork | 0.160 | 0.324 | 0.000 | 0.128 |
| Dormant units, cut-off 0.01 | Ordinary ResNet50 | 0.156 | 0.527 | 0.156 | 0.000 |

The route decides the direction of travel, not only its size. Through the fixed
statistics the hypernetwork's effective rank falls from 5.56 to 2.79 over the
nine groups; through matched statistics the same five checkpoints rise from
106.27 to 143.00, a factor of 51 apart at boundary 9. The control behaves the
same way, falling from 7.30 to 1.14 and rising to 110.51, and its fixed reading
at boundary 9 is lower than any value the hypernetwork reaches although it
learns every group at the moment that group is given to it. The dormant-unit
fraction tells the same story: on the fixed route the control rises further than
the hypernetwork, and on the matched route the two move in opposite directions.

Appendix B locates the failure. A BatchNorm layer performs two operations and
the protocol supplies a wrong value for each; correcting either alone recovers
0.2 and 4.3 percent of the gap at boundary 9, so a partial fix such as centring
the features does not repair the reading.

## 3.2 The indicators fall while measured learning rises

Read through the protocol, the stream produces the fall these measures were
built to detect. Over the same stream the direct measurement moves the other
way.

Table 2. Each indicator and the direct probe, averaged over the fifteen runs.
The last column counts the runs that move in the stated direction.

| Quantity | Boundary 0 | Boundary 9 | Runs |
| --- | --- | --- | --- |
| Effective rank | 5.56 | 2.09 | fell in 15 of 15 |
| srank | 215.6 | 81.1 | fell in 15 of 15 |
| Dormant units, cut-off 0.01 | 0.160 | 0.342 | rose in 15 of 15 |
| Probe accuracy | 26.20% | 44.08% | rose in 15 of 15 |

Boundary 0 is the comparison the literature uses: the generator is still at its
random starting values, so the two endpoints ask whether a model that has
learned nine tasks fits a new one better or worse than a freshly started model
given the same budget. A loss of plasticity is the case where it does worse.
Every run does better, by 17.87 points on average against a chance level of 5
percent. The smallest gain is 15.90 points and the largest noise floor is 2.72,
so no run comes close to its own floor.

## 3.3 The size of the fall does not rank the runs

Both quantities vary widely across the fifteen runs. The effective-rank fall
runs from 29.8 to 80.6 percent and the probe gain from 15.90 to 20.49 points.
If the fall reports the ability to learn, the runs that fell furthest should be
the runs that gained least.

Table 3. Correlation between each indicator's change and the probe's change,
over the fifteen runs. The last column repeats the test on the boundary-9
values themselves rather than on the changes.

| Measure | Predicted sign | r | r squared | r, end values |
| --- | --- | --- | --- | --- |
| Effective rank | positive | -0.054 | 0.3% | -0.165 |
| srank | positive | -0.178 | 3.2% | -0.212 |
| Dormant units, cut-off 0.01 | negative | +0.129 | 1.7% | +0.164 |
| Dormant units, cut-off 0.025 | negative | +0.125 | 1.6% | +0.162 |
| Dormant units, cut-off 0.1 | negative | +0.112 | 1.3% | +0.139 |

![](figures-v3/v3-fig3-rank-change-vs-probe-change.png)

Figure 1. Change in effective rank against change in probe accuracy, one point
per run. Error bars are each run's own standard error on the probe change. The
dashed line is the least-squares fit.

Each measure predicts a sign, and all five come out with the wrong one. The
last column sets the differences aside and asks whether, among the fifteen
finished models, the one with the higher reading fits the unseen task better.
It does not, in the same direction.

The eleven repetitions show this is not the outcome's fault. Of the spread in
the gain across the fifteen runs, 46.2 percent by variance is real difference
between models rather than measurement noise, which caps any predictor at a
correlation of 0.680. The largest value in Table 3 is 0.212, and at fifteen
runs a correlation needs 0.514 to be separable from zero at the 5 percent
level.

## 3.4 The indicators call a damaged model healthy

If the indicators report anything about the health of a model, a model with one
group destroyed to chance should read worse than the same model one step
earlier.

Table 4. The unlearning step and the weight-scale controls, averaged over the
fifteen streams. The ordinary learning step is the average absolute change from
one boundary to the next over the 120 steps after the first.

| Quantity | Effective rank | srank |
| --- | --- | --- |
| Change at the unlearning step | +1.580 | +73.6 |
| One ordinary learning step | 0.162 | 10.025 |
| Change with the weight norm held at its boundary 9 value | +0.754 | +37.7 |
| Change produced by shrinking the undamaged model alone | +1.314 | +53.9 |

Both rank measures rise at the unlearning step, in every stream, and a rise is
the direction they report as healthy. The movement is large beside ordinary
learning: effective rank moves 1.580 against 0.162 for a typical learning step,
9.8 times as far and in the opposite direction.

Part of that movement is the size of the weights, which the unlearning step
leaves at 0.757 of its earlier value. Both rank measures are normalised so that
multiplying the feature matrix by a constant cannot change them, but that does
not cover the weights, which sit upstream: a residual block adds its input to a
branch of weighted layers, so scaling every weight changes the branch and
leaves the input path alone. Holding the norm fixed removes about half of the
movement and imposing it alone reproduces most of it; the two bounds disagree,
so no single share is available. What survives the first control is +0.754 on
effective rank, 4.7 times an ordinary learning step, in 14 of the 15 streams,
so the weight scale does not explain the result away.

# 4 Discussion

The three measures respond to something, and on this evidence it is not the
ability to learn. Two of the things they do respond to are identified here: the
reading protocol, which decides the sign of the result and is not usually
reported, and the size of the generated weights, which reaches the reading
along a path the measures' own normalisation does not cover.

The protocol result travels furthest, because it does not depend on the
architecture: the control contains no hypernetwork and reproduces the published
fall more strongly while learning every group it is given. An indicator value
is therefore incomplete without the protocol that produced it, and a change in
one is not evidence about learning until the direct measurement has been run at
least once on the same models. None of this shows the measures are meaningless
everywhere: this is one architecture, one dataset and one kind of learning.

# 5 Limitations

One model family, one hypernetwork, one dataset. The probe task is drawn from
the same dataset as the stream and the probe trains the generator, so part of
the measured gain is ordinary transfer; the frozen-generator arm that would
isolate the code space reads chance at both boundaries. The indicators move
almost entirely at the first task, so the endpoint comparison is largely
untrained against trained, and the probe was not read at boundary 1. The fifteen
runs share five starting points, and the three orderings of one seed, which
begin from the same model, differ by up to about two points on the boundary-0
probe reading for reasons not established here. No test for srank was stated
before the measurements were taken. Appendix B is on ordering A alone, and the
control has three seeds against the hypernetwork's five.

# References

Carried over from version 1 and not yet transcribed into this file.

# Appendix A Indicator trajectories over the stream

Fifteen streams were run: three orderings of the nine task groups, five seeds
each. All five measures were read at every task boundary on the same 256
in-stream images. Every run moves the way a loss of plasticity would predict.
Effective rank and srank fall in all fifteen, and the dormant-unit fraction
rises in all fifteen at each of the three cut-offs.

Nearly all of that movement happens at the first task (Table A1, Figure A1).
Averaged over the fifteen runs, effective rank falls from 5.56 to 1.72 after
one task of the nine and ends at 2.09. srank falls from 215.6 to 66.1 and ends
at 81.1. The dormant fraction at cut-off 0.01 rises from 0.160 to 0.366 and
ends at 0.342. Each measure passes its boundary-9 value after a single task
and then moves back a little, and the last column of Table A1 counts how many
runs move back that way.

Table A1. Each measure averaged over the fifteen runs, at the start of the
stream, after the first task, and after the ninth. The last column counts the
runs whose boundary-9 reading has moved back toward the starting value
compared with boundary 1.

| Measure | Boundary 0 | Boundary 1 | Boundary 9 | Runs moving back |
| --- | --- | --- | --- | --- |
| Effective rank | 5.56 | 1.72 | 2.09 | 10 of 15 |
| srank | 215.6 | 66.1 | 81.1 | 10 of 15 |
| Dormant units, cut-off 0.01 | 0.160 | 0.366 | 0.342 | 15 of 15 |
| Dormant units, cut-off 0.025 | 0.197 | 0.401 | 0.381 | 15 of 15 |
| Dormant units, cut-off 0.1 | 0.290 | 0.480 | 0.469 | 13 of 15 |

The direction is the same everywhere, but the size of the change is not. The
five seeds of one ordering differ from each other about as much as the three
orderings differ from each other (Table A2). For effective rank the mean change
is 3.6 times the spread between runs; for srank it is 2.2 times. A measure
whose mean is only twice its spread cannot separate two models from one run
each, which is the point Appendix A returns to.

Table A2. Change from boundary 0 to boundary 9, as a percent of the boundary 0
value. Each ordering column gives the mean over its five seeds, with the
standard deviation between those seeds in brackets. The last column is the
mean over all fifteen runs divided by their standard deviation.

| Measure | Ordering A | Ordering B | Ordering C | All fifteen | Mean / spread |
| --- | --- | --- | --- | --- | --- |
| Effective rank | -50.3 (18.1) | -63.6 (20.0) | -69.7 (6.0) | -61.2 (17.0) | 3.6 |
| srank | -45.5 (30.9) | -67.9 (31.4) | -76.2 (16.4) | -63.2 (28.5) | 2.2 |
| Dormant units, cut-off 0.01 | +104.8 (44.8) | +121.7 (50.4) | +122.2 (31.2) | +116.2 (40.6) | 2.9 |
| Dormant units, cut-off 0.025 | +86.8 (35.3) | +99.2 (39.6) | +100.1 (25.4) | +95.4 (32.0) | 3.0 |
| Dormant units, cut-off 0.1 | +58.7 (20.0) | +64.2 (21.6) | +64.5 (14.3) | +62.5 (17.7) | 3.5 |

![](figures-v3/v3-fig1-indicator-trajectories.png)

Figure A1. Each measure against the task boundary, on the in_stream images. One
thin grey line per run, with the mean over the fifteen runs in black. The three
cut-offs behave alike, so only 0.01 is drawn.

A seed fixes the starting generator, and the ordering decides only which task
group comes first, so the three orderings of one seed begin from the same
model. The boundary-0 readings confirm this: they agree across the three
orderings to within floating-point noise. The indicator reading at boundary 0
therefore repeats exactly, which matters for Appendix C, where the probe's
reading of the same point does not.

Under this reading protocol the stream does produce the fall these measures
were built to detect. The next section reads the same models through
statistics measured from their own weights and finds the opposite direction,
so the protocol is part of what Table A1 reports. Whether the size of the fall
tracks how well the model can still learn is the question Appendix D puts to it.
The per-run values are in the stored records.

A single-run comparison between orderings is not supported by this spread.
Version 1 of this work reported an ordering effect from one run per ordering.
Table A2 shows that the five seeds of one ordering differ from each other about
as much as the three orderings differ from each other, so that comparison is
withdrawn.

# Appendix B The reading protocol in full

Every reading in the main body is taken through one fixed set of BatchNorm
statistics. The hypernetwork generates the target network's weights
but writes no BatchNorm running statistics, so the trainer keeps the statistics
that a freshly built ResNet50 has, a running mean of zero and a running
variance of one at every layer, and passes those to every forward pass. They
are never updated. This section asks what the indicators read when the
statistics are measured from the model's own weights instead.

Two models are read. The first is the five ordering A streams of Appendix A,
read at boundary 0 and boundary 9 from the saved checkpoints, with the weights
generated from the same fixed reference code used throughout. The second is a
control: an ordinary ResNet50 with one classifier head per group, trained on the
same nine groups in the same order with the same settings, three seeds. The
control shares no component with the hypernetwork, so any behaviour common to
both belongs to the reading and not to the architecture. Nothing is trained for
the hypernetwork half; both routes are forward passes over the same 256
in-stream images. The matched statistics are measured on 2048 training images
from the nine learned groups, with the BatchNorm layers in training mode and a
cumulative average, so the result does not depend on the order of the batches.

Table B1. Each measure at the start and the end of the stream, read twice from
the same models. The fixed columns use the protocol used throughout. The
matched columns use statistics measured from the model's own weights. The
hypernetwork rows are the mean over five seeds of ordering A, the control rows
the mean over three seeds.

| Measure | Network | Fixed, b0 | Fixed, b9 | Matched, b0 | Matched, b9 |
| --- | --- | --- | --- | --- | --- |
| Effective rank | Hypernetwork | 5.56 | 2.79 | 106.27 | 143.00 |
| Effective rank | Ordinary ResNet50 | 7.30 | 1.14 | 7.30 | 110.51 |
| srank | Hypernetwork | 215.6 | 120.4 | 248.2 | 248.4 |
| srank | Ordinary ResNet50 | 226.0 | 2.7 | 226.0 | 245.3 |
| Mean row share | Hypernetwork | 0.981 | 0.983 | 0.805 | 0.865 |
| Mean row share | Ordinary ResNet50 | 0.982 | 0.960 | 0.982 | 0.709 |
| Dormant units, cut-off 0.01 | Hypernetwork | 0.160 | 0.324 | 0.000 | 0.128 |
| Dormant units, cut-off 0.01 | Ordinary ResNet50 | 0.156 | 0.527 | 0.156 | 0.000 |

The reading route decides the direction of travel, not only its size. Read
through the fixed statistics the hypernetwork's effective rank falls from 5.56
to 2.79 over the nine groups, which is the result Appendix A reports. Read
through matched statistics the same five checkpoints rise from 106.27 to 143.00.
The control behaves the same way, falling from 7.30 to 1.14 on the fixed route
and rising from 7.30 to 110.51 on the matched one. The two routes agree at
boundary 0 for the control, because a network that has not trained yet has not
moved its statistics, and they separate from the first group onward.

The control is learning throughout the stream that produces its fall. Its fixed
reading of 1.14 at boundary 9 is lower than any value the hypernetwork reaches,
and it is taken from a network that reaches the accuracy in Table B6 on each
group at the moment that group is learned. An indicator that reports a severe
loss on a network with no loss to report is not measuring the quantity it is
read for.

srank behaves the same way and reaches a floor. On the fixed route the control
falls from 226.0 to 2.7, so it uses three of the 256 available directions. On
the matched route it ends at 245.3. The hypernetwork's matched readings sit at
248.2 and 248.4, which is close enough to the 256 ceiling that this measure
cannot separate a model that is flat from one that is saturated, and no claim is
made from the matched srank column.

The dormant-unit fraction gives the same verdict on the fixed route and a
different result on the matched one. On the fixed route the control rises from
0.156 to 0.527, a larger rise than the hypernetwork's 0.160 to 0.324, so the
reported rise is reproduced more strongly by the network that has no problem.
On the matched route the two architectures move in opposite directions. The
control starts at 0.156 and reads 0.000 from the first group onward in all three
seeds and at all nine boundaries. The hypernetwork starts at 0.000 in all five
seeds and ends at 0.128, with per-seed values between 0.109 and 0.181. Ordinary
sequential training removes every dormant unit in this setting, and generating
the weights from a task code creates them. The control's floor at zero means
this comparison establishes a difference in direction and not a difference in
size.

The mean row share is the one measure whose matched readings separate the two
architectures by level as well. Both networks read about 0.98 on the fixed
route at both boundaries, which is the shared-direction result of Appendix D.
On the matched route the hypernetwork rises from 0.805 to 0.865 while the
control falls from 0.982 to 0.709.

Table B2 locates where the fall on the fixed route happens. The 256 by 2048
feature matrix is read at the classifier's input, which in a ResNet50 is the
output of the last ReLU, so every value in it is non-negative. If the fixed
statistics shift the whole signal upward, the ReLU removes little, every image
gives a similar row, and effective rank falls for that reason alone. The same
matrix was therefore read one step earlier, at the input of the last ReLU, where
negative values are still present.

Table B2. Effective rank and mean row share on the fixed route, read at the input
of the last ReLU and at its output, with the share of values the ReLU sets to
zero. Same models and same images as Table B1.

| Measure | Network | Before, b0 | Before, b9 | After, b0 | After, b9 |
| --- | --- | --- | --- | --- | --- |
| Effective rank | Hypernetwork | 6.17 | 2.47 | 5.56 | 2.79 |
| Effective rank | Ordinary ResNet50 | 8.34 | 1.10 | 7.30 | 1.14 |
| Mean row share | Hypernetwork | 0.981 | 0.984 | 0.981 | 0.983 |
| Mean row share | Ordinary ResNet50 | 0.982 | 0.960 | 0.982 | 0.960 |
| Share the ReLU sets to zero | Hypernetwork | 0.331 | 0.492 | . | . |
| Share the ReLU sets to zero | Ordinary ResNet50 | 0.338 | 0.751 | . | . |

The fall is already complete before the ReLU runs. The hypernetwork reads 2.47
before the ReLU against 2.79 after it, and the control reads 1.10 against 1.14.
The mean row share is unchanged by the ReLU to three decimals on both networks
at both boundaries. The ReLU is nevertheless doing work, removing between 0.331
and 0.751 of all values, so this is not a case of a nonlinearity with nothing to
remove. The collapse therefore occurs in the linear part of the network. Where in that
part, and which half of the BatchNorm operation is responsible, are the two
questions the rest of this section answers.

The error the protocol makes grows with depth. Each BatchNorm layer is told to
divide by a spread of one. The measured spread at boundary 9 runs from 0.838 to
5.752 across the 53 layers, so most layers pass their signal on several times
larger than they should, and the next layer repeats the mistake. Table B3 reads
the average size of the values at the output of each of the 16 blocks, under the
fixed route and under matched statistics, at the end of the stream.

Table B3. The average size of the values at each block output, on the five
ordering A models at boundary 9, before any averaging over the picture. The
fixed column is the study's protocol and the measured column uses statistics
taken from the model's own weights.

| Block | Channels | Fixed | Measured |
| --- | --- | --- | --- |
| layer1.0 | 256 | 11.87 | 1.203 |
| layer1.1 | 256 | 179 | 1.445 |
| layer1.2 | 256 | 1169 | 1.593 |
| layer2.0 | 512 | 1.816e+04 | 1.292 |
| layer2.1 | 512 | 2.692e+05 | 1.600 |
| layer2.2 | 512 | 1.579e+06 | 1.593 |
| layer2.3 | 512 | 8.098e+06 | 1.452 |
| layer3.0 | 1024 | 1.254e+08 | 1.075 |
| layer3.1 | 1024 | 1.063e+09 | 1.382 |
| layer3.2 | 1024 | 4.058e+09 | 1.654 |
| layer3.3 | 1024 | 3.149e+10 | 1.755 |
| layer3.4 | 1024 | 1.097e+11 | 1.937 |
| layer3.5 | 1024 | 7.898e+11 | 2.120 |
| layer4.0 | 2048 | 4.305e+12 | 0.4835 |
| layer4.1 | 2048 | 2.468e+13 | 0.5234 |
| layer4.2 | 2048 | 1.090e+14 | 0.4676 |

The fixed route grows by roughly a factor of ten at each block and ends about
thirteen orders of magnitude above where it started. The measured route stays
near one across all sixteen. At boundary 0 the same fixed column ends at
2.968e4 rather than 1.090e14, so the error is small on an untrained generator
and grows as the weights travel. That is the direct explanation for why the
reading gets worse over the stream.

The rank does not follow the same shape. On the fixed route at boundary 9
effective rank reads 5.135 at the first block, 2.337 at the second, and 2.789 at
the sixteenth. Mean row share reads 0.983 at the first block and 0.983 at the
sixteenth. Almost all of the collapse is present after one block, and fifteen
further blocks of a growing scale do not lower it. So the growth in Table B3
explains the size of the values and not the rank, and the two have to be
reported separately.

A BatchNorm layer performs two operations, subtracting a running mean and
dividing by the square root of a running variance, and the protocol supplies a
wrong value for each. Table B4 separates them. The same five models are read
four ways: with the protocol's values for both, with the measured mean and the
protocol's variance, with the protocol's mean and the measured variance, and
with both measured. The last column gives each arm's recovery as a share of the
distance between the protocol reading and the fully measured one at boundary 9.

Table B4. Effective rank at the classifier under each set of BatchNorm
statistics, averaged over the five ordering A models, on the in_stream images.

| Statistics supplied | Boundary 0 | Boundary 9 | Share of the gap at boundary 9 |
| --- | --- | --- | --- |
| Mean 0, variance 1 (the protocol) | 5.563 | 2.789 | . |
| Measured mean, variance 1 | 63.615 | 3.093 | 0.2 |
| Mean 0, measured variance | 32.960 | 8.860 | 4.3 |
| Both measured | 106.274 | 143.001 | 100.0 |

Neither operation accounts for the result on its own. At boundary 9 the gap
between the protocol reading and the measured one is 140.212. Correcting only
the subtraction recovers 0.2 percent of it and correcting only the division
recovers 4.3 percent, while correcting both recovers all of it. The failure is
therefore not attributable to either operation separately.

The two operations are close to separable at the start of the stream and are not
separable at the end. At boundary 0 the same two corrections recover 57.6 and
27.2 percent of a gap of 100.711, which sums to 84.8 percent; at boundary 9 they
sum to 4.5 percent. The measured statistics give the same account from the other
side, with spreads running 0.362 to 2.358 at boundary 0 and 0.838 to 5.752 at
boundary 9.

Each single correction also introduces a second fault. Correcting the division
alone brings the average size of the values at the last block down from 1.090e14
to 1.969, close to the measured arm's 0.4676, but raises mean row share to
0.993, the highest of the four arms, because the offset that is never subtracted
is then the largest remaining structure in the signal. Correcting the
subtraction alone lowers mean row share to 0.964 and raises the average size to
1.674e14. Neither arm is a partial repair of the protocol.

The practical consequence is the one a reader is most likely to need. A
correction to one half of the protocol, such as centring the features before
measuring, does not restore the reading at the end of a stream.

One limit is stated rather than tested. On the fixed route the values reach the
order of 1e14 in a single-precision forward pass. That is far inside the range
of the format, and the rank arithmetic itself is carried out in double
precision, so no overflow occurs. How much relative precision remains at that
magnitude has not been measured. The measured arm, on which every comparison in
this section rests, stays near one throughout.

One further choice is specific to a hypernetwork and is settled by the same
result. The generator does not hold one set of weights. It writes a set from
whatever 32-number task code it is handed, so a different code gives a
different ResNet50 from the same generator. An instruction to read the rank of
such a model is incomplete until the code is named. Every reading in this paper
uses one fixed reference code, drawn from seed 20261030 and never trained, so
that a change between boundaries is a change in the generator and not in the
code.

Table B5 asks what that choice costs. The same models are read a second time
through the nine codes each stream actually trained, averaged over the nine.

Table B5. Effective rank at the classifier at boundary 9, read through the fixed
reference code and through the codes each stream trained. The five-stream rows
are ordering A and are the same models and images as Tables B1 to B4. The
fifteen-stream rows are all three orderings, and the matched route was not run
on them. The last column is the sum over the 53 BatchNorm layers of the base-10
logarithm of the measured spread, which is zero if the protocol's assumption
holds at every layer.

| Streams | Code | Fixed statistics | Matched statistics | Total log spread |
| --- | --- | --- | --- | --- |
| Five, ordering A | reference | 2.789 | 143.001 | 20.42 |
| Five, ordering A | trained, averaged | 3.837 | 137.542 | 13.17 |
| Fifteen, all orderings | reference | 2.095 | . | . |
| Fifteen, all orderings | trained, averaged | 2.792 | . | . |

Under the fixed statistics the code matters. A trained code reads 3.837 against
the reference code's 2.789, a ratio of 1.38, on the same model and the same
images. The fifteen-stream rows give the same ratio, 2.792 against 2.095. The
two subsets differ because ordering A falls least of the three (Table A2), and
the fifteen-stream reference value is the one Table A1 reports.

Under matched statistics the code stops mattering. The same comparison reads
137.542 against 143.001, a ratio of 0.96. When each generated network is read
through statistics that match it, which code produced the weights barely
changes the answer.

The reason is that a trained code sits closer to what the protocol assumes. The
last column of Table B5 adds up, across the 53 BatchNorm layers, how far each
layer's measured spread is from the assumed value of one, on a logarithmic
scale so that layers above and below one do not cancel. The reference code
reaches 20.42 and the trained codes 13.17. Per layer the average distance is
0.393 against 0.268. The protocol is wrong about both, and it is less wrong
about a code the generator has trained, which is why that code is penalised
less. This quantity is a measure of the mismatch and not the growth itself: the
spreads are measured along the matched path, so their sum does not predict the
size of the values in Table B3.

The code sensitivity is therefore not a second trap. It is the fixed-statistics
result seen from another angle, and a reader who corrects the statistics has
corrected it too. What survives is the reporting rule: a rank level published
for a hypernetwork is a property of the model and the code together, so the
code has to be stated for the number to be comparable with anyone else's.

Table B6. Accuracy of the control network, to be read from the run log.

# Appendix C The direct probe

The probe asks the model the question the indicators only guess at: give it a
task it has never seen, let it train on that task with a fixed budget, and see
how well it does. A copy of the whole model is taken first and put back
afterwards, so the measurement cannot change the model it measures. Inside
that copy both the generator and a fresh 32-number code for the new task
train; neither is frozen. The probe was run on the held-out group of twenty
classes at the start of each stream and again at the end, eleven times at each
point.

This makes the boundary-0 reading the comparison the plasticity literature
uses. At boundary 0 the generator is still at its random starting values, so
the question the two endpoints put is whether a model that has learned nine
tasks fits a new one better or worse than a freshly started model given the
same budget. A loss of plasticity is the case where it does worse.

Every run ends better than it started (Table C1, Figure C1). Accuracy goes from
26.20% before any task to 44.08% after nine, a gain of 17.87 points against a
chance level of 5%. The smallest gain in the fifteen runs is 15.90 points and
the largest is 20.49. The spread between runs is small next to the gain
itself.

Table C1. Probe accuracy on the held-out group, averaged over the fifteen runs
and over the eleven repetitions at each point.

| Point in the stream | Probe accuracy | Gain over chance |
| --- | --- | --- |
| Boundary 0, before any task | 26.20% | 21.20 points |
| Boundary 9, after nine tasks | 44.08% | 39.08 points |

The gain is far larger than the measurement can produce on its own. Each run
carries its own floor, built from the spread of its eleven repetitions at
boundary 0 and read as the smallest change that would survive that spread at
the 5% level. The largest floor in the fifteen runs is 2.72 points, and the
smallest gain is 15.90 (Table C2). No run comes close to its floor, so the rise
is not a reading error.

Table C2. The gain for each run against its own noise floor. The floor is
1.96 s sqrt(1/11 + 1/11), where s is the standard deviation of the eleven
boundary-0 repetitions for that run.

| Run | Boundary 0 | Boundary 9 | Gain | Floor | Gain clears floor |
| --- | --- | --- | --- | --- | --- |
| A0 | 26.13% | 43.72% | +17.59 | 1.70 | yes |
| A1 | 27.25% | 43.65% | +16.40 | 2.72 | yes |
| A2 | 26.06% | 43.75% | +17.68 | 2.10 | yes |
| A3 | 26.55% | 42.76% | +16.22 | 1.26 | yes |
| A4 | 26.78% | 43.90% | +17.12 | 2.07 | yes |
| B0 | 25.50% | 42.39% | +16.89 | 1.62 | yes |
| B1 | 26.35% | 43.81% | +17.45 | 1.91 | yes |
| B2 | 26.15% | 44.24% | +18.09 | 1.81 | yes |
| B3 | 26.21% | 44.03% | +17.82 | 2.63 | yes |
| B4 | 26.05% | 45.28% | +19.24 | 1.60 | yes |
| C0 | 27.41% | 43.31% | +15.90 | 1.67 | yes |
| C1 | 25.27% | 44.45% | +19.18 | 1.82 | yes |
| C2 | 24.95% | 45.44% | +20.49 | 1.48 | yes |
| C3 | 26.39% | 45.36% | +18.97 | 2.31 | yes |
| C4 | 26.01% | 45.04% | +19.03 | 2.27 | yes |

![](figures-v3/v3-fig2-probe-endpoints.png)

Figure C1. Probe accuracy for each run. The open circle is boundary 0 and the
filled circle is boundary 9. The dotted line is chance for a twenty-class
group.

One caveat belongs here. The three orderings of a seed start from the same
model, so their three boundary-0 readings should agree, and they do not: for
seed 0 they are 26.13%, 25.50% and 27.41%. The largest such gap is about two
points, which is roughly the size of a noise floor and well under the gain, so
it does not touch the result of this section. Appendix C reports the spread
in full and states what is known about its cause.

This is the result the indicators have to be judged against. Over the nine
tasks every indicator said the model was losing the ability to learn, and the
direct measurement says it fits a new task better, in all fifteen runs, by
more than ten times the largest floor. The probe trains the generator, so part
of that gain is ordinary transfer between tasks drawn from one dataset, and
Section 5 of the main body lists this among the limits. Transfer does not explain the
disagreement, though: the indicators are read as a statement about the model
itself, and the model they call least able is the one that learns best. Appendix D takes the weaker question: set
the direction aside, and ask only whether the runs that fell furthest are the
runs that gained least.

# Appendix D Correlations in full

Each of the fifteen runs gives a pair of numbers: how far an indicator moved
over the nine tasks, and how far probe accuracy moved over the same span. Both
vary widely. The effective-rank fall runs from 29.8% to 80.6%, a factor of
2.7, and the probe gain from 15.90 to 20.49 points (Table D1). If the fall
reports the ability to learn, the runs that fell furthest should be the runs
that gained least.

Table D1. Change from boundary 0 to boundary 9 on the in_stream images, across
the fifteen runs. Indicator changes are a percent of the boundary 0 value; the
probe change is in accuracy points.

| Quantity | Mean | Lowest | Highest |
| --- | --- | --- | --- |
| Effective rank | -61.20% | -80.58% | -29.81% |
| srank | -63.19% | -97.92% | -15.04% |
| Dormant units, cut-off 0.01 | +116.22% | +50.90% | +174.51% |
| Dormant units, cut-off 0.025 | +95.37% | +44.88% | +142.61% |
| Dormant units, cut-off 0.1 | +62.45% | +34.67% | +86.63% |
| Probe accuracy | +17.87 | +15.90 | +20.49 |

They are not. The correlation between the effective-rank fall and the probe
gain is -0.054, which accounts for 0.3% of the spread in the gain (Table D2,
Figure 1). srank gives -0.178 and the three dormant-unit cut-offs give +0.112
to +0.129.

Each measure also predicts a sign. A run whose rank fell less should have more
ability left, so the rank measures predict a positive correlation; dormant
units are read the other way and predict a negative one. All five come out
with the wrong sign, and the largest of them accounts for 3.2% of the spread.

Table D2. Correlation between each indicator's change and the probe's change,
over the fifteen runs. The last column repeats the test on the boundary-9
values themselves rather than on the changes.

| Measure | Predicted sign | r | r squared | r, end values |
| --- | --- | --- | --- | --- |
| Effective rank | positive | -0.054 | 0.3% | -0.165 |
| srank | positive | -0.178 | 3.2% | -0.212 |
| Dormant units, cut-off 0.01 | negative | +0.129 | 1.7% | +0.164 |
| Dormant units, cut-off 0.025 | negative | +0.125 | 1.6% | +0.162 |
| Dormant units, cut-off 0.1 | negative | +0.112 | 1.3% | +0.139 |

Both quantities in that test are differences, and a difference inherits the
noise of both of its endpoints. The last column of Table D2 therefore sets the
differences aside and asks the plainer question: among the fifteen finished
models, does the one with the higher indicator reading fit the unseen task
better? It does not. Effective rank gives -0.165 and srank -0.212 where a
positive value was predicted, and the three dormant-unit cut-offs give +0.139
to +0.164 where a negative value was predicted. The sign is wrong in the same
way, so the result is not an artifact of dividing by a starting value.

A failed correlation means little if the gain it fails to predict is mostly
measurement noise, and the eleven repetitions at each point let the two be
separated. The gain's spread across the fifteen runs is 1.300 points. The
repetitions account for 0.953 of that, leaving 0.883 as real difference
between models, which is 46.2% of the spread by variance. An outcome measured
that well caps any predictor at a correlation of 0.680, the square root of
that share. That ceiling covers the change test only; the end-value test in
the last column does not take a difference and so is not subject to it. It
does not need to be: the largest value anywhere in Table D2 is 0.212, and at
fifteen runs a correlation needs 0.514 to be separable from zero at the 5%
level.

Neither the size of an indicator's change nor its final value therefore says
anything usable about how well the model fits an unseen task, and the little
relation there is runs the wrong way. Appendix C asked the indicators to get
a direction right. This section asks less of them: set the direction aside and
only rank fifteen runs against each other. They fail that too. Appendix A
gives the per-run values.

# Appendix E The unlearning step

Appendices A to D ask whether the indicators track ordinary change. This
section asks an easier question. After the ninth group, each stream performs an
unlearning step that destroys one group it has already learned. The destroyed
group reads 5.00 percent afterwards in all fifteen streams, which is exactly
chance for a twenty-class task, so the damage is complete rather than partial.
If the indicators report anything about the health of a model, a model with one
group destroyed should read worse than the same model one step earlier.

The comparison is between boundary 9 and boundary 10 of the same stream. These
differ by the unlearning step alone. Both readings are taken on the same
in-stream images through the same protocol, so nothing but the model changes.

Table E1 also gives the size of an ordinary learning step, for scale. That number
is the average absolute change in the indicator from one boundary to the next
over the stream, excluding the first step, because Appendix A shows the first
step is not an ordinary one. Without it there is no way to judge whether the
movement at the unlearn is large or small.

Table E1. The unlearning step, averaged over the fifteen streams, on the
in-stream images. The last column counts streams in which the indicator moved in
the direction the measure reports as healthy. The final row of each block is the
average absolute change across one ordinary learning step, over the 120 steps
after the first.

| Quantity | Boundary 9 | Boundary 10 | Change | Healthy direction |
| --- | --- | --- | --- | --- |
| Accuracy of the destroyed group, percent | . | 5.00 | . | . |
| Effective rank | 2.095 | 3.675 | +1.580 | 15 of 15 |
| Effective rank, one ordinary learning step | . | . | 0.162 | . |
| srank | 81.1 | 154.7 | +73.6 | 15 of 15 |
| srank, one ordinary learning step | . | . | 10.025 | . |

Both rank measures rise at the unlearning step, in every stream. A rise is the
direction these measures report as healthy. So on a model whose accuracy on one
group has just been reduced to chance, both indicators read better than they did
before the damage.

The movement is also large beside ordinary learning. Effective rank moves 1.580
at the unlearn against 0.162 for a typical learning step, which is 9.8 times as
far and in the opposite direction. srank moves 73.6 against 10.025, which is 7.3
times as far. A measure that barely responds to eight groups of learning and then
moves roughly ten times further, in the healthy direction, when a group is
destroyed is not reporting the health of the model.

The accuracy of the destroyed group before the unlearning step is not recorded in
the rescaling-control file, which is why that cell is empty. It is not needed for
the comparison: a reading of exactly chance establishes that the group is gone.

Appendix F asks what the indicators are responding to instead.

# Appendix F The contribution of weight scale

Both rank measures divide the singular values of the feature matrix by their
sum, so multiplying the whole feature matrix by a constant cannot change either
of them. That property is usually read as meaning the measures are free of
scale. It does not follow, because the property holds of the feature matrix and
the weights sit upstream of it.

A residual block adds its input to a branch of several weighted layers.
Multiply every generated weight by a constant and the branch changes by a power
of that constant, set by how many weight layers it contains, while the input
path carries no weights and does not change at all. Their sum is therefore not
the original sum multiplied by any single number. The generated bias terms and
the generated BatchNorm scale and offset break the same symmetry a second time.
So a change in the size of the weights does not reach the features as one
multiplying constant, and a normalisation that only removes such a constant
cannot remove it.

The ReLU is not the reason. For any positive constant c, the ReLU of c times a
value equals c times the ReLU of that value, so a positive rescale passes
through it unchanged.

The unlearning step changes the size of the generated weights. Averaged over the
fifteen streams, the weight norm after the unlearn is 0.757 of its value before.
That is a large change in one step, and it is in the shrinking direction. So the
rise reported in Appendix E has a second possible explanation that has nothing
to do with the representation.

Two interventions separate the two explanations, and they are not equivalent.
The first takes the damaged model and scales its generated weights back up to the
norm they had before the unlearn, which asks how much of the rise survives when
the size is held fixed. The second takes the undamaged model and shrinks its
weights by the same factor the unlearn produced, changing nothing else, which
asks how much of the rise the size alone can produce.

Table F1. The two interventions, averaged over the fifteen streams, on the
in-stream images. The share is the ratio of the averages, not the average of the
per-run ratios, because several runs have a change near zero and dividing by it
gives per-run shares ranging from below minus one to above one.

| Quantity | Effective rank | srank |
| --- | --- | --- |
| Change at the unlearn | +1.580 | +73.6 |
| Change with the weight norm held at its boundary 9 value | +0.754 | +37.7 |
| Change produced by shrinking the undamaged model alone | +1.314 | +53.9 |
| Share carried by the weight norm, first intervention | 52.3 percent | 48.8 percent |
| Share carried by the weight norm, second intervention | 83.2 percent | 73.3 percent |

The weight norm carries a large part of the movement on both measures, and the
two interventions do not agree on how large. On effective rank they give 52.3
percent and 83.2 percent, a gap of about a third of the whole change. On srank
they give 48.8 percent and 73.3 percent. The two paths would agree if the size
and the direction contributed independently. They do not, so there is no single
number for the share and both have to be reported.

What survives the first intervention is still in the unhealthy direction for the
argument of Appendix E. Holding the norm fixed leaves +0.754 on effective rank,
which is 4.7 times the size of an ordinary learning step from Table E1, in 14 of
the 15 streams. So the weight scale does not explain the result away. The reading
still improves after the damage.

Two limits. The interventions change the weight norm by rescaling, which is a
single global factor and not the kind of change training produces, so they bound
the contribution rather than measure it as training would. And the second
intervention moves in the same direction in 11 of 15 streams against 14 of 15 for
the first, so it is the less consistent of the two.

# Appendix G Learning speed on the held-out group

Appendix C reports one number per probe run: the accuracy reached after a fixed
budget of 250 steps. That number mixes two quantities that the loss-of-plasticity
literature keeps apart. The first is speed, meaning how quickly the model fits
the new task. The second is the ceiling, meaning how well it can do once it has
finished. A model that has lost plasticity is expected to be slower, and the
usual evidence for that is a learning curve rather than a single endpoint.

No new run was needed. The probe already recorded four readings on every curve,
at steps 62, 125, 188 and 250, each with the training loss and the test accuracy.
Runs are compared in matched pairs. A boundary 0 run and a boundary 9 run that
share an ordering, a seed, a learning rate and a repetition number started from
the same fresh task code and saw the same batches in the same order, so they
differ only in the model being probed. Pairing removes the spread that comes from
the starting code, which is the largest source of noise in this measurement.

Table G1. Test accuracy on the held-out group at each of the four readings,
averaged over the runs at that boundary, with the paired gap against boundary 0
beneath. Boundaries 0 and 9 use all fifteen streams; boundary 4 is ordering A
only. A star marks a gap larger than its own 95 percent interval.

| Boundary | Step 62 | Step 125 | Step 188 | Step 250 | Runs |
| --- | --- | --- | --- | --- | --- |
| t = 0 | 9.78 | 15.07 | 18.47 | 21.27 | 495 |
| t = 4 | 30.58 | 36.33 | 36.44 | 38.84 | 165 |
| t = 9 | 32.85 | 36.96 | 38.25 | 39.57 | 495 |
| gap, t = 4 minus t = 0 | +20.78* | +21.15* | +17.69* | +17.63* | 165 |
| gap, t = 9 minus t = 0 | +23.07* | +21.89* | +19.78* | +18.29* | 495 |

The gap between the end of the stream and the start is widest at the first
reading and narrows from there. It is +23.07 points at step 62 and +18.29 points
at step 250. That is the shape of a difference in speed. A difference in ceiling
alone would show the opposite shape, with the two curves close early and
separating as they approach their limits.

The first reading also lies outside the range the untrained generator reaches at
all. At step 62 the end-of-stream model is at 32.85 percent, which is higher than
the 21.27 percent the untrained generator reaches after the whole budget of 250
steps. One quarter of the budget at boundary 9 beats the full budget at boundary
0.

The training loss agrees. At boundary 0 it moves from 3.00 to 2.31 over the
budget. At boundary 9 it moves from 1.91 to 0.70. The model late in the stream
starts lower and falls further, so the effect is present in fitting the probe
task and not only in generalising from it.

Table G2 separates speed from the ceiling directly. For each run, its accuracy at
step 62 is divided by its own accuracy at step 250. The result says how much of
that run's own final result was already reached early. Dividing each run by its
own ceiling is what makes this a measure of speed: a model that ends higher while
moving at the same pace gives the same share.

Table G2. Share of a run's own step-250 accuracy already reached at step 62,
averaged over the runs at that boundary, with the paired gap against boundary 0
and its 95 percent interval.

| Boundary | Share at step 62 | Paired gap against t = 0 | 95 percent interval | Runs |
| --- | --- | --- | --- | --- |
| t = 0 | 0.476 | . | . | 495 |
| t = 4 | 0.787 | +0.299 | 0.052 | 165 |
| t = 9 | 0.837 | +0.361 | 0.028 | 495 |

The share rises from 0.476 to 0.837 over the stream, and the paired gain of
+0.361 is more than twelve times its own interval. The model late in the stream
is faster at the held-out group, not only better at it. This is the opposite of
the behaviour a loss of plasticity predicts, measured on the quantity that
literature asks about.

One comparison in this section disagrees with Appendix D and both numbers are
reported. Appendix D finds the accuracy endpoint complete at boundary 4, with no
detectable change from boundary 4 to boundary 9. The speed share does change over
that interval. On ordering A, where boundary 4 was measured, the share rises by
+0.076 from boundary 4 to boundary 9. Counting every repetition as an observation
gives a 95 percent interval of 0.044, and counting each seed once, which is the
conservative treatment because the eleven repetitions of one seed probe the same
two models, gives 0.063. The gain is larger than both intervals, but it is larger
than the conservative one by a small margin, and it rests on five seeds of a
single ordering. It is reported as evidence that the speed gain continues past the
point where the accuracy endpoint stops, and not as a settled result.

**Script.** `scripts/learning_speed.py`.

# Appendix H Work not yet carried out on the present design

| Measurement | Notebook | Cost to repeat |
| --- | --- | --- |
| Probe at five points along the stream | `40_probe_curve.ipynb` | forks and discards |
| Random-label fitting | `41_matched_random_labels.ipynb` | forks and discards |
| Learning speed of an unseen task | `40_probe_curve.ipynb` data | no new run needed |
| Learning after deliberate damage | 7 November record | one stream |
| Curvature at a trained code | `42_curvature.ipynb` | forks and discards |
| Random-direction control | `39_random_direction_control.ipynb` | trains nothing |
| Effective-rank convergence across seeds | stored records | no new run needed |
| Probe at the middle of the stream | `47_probe_on_the_revised_streams.ipynb` | 495 probe runs |
