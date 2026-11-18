# Rank-based plasticity indicators and directly measured learning ability in a hypernetwork continual learner

Skeleton, 15 November 2026. Title, abstract and section plan only. Each
section below states what it will contain, which experiment supplies it, and
which notebook produced the data. No results are written yet.

Two alternative titles are recorded in the paper plan.

## Abstract

A network that learns tasks in sequence can become worse at learning the next
one. Detecting that loss matters, but measuring it directly means training the
network on a further task, which is costly enough that it is rarely repeated.
Practitioners therefore rely on measures read from the weights or activations
alone, with no training. Three are in wide use: effective rank, srank, and the
fraction of dormant units. Each is normalized so that overall scale cannot
affect it, which is taken to mean that two readings may be compared directly.

How closely these measures follow the quantity they stand for is largely
untested, because the direct measurement is usually run once and so carries no
error bar of its own. We run it 990 times. Each run is short and works on a
copy that is restored afterwards, so the same point can be measured again and
again and each model gets its own noise floor.

We run fifteen streams of nine 20-class Tiny ImageNet tasks, as three task
orderings with five random seeds each, holding a tenth group of classes out of
training. At the start and the end of every stream we train the model on that
held-out group for a fixed short budget, eleven times at each point, and
record how well it fits.

Across the fifteen runs the size of the indicator change does not account for
the differences in measured learning, and its sign is the opposite of the one
the usual reading predicts. The same holds when the end values are used in
place of the changes. The indicators fall in every run while the model's
measured fit to an unseen task improves in every run. A model with one task deliberately
reduced to chance accuracy moves the indicators in the direction they report
as healthy, and holding the weight scale fixed removes about half of that
movement. We also retract an earlier single-run result on task ordering, which
the five-seed measurement does not support.

In this setting, a reading of these measures is not evidence about the ability
to learn. The weight-scale result shows where the failure comes from: both
rank measures reach the generated weights along a path that their own scale
normalization does not cover, so a comparison between models of different
weight scale is in part a comparison of those scales.

---

# 1 Introduction

**Contents.** The problem of diagnosing lost plasticity without training.
The implicit promise of a scale-normalized measure. Why a hypernetwork is a
useful test setting: the generator can be frozen and a task code trained in
its place, which gives a direct measurement at a cost low enough to repeat.
Statement of contributions.

**Contributions to list.** One per result that is measured on the present
design. Draft wording to be settled after Section 5 is written, so the claims
and their evidence are fixed first.

**Source.** No new data. Written from Sections 5 and 6.

**Status.** To write last.

---

# 2 Related work

**Contents.** Loss of plasticity in continual learning, and the correlates
usually named for it. The three measures and their original sources: effective
rank (Roy and Vetterli, 2007), srank (Kumar et al., 2021), dormant units
(Sokar et al., 2023). The unlearning method that supplies the damaged model
(UnCLe). What is new here, which is the controls rather than the measures or
the setting.

**Carry over.** Section 2 of the version 1 draft can be reused with small
changes. The measures and their sources have not changed.

**Source.** Literature only.

**Status.** Mostly carried over from version 1.

---

# 3 Method

## 3.1 Model and task stream

**Contents.** The hypernetwork and the ResNet50 it generates, with parameter
counts. The 32-number task code. The class split: 200 Tiny ImageNet classes in
10 disjoint groups of 20, nine learned and group 9 held out. Chance accuracy
of 5 percent. Why the eight-task, ten-class split used in version 1 was
replaced.

**Source.** Configuration in `uncle/config.py`. Design recorded in the
trajectory page for 12 November 2026.

**Status.** To write. Replaces Section 3.1 of version 1, which describes the
retired design.

## 3.2 Training and the learning-rate sweep

**Contents.** Optimizer, step counts, batch size, and how the learning rate
was chosen. Why a sweep was needed before the streams were run.

**Experiment.** Learning-rate sweep.

**Notebook.** `43_learning_rate_sweep.ipynb`.

**Status.** To write. Belongs in Method, not in Results.

## 3.3 Reading protocol

**Contents.** The four choices that make a reading describe the generator
rather than whichever task is loaded: one reference code (seed 20261030),
fixed BatchNorm statistics with the model in evaluation mode, one layer (the
2048 features at the input to the final classifier), and three fixed image
sets of 256 images each. What the three image sets are and why they differ in
distance from the training distribution.

Two of those four choices change the value that is read, not only which model
is read. The fixed BatchNorm statistics decide the direction of every result in
Section 5.1, and the choice of code shifts the level. Section 5.2 measures
both. This section should therefore state plainly that a rank value reported
for a hypernetwork is a property of the model and the code together, and that
the reference code is a choice made to keep the comparison across boundaries
honest rather than a neutral default.

**Carry over.** Section 3.2 of version 1 is correct as written and applies
unchanged to the present design.

**Status.** Carried over from version 1.

## 3.4 Measures

**Contents.** Definition of effective rank, srank with delta = 0.01, and the
dormant-unit fraction at cut-offs 0.01, 0.025 and 0.1. The normalization each
one applies, and what that normalization is taken to guarantee.

**Carry over.** Section 3.3 of version 1.

**Status.** Carried over from version 1.

## 3.5 The direct probe

**Contents.** The procedure: copy the model, draw a fresh 32-number code for
the held-out group, train the generator and that code together for a fixed
number of steps, record accuracy, and restore the model. What trains and what
does not, stated plainly, because the paper's claim depends on it. The
learning-rate choice within the probe, made on a validation split. The eleven
repetitions and the noise floor taken from their spread. The two control arms
that exist in the code and were not used here: `code_only`, which freezes the
generator, and `train_code=False`, which freezes the code.

**Experiment.** Experiment 2 in the explainer.

**Notebook.** `47_probe_on_the_revised_streams.ipynb`. Earlier version on the
retired design: `38_matched_probe.ipynb`.

**Status.** To write. New section; version 1 had no equivalent.

## 3.6 The unlearning step

**Contents.** What UnCLe does to the generated parameters of a forgotten task,
the 100-step burn-in, and how the resulting model is read. The rescaling
control: four readings per pair, so that the part of the indicator movement
caused by weight scale can be separated from the rest.

**Experiment.** Experiment 4 in the explainer.

**Notebook.** `46_rescaling_control.ipynb`.

**Status.** To write.

---

# 4 Experimental design

## 4.1 Orderings and seeds

**Contents.** The three orderings, with the table of partition indices. Which
six groups are fixed and which three rotate. Five seeds per ordering, fifteen
streams in total. Why five seeds, and what the 13 November spread reading
showed about the number of runs a single comparison needs.

**Experiments.** The stream runs.

**Notebooks.** `44_order_A_five_seeds.ipynb`, `45_orders_B_and_C.ipynb`.

**Table to include.** The three orderings by partition index. Carried over in
form from Table 1 of version 1, with new values.

**Status.** To write.

## 4.2 What each measurement contributes

**Contents.** A short table mapping each result section to the experiment and
notebook behind it, and to the design it was measured on. This is the section
that keeps the retired-design results visibly separate from the present ones.

**Status.** To write.

---

# 5 Results

## 5.1 The indicators move at the first task and then hold

Fifteen streams were run: three orderings of the nine task groups, five seeds
each. All five measures were read at every task boundary on the same 256
in-stream images. Every run moves the way a loss of plasticity would predict.
Effective rank and srank fall in all fifteen, and the dormant-unit fraction
rises in all fifteen at each of the three cut-offs.

Nearly all of that movement happens at the first task (Table 2, Figure 1).
Averaged over the fifteen runs, effective rank falls from 5.56 to 1.72 after
one task of the nine and ends at 2.09. srank falls from 215.6 to 66.1 and ends
at 81.1. The dormant fraction at cut-off 0.01 rises from 0.160 to 0.366 and
ends at 0.342. Each measure passes its boundary-9 value after a single task
and then moves back a little, and the last column of Table 2 counts how many
runs move back that way.

Table 2. Each measure averaged over the fifteen runs, at the start of the
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
orderings differ from each other (Table 3). For effective rank the mean change
is 3.6 times the spread between runs; for srank it is 2.2 times. A measure
whose mean is only twice its spread cannot separate two models from one run
each, which is the point Section 5.7 returns to.

Table 3. Change from boundary 0 to boundary 9, as a percent of the boundary 0
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

Figure 1. Each measure against the task boundary, on the in_stream images. One
thin grey line per run, with the mean over the fifteen runs in black. The three
cut-offs behave alike, so only 0.01 is drawn.

A seed fixes the starting generator, and the ordering decides only which task
group comes first, so the three orderings of one seed begin from the same
model. The boundary-0 readings confirm this: they agree across the three
orderings to within floating-point noise. The indicator reading at boundary 0
therefore repeats exactly, which matters for Section 5.8, where the probe's
reading of the same point does not.

Under this reading protocol the stream does produce the fall these measures
were built to detect. The next section reads the same models through
statistics measured from their own weights and finds the opposite direction,
so the protocol is part of what Table 2 reports. Whether the size of the fall
tracks how well the model can still learn is the question Section 5.4 puts to it.
Appendix A gives the per-run values.

## 5.2 The reading protocol and the indicator values

Every reading in this paper is taken through one fixed set of BatchNorm
statistics. The hypernetwork generates the target network's weights
but writes no BatchNorm running statistics, so the trainer keeps the statistics
that a freshly built ResNet50 has, a running mean of zero and a running
variance of one at every layer, and passes those to every forward pass. They
are never updated. This section asks what the indicators read when the
statistics are measured from the model's own weights instead.

Two models are read. The first is the five ordering A streams of Section 5.1,
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

Table 4. Each measure at the start and the end of the stream, read twice from
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
to 2.79 over the nine groups, which is the result Section 5.1 reports. Read
through matched statistics the same five checkpoints rise from 106.27 to 143.00.
The control behaves the same way, falling from 7.30 to 1.14 on the fixed route
and rising from 7.30 to 110.51 on the matched one. The two routes agree at
boundary 0 for the control, because a network that has not trained yet has not
moved its statistics, and they separate from the first group onward.

The control is learning throughout the stream that produces its fall. Its fixed
reading of 1.14 at boundary 9 is lower than any value the hypernetwork reaches,
and it is taken from a network that reaches the accuracy in Table 9 on each
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
route at both boundaries, which is the shared-direction result of Section 5.4.
On the matched route the hypernetwork rises from 0.805 to 0.865 while the
control falls from 0.982 to 0.709.

Table 5 locates where the fall on the fixed route happens. The 256 by 2048
feature matrix is read at the classifier's input, which in a ResNet50 is the
output of the last ReLU, so every value in it is non-negative. If the fixed
statistics shift the whole signal upward, the ReLU removes little, every image
gives a similar row, and effective rank falls for that reason alone. The same
matrix was therefore read one step earlier, at the input of the last ReLU, where
negative values are still present.

Table 5. Effective rank and mean row share on the fixed route, read at the input
of the last ReLU and at its output, with the share of values the ReLU sets to
zero. Same models and same images as Table 4.

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
larger than they should, and the next layer repeats the mistake. Table 6 reads
the average size of the values at the output of each of the 16 blocks, under the
fixed route and under matched statistics, at the end of the stream.

Table 6. The average size of the values at each block output, on the five
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
further blocks of a growing scale do not lower it. So the growth in Table 6
explains the size of the values and not the rank, and the two have to be
reported separately.

A BatchNorm layer performs two operations, subtracting a running mean and
dividing by the square root of a running variance, and the protocol supplies a
wrong value for each. Table 7 separates them. The same five models are read
four ways: with the protocol's values for both, with the measured mean and the
protocol's variance, with the protocol's mean and the measured variance, and
with both measured. The last column gives each arm's recovery as a share of the
distance between the protocol reading and the fully measured one at boundary 9.

Table 7. Effective rank at the classifier under each set of BatchNorm
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

Table 8 asks what that choice costs. The same models are read a second time
through the nine codes each stream actually trained, averaged over the nine.

Table 8. Effective rank at the classifier at boundary 9, read through the fixed
reference code and through the codes each stream trained. The five-stream rows
are ordering A and are the same models and images as Tables 4 to 7. The
fifteen-stream rows are all three orderings, and the matched route was not run
on them.

| Streams | Code | Fixed statistics | Matched statistics |
| --- | --- | --- | --- |
| Five, ordering A | reference | 2.789 | 143.001 |
| Five, ordering A | trained, averaged | 3.837 | 137.542 |
| Fifteen, all orderings | reference | 2.095 | . |
| Fifteen, all orderings | trained, averaged | 2.792 | . |

Under the fixed statistics the code matters. A trained code reads 3.837 against
the reference code's 2.789, a ratio of 1.38, on the same model and the same
images. The fifteen-stream rows give the same ratio, 2.792 against 2.095. The
two subsets differ because ordering A falls least of the three (Table 3), and
the fifteen-stream reference value is the one Table 2 reports.

Under matched statistics the code stops mattering. The same comparison reads
137.542 against 143.001, a ratio of 0.96. When each generated network is read
through statistics that match it, which code produced the weights barely
changes the answer.

The code sensitivity is therefore not a second trap. It is the fixed-statistics
result seen from another angle, and a reader who corrects the statistics has
corrected it too. What survives is the reporting rule: a rank level published
for a hypernetwork is a property of the model and the code together, so the
code has to be stated for the number to be comparable with anyone else's.

Table 9. Accuracy of the control network, to be read from the run log.

**Table to include.** Three rows, one per seed, with the mean accuracy over the
nine groups at the moment each group is learned, and the mean accuracy of those
same groups at the end of the stream. The control has no preservation term, so
the second column is expected to be much lower; that difference is a property of
the training objective and is not a plasticity measurement.

**Experiment.** The 16 November record, the ordinary-network and buffer-route
results.

**Notebooks.** `51_an_ordinary_network_on_the_same_stream.ipynb`,
`52_both_buffer_routes_on_the_hypernetwork.ipynb`.

**Status.** Written except for Table 9. Ordering A only throughout. The
control has three seeds against the hypernetwork's five, and the depth and
four-arm readings in Tables 6 and 7 are on the hypernetwork alone. The
matched route has not been read through the trained codes on orderings B and C,
which is why two cells of Table 8 are empty.

---

## 5.3 Measured learning ability rises over the same stream

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

Every run ends better than it started (Table 10, Figure 2). Accuracy goes from
26.20% before any task to 44.08% after nine, a gain of 17.87 points against a
chance level of 5%. The smallest gain in the fifteen runs is 15.90 points and
the largest is 20.49. The spread between runs is small next to the gain
itself.

Table 10. Probe accuracy on the held-out group, averaged over the fifteen runs
and over the eleven repetitions at each point.

| Point in the stream | Probe accuracy | Gain over chance |
| --- | --- | --- |
| Boundary 0, before any task | 26.20% | 21.20 points |
| Boundary 9, after nine tasks | 44.08% | 39.08 points |

The gain is far larger than the measurement can produce on its own. Each run
carries its own floor, built from the spread of its eleven repetitions at
boundary 0 and read as the smallest change that would survive that spread at
the 5% level. The largest floor in the fifteen runs is 2.72 points, and the
smallest gain is 15.90 (Table 11). No run comes close to its floor, so the rise
is not a reading error.

Table 11. The gain for each run against its own noise floor. The floor is
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

Figure 2. Probe accuracy for each run. The open circle is boundary 0 and the
filled circle is boundary 9. The dotted line is chance for a twenty-class
group.

One caveat belongs here. The three orderings of a seed start from the same
model, so their three boundary-0 readings should agree, and they do not: for
seed 0 they are 26.13%, 25.50% and 27.41%. The largest such gap is about two
points, which is roughly the size of a noise floor and well under the gain, so
it does not touch the result of this section. Section 5.8 reports the spread
in full and states what is known about its cause.

This is the result the indicators have to be judged against. Over the nine
tasks every indicator said the model was losing the ability to learn, and the
direct measurement says it fits a new task better, in all fifteen runs, by
more than ten times the largest floor. The probe trains the generator, so part
of that gain is ordinary transfer between tasks drawn from one dataset, and
Section 7 lists this among the limits. Transfer does not explain the
disagreement, though: the indicators are read as a statement about the model
itself, and the model they call least able is the one that learns best. Section 5.4 takes the weaker question: set
the direction aside, and ask only whether the runs that fell furthest are the
runs that gained least.

## 5.4 The size of the fall does not track measured learning

Each of the fifteen runs gives a pair of numbers: how far an indicator moved
over the nine tasks, and how far probe accuracy moved over the same span. Both
vary widely. The effective-rank fall runs from 29.8% to 80.6%, a factor of
2.7, and the probe gain from 15.90 to 20.49 points (Table 12). If the fall
reports the ability to learn, the runs that fell furthest should be the runs
that gained least.

Table 12. Change from boundary 0 to boundary 9 on the in_stream images, across
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
gain is -0.054, which accounts for 0.3% of the spread in the gain (Table 13,
Figure 3). srank gives -0.178 and the three dormant-unit cut-offs give +0.112
to +0.129.

Each measure also predicts a sign. A run whose rank fell less should have more
ability left, so the rank measures predict a positive correlation; dormant
units are read the other way and predict a negative one. All five come out
with the wrong sign, and the largest of them accounts for 3.2% of the spread.

Table 13. Correlation between each indicator's change and the probe's change,
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

Figure 3. Change in effective rank against change in probe accuracy, one point
per run. Error bars are each run's own standard error on the probe change,
from eleven repetitions. The dashed line is the least-squares fit. The runs
furthest left fell most; they did not gain least.

Both quantities in that test are differences, and a difference inherits the
noise of both of its endpoints. The last column of Table 13 therefore sets the
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
does not need to be: the largest value anywhere in Table 13 is 0.212, and at
fifteen runs a correlation needs 0.514 to be separable from zero at the 5%
level.

Neither the size of an indicator's change nor its final value therefore says
anything usable about how well the model fits an unseen task, and the little
relation there is runs the wrong way. Section 5.3 asked the indicators to get
a direction right. This section asks less of them: set the direction aside and
only rank fifteen runs against each other. They fail that too. Appendix A
gives the per-run values.

## 5.5 Indicator response to a deliberately damaged model

Sections 5.1 to 5.4 ask whether the indicators track ordinary change. This
section asks an easier question. After the ninth group, each stream performs an
unlearning step that destroys one group it has already learned. The destroyed
group reads 5.00 percent afterwards in all fifteen streams, which is exactly
chance for a twenty-class task, so the damage is complete rather than partial.
If the indicators report anything about the health of a model, a model with one
group destroyed should read worse than the same model one step earlier.

The comparison is between boundary 9 and boundary 10 of the same stream. These
differ by the unlearning step alone. Both readings are taken on the same
in-stream images through the same protocol, so nothing but the model changes.

Table 14 also gives the size of an ordinary learning step, for scale. That number
is the average absolute change in the indicator from one boundary to the next
over the stream, excluding the first step, because Section 5.1 shows the first
step is not an ordinary one. Without it there is no way to judge whether the
movement at the unlearn is large or small.

Table 14. The unlearning step, averaged over the fifteen streams, on the
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

Section 5.6 asks what the indicators are responding to instead.

**Experiment.** The rescaling control, boundary 9 against boundary 10.

**Notebook.** `46_rescaling_control.ipynb`.

**Status.** Written. The srank row was read from the stored records on 17
November; earlier records gave its direction only.

## 5.6 The contribution of weight scale

Both rank measures divide the singular values of the feature matrix by their
sum, so multiplying the whole feature matrix by a constant cannot change either
of them. That property is usually read as meaning the measures are free of
scale. It does not follow. The path from the generated weights to the feature
matrix passes through a ReLU, which is not linear, and through BatchNorm
statistics that are held fixed, so a change in the size of the weights does not
reach the features as a single multiplying constant and the normalisation does
not remove it.

The unlearning step changes the size of the generated weights. Averaged over the
fifteen streams, the weight norm after the unlearn is 0.757 of its value before.
That is a large change in one step, and it is in the shrinking direction. So the
rise reported in Section 5.5 has a second possible explanation that has nothing
to do with the representation.

Two interventions separate the two explanations, and they are not equivalent.
The first takes the damaged model and scales its generated weights back up to the
norm they had before the unlearn, which asks how much of the rise survives when
the size is held fixed. The second takes the undamaged model and shrinks its
weights by the same factor the unlearn produced, changing nothing else, which
asks how much of the rise the size alone can produce.

Table 15. The two interventions, averaged over the fifteen streams, on the
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
argument of Section 5.5. Holding the norm fixed leaves +0.754 on effective rank,
which is 4.7 times the size of an ordinary learning step from Table 14, in 14 of
the 15 streams. So the weight scale does not explain the result away. The reading
still improves after the damage.

Two limits. The interventions change the weight norm by rescaling, which is a
single global factor and not the kind of change training produces, so they bound
the contribution rather than measure it as training would. And the second
intervention moves in the same direction in 11 of 15 streams against 14 of 15 for
the first, so it is the less consistent of the two.

**Experiment.** The rescaling control, all four arms.

**Notebook.** `46_rescaling_control.ipynb`.

**Status.** Written. The srank column was read from the stored records on 17
November.

## 5.7 Comparison across task orderings, and a retraction

**Contents.** The 6 November single-run result on task ordering, stated
plainly. The same measurement with five seeds per ordering, and the spread
between seeds within one ordering. The gap between orderings expressed in
standard errors. The retraction.

The srank rows reach the usual bar. Two reasons they are not reported as a
result: the three image sets are not independent tests, and 63 comparisons at
that bar produce one or two by chance.

**Experiment.** Experiment 5 in the explainer.

**Notebooks.** `44_order_A_five_seeds.ipynb`, `45_orders_B_and_C.ipynb`.

**Tables to include.** The old single-run values against the five-seed means
with their spreads. The widest gap between orderings for every measure, in
standard errors.

**Figure.** The two measurements side by side, with the seed spread shown on
the second.

**Status.** To write.

## 5.8 Reproducibility of the probe

**Contents.** The unplanned control. Boundary 0 is rebuilt from the seed with
no saved file, and the probe's own random seed ignores the ordering, so each
seed's starting configuration was probed three times. The spread between those
three readings, and what it bounds.

The cause of the spread is not established. Floating-point arithmetic on the
GPU and an unintended path from the ordering to the starting generator are
both open. This should be stated as open, not resolved.

**Experiment.** Experiment 6 in the explainer.

**Notebook.** `47_probe_on_the_revised_streams.ipynb`.

**Table to include.** Five rows, one per seed, with the three readings and the
largest gap.

**Status.** To write. The cause should be checked before submission.

---

## 5.9 Learning speed on the held-out group

Section 5.3 reports one number per probe run: the accuracy reached after a fixed
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

Table 16. Test accuracy on the held-out group at each of the four readings,
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

Table 17 separates speed from the ceiling directly. For each run, its accuracy at
step 62 is divided by its own accuracy at step 250. The result says how much of
that run's own final result was already reached early. Dividing each run by its
own ceiling is what makes this a measure of speed: a model that ends higher while
moving at the same pace gives the same share.

Table 17. Share of a run's own step-250 accuracy already reached at step 62,
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

One comparison in this section disagrees with Section 5.4 and both numbers are
reported. Section 5.4 finds the accuracy endpoint complete at boundary 4, with no
detectable change from boundary 4 to boundary 9. The speed share does change over
that interval. On ordering A, where boundary 4 was measured, the share rises by
+0.076 from boundary 4 to boundary 9. Counting every repetition as an observation
gives a 95 percent interval of 0.044, and counting each seed once, which is the
conservative treatment because the eleven repetitions of one seed probe the same
two models, gives 0.063. The gain is larger than both intervals, but it is larger
than the conservative one by a small margin, and it rests on five seeds of a
single ordering. It is reported as evidence that the speed gain continues past the
point where the accuracy endpoint stops, and not as a settled result.

**Experiment.** No new run. Read from the probe records already in storage.

**Script.** `scripts/learning_speed.py`.

**Status.** Written. The boundary 4 rows are ordering A only, which is stated in
the captions.

---

# 6 Discussion

**Contents.** What the measures appear to respond to, given Sections 5.4 and
5.6. Why a scale-normalized measure can still carry a scale signal in this
architecture. What a reader should take from the result, and what they should
not: this is one architecture, one dataset, and one kind of learning.

**Source.** No new data.

**Status.** To write after Section 5.

---

# 7 Limitations

**Contents.** One model family, one hypernetwork, one dataset. The probe task
is drawn from the same dataset as the stream, and the probe trains the
generator, so part of the gain in Section 5.3 is ordinary transfer between
related tasks; a frozen-generator arm exists in the code and was not run. The
indicators move almost entirely at the first task (Section 5.1), so the
comparison between boundary 0 and boundary 9 is largely a comparison between
an untrained and a trained generator; the probe was not read at boundary 1, so
the middle of the stream is untested on either side. The fifteen runs share
five starting points, so they are not fifteen independent draws. The
unresolved cause of the Section 5.8 spread. The absence of a pre-stated test
for srank.

**Status.** To write.

---

# 8 Work not yet carried out on the present design

**Contents.** A table of measurements that exist on the retired eight-task
design and have not been repeated here. No numbers from those measurements
appear anywhere else in the paper.

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

**Status.** Table drafted. Each row leaves the table as its measurement is
repeated and moves into Section 5.

---

# References

**Status.** Carried over from version 1. Add nothing that is not cited.

---

# Appendix A Per-run tables

**Contents.** The full fifteen-row tables that Section 5 summarizes, so that
every number quoted in the body can be traced to a run.

**Status.** To assemble once Section 5 is written.
