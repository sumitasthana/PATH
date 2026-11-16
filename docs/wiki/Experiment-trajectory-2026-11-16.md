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
