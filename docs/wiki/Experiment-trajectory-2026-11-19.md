# 19 November 2026

Version 3 of the plasticity-indicators paper was audited against the code and
the saved logs. No GPU ran on this day. Every number below comes from records
that already existed, and two of them change what the paper claims.

## Why the audit happened

The draft argued that the three rank-based indicators fail even once the
reading protocol is set aside. That argument was circular: Sections 3.3 to 3.5
are all read through the fixed BatchNorm protocol that Section 3.1 shows to be
broken, so they cannot be independent evidence against it. The abstract now
says the opposite plainly, that the fixed protocol is the reading a
hypernetwork gets by default and the rest of the paper stays with it, and the
claim is narrowed to what that supports. Whether the three results survive
matched statistics is untested on these fifteen streams and is listed as such.

## Result: the correlation test depends on how the runs are counted

Table 4 and Table D2 correlate each indicator's change with the probe's change
over fifteen runs. The fifteen runs are five starting points. Orderings A, B
and C at one seed are three streams written by the same starting network, so a
seed gives three correlated points and not three separate ones.

Averaging the three orderings of each seed first and correlating the five
averages gives a different answer for the two rank measures.

| Measure | Predicted | r, 15 runs | r, 5 seeds | r end, 15 runs | r end, 5 seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| Effective rank | positive | -0.054 | +0.327 | -0.165 | -0.350 |
| srank | positive | -0.178 | +0.209 | -0.212 | -0.049 |
| Dormant units, 0.01 | negative | +0.129 | +0.077 | +0.164 | +0.187 |
| Dormant units, 0.025 | negative | +0.125 | +0.076 | +0.162 | +0.175 |
| Dormant units, 0.1 | negative | +0.112 | +0.095 | +0.139 | +0.178 |

At five seeds both rank measures turn over to the sign they predict. The
draft's claim that all five come out with the wrong sign holds at fifteen runs
and does not hold at five seeds, so the paper no longer rests on it.

What does hold at either count is size. Nothing in the table is separable from
zero: fifteen runs need 0.514 and five seeds need 0.878, and the largest value
anywhere is 0.350. The five numbers are also closer to two. Across the fifteen
runs the three dormant cut-offs correlate 0.993 to 0.999 with one another and
effective rank correlates 0.690 with srank.

`scripts/upgrade_suite/seed_clustered_correlations.py` computes every value
from the published logs. Its check reproduces the two rank rows of the old
Table D2 within 0.0002 and the three dormant rows within 0.0011, so cut-off
0.025 rounds to 0.124 here against the published 0.125. The script that first
produced Table D2 is not in the repository, so that last digit is unexplained
and the published values were left alone.

## Result: the curvature split is not the time window

Table 2 compared a fixed-code arm read at boundaries 0 and 9 against a
trained-code arm read at boundaries 4 and 9. Four things differed between the
two columns at once: the task code, the BatchNorm statistics, the group of
pictures, and the span of stream covered.

The last of those was free to remove. The curvature run measured the fixed arm
at boundary 4 as well, 600 readings already in the published log and never
reported. Over boundary 4 to boundary 9, the trained arm's own span:

| Seed | Fixed arm, b4 | Fixed arm, b9 | Change | Floor | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 2.077e8 | 2.784e10 | 2.763e10 | 2.502e9 | rose |
| 1 | 1.192e10 | 4.020e11 | 3.900e11 | 3.756e10 | rose |
| 2 | 2.044e8 | 3.628e10 | 3.607e10 | 3.879e9 | rose |
| 3 | 1.896e9 | 1.126e12 | 1.124e12 | 1.412e11 | rose |
| 4 | 543.2 | 5.730e5 | 5.724e5 | 7.361e4 | rose |

All five seeds rise past their own floor and all three step sizes agree, which
is what the longer span reports. The trained arm over that same span rises on
2 of 5 and falls on 2. So the two arms still disagree once the span is held
fixed, and the span is no longer one of the things that differ.

The size of the rise does depend on the span. The ratio falls from 6.53e7 to
5.95e12 over boundary 0 to 9, down to 33.7 to 1.06e3 over boundary 4 to 9, so
most of the published rise belongs to the first four tasks. Three variables
still differ between the arms: the task code, the statistics and the picture
group. No arm separates them.

`scripts/upgrade_suite/curvature_same_window.py` reproduces Tables I1 and I2
from the log before printing the new span.

## Correction: the collapse was not shown to be linear

The draft said the rank collapse occurs in the linear part of the network. The
measurement behind that is a reading at the input of the last ReLU, which shows
the collapse is already complete when the signal arrives there.

That clears the last ReLU and nothing else. The signal reaching it has already
passed the forty-nine ReLUs before it, so the reading cannot locate the
collapse in the linear part. The claim is replaced in the paper and in the
16 November record with what the measurement supports: the last ReLU is not
what collapses the rank.

The code is correct and does what its own documentation says. Only the
conclusion drawn from it was too broad.

## Correction: the dormant count is not read at the classifier

The Method said all three measures are read at the 2048 features entering the
final classifier. Effective rank and srank are. The dormant-unit fraction is
not: `activation_sites` in `uncle/plasticity_indicators.py` hooks every ReLU
call in the network and scores each unit against the mean of its own call site.
The Method now says so.

## Finding: Table B6 cannot be filled from storage

The control network's per-group accuracy was to be read from notebook 51's run
log. It is not there. The published file holds 180 rows, which is three seeds
by ten boundaries by two buffer modes by three picture sets, and every field is
an indicator field.

Notebook 51 builds the accuracies in a local list and prints them in the next
cell. Nothing uploads them, and the committed notebook has no stored output, so
the numbers existed only on screen in a session that is closed. The placeholder
in the paper now says this instead of naming a source that does not hold them.

The repair is a repeat of the run, about five minutes a seed for three seeds.
The upload line should be added at the same time.

## Correction to an earlier reading of the records

An audit note written earlier on 19 November reported that the boundary 0
rebuild is an open problem, citing notebook 52's 11.790 against the logs'
5.563. That was read from the "Open" section of the 16 November record without
reading the resolution later on the same page. The cause was found and fixed on
16 November: the notebook did not seed the global random number generator
before building, so it was reading models that were never part of the study.
Both notebooks now seed inside `build`, and the rebuild matches the logs.

This matters for what can be run next. Boundary 0 has no checkpoint, and it
does not need one. A matched-statistics pass over the fifteen streams can read
boundaries 0, 9 and 10, so the change columns of Table 4 are reachable and
nothing is blocked on an unexplained number.

## What was published into version 3

Item 1, the circularity, was reframed in text across the abstract, the
introduction, Sections 3.3 to 3.5, the discussion and the limitations.

Then nine smaller repairs, all from existing data and none needing a GPU: the
two corrections above; the seed-clustered correlation columns in Tables 4 and
D2; the fixed arm's shared-window column in Table 2 and a new Table I4; the
990-run accounting written into the Method; a reconciliation of Table C1
against Table G1, which differ because C1 keeps the best learning rate in each
repetition while G1 averages all three; the CIFAR-100 probe as Table C3 and the
frozen-generator control as Table C4, both ordering A and five of the fifteen
runs; an honest Table B6 placeholder; and a forward reference inside Appendix C
that pointed at Appendix C.

## Next decision

Three things remain, in order of what they cost.

Repeat notebook 51 for Table B6, fifteen minutes.

Run the matched-statistics pass on all fifteen streams at boundaries 0, 9 and
10, under an hour and mostly transfer time. This is the one that decides
whether the corrected measures track the probe or still fail, and Sections 3.3
to 3.5 cannot be closed without it.

A random-label probe and an induced plasticity-loss stress test are both
proposed and neither has started. The stress test alone costs more than
everything else in this plan together, and it only makes sense after the
matched pass reports.
