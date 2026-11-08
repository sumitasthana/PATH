# Experiment trajectory: 8 November 2026

Back to the [index](Experiment-trajectory).

## Most of the unlearn rank rise is the weight scale, not the representation

The 7 November run reported that all three weight-reading indicators read
healthier right after task 3 was destroyed. The same step shrank the generated
weights by 8.4 percent, and that was recorded as an open alternative
explanation. It has now been tested, and it carries most of the effect.

### How it was tested

Four arms, nothing trained, every arm a forward pass. The script is
`scripts/norm_control.py` and it runs on the CPU in about twenty minutes, most
of that spent downloading two models from R2.

| Arm | What it is |
| --- | --- |
| `t8` | the finished eight-task model, as it is |
| `t9` | the post-unlearn model, as it is |
| `t9_at_t8_norm` | the post-unlearn weights multiplied by 1.091386, which puts their norm back where boundary 8 had it |
| `t8_at_t9_norm` | the eight-task weights multiplied by 0.916266, which shrinks their norm to where the unlearn left it |

The two models are the source job's `checkpoint.pt` and the `boundary_9.pt`
kept by the branch run. The scale is applied to every generated weight at once,
after the generator writes them and before the target network runs, so the
generator itself is untouched and only the size of what it writes changes.

The first two arms exist to check the script against the record. They
reproduce it: effective rank 2.9704 against the recorded 2.9708 at boundary 8
and 3.2661 against 3.2664 at boundary 9, with srank identical at 175 and 188.
So this measures what the run measured.

### What came back

| Arm | Rank, in stream | Rank, probe | srank, in stream | srank, probe |
| --- | ---: | ---: | ---: | ---: |
| `t8` | 2.9704 | 2.9679 | 175 | 178 |
| `t9` | 3.2661 | 3.2760 | 188 | 190 |
| `t9_at_t8_norm` | 3.0679 | 3.0714 | 179 | 182 |
| `t8_at_t9_norm` | 3.0858 | 3.0916 | 184 | 186 |

Read it twice, once in each direction.

**Taking the norm change away from the unlearned model removes about two
thirds of the rise.** Growing the post-unlearn weights back to the boundary 8
norm drops effective rank from 3.2661 to 3.0679, which is 67 percent of the
0.2957 gap, and srank from 188 to 179, which is 69 percent of the 13 point gap.
The probe set gives 66 and 67 percent.

**Imposing the same norm change on the undamaged model reproduces much of it.**
Shrinking the eight-task weights, and changing nothing else, moves effective
rank from 2.9704 to 3.0858 and srank from 175 to 184. That is 39 percent of the
rank gap and 69 percent of the srank gap, produced by the weight scale alone on
a model that was never unlearned.

The two directions agree closely on srank, at 69 percent both ways, and
disagree on effective rank, 67 percent one way and 39 percent the other. That
asymmetry is expected: the network is not linear, so removing a change at one
point and adding it at another need not cost the same.

### What is left over

After the norm is held fixed, boundary 9 still reads above boundary 8, by
0.0975 on effective rank and 4 on srank in stream, and by 0.1035 and 4 on the
probe set. That residual is the part the weight scale does not explain.

It is too small to call a signal from one seed. The late stream's own
boundary-to-boundary movement was -0.109, -0.183 and -0.004 on effective rank
and -3, 0 and 0 on srank, so a residual of +0.0975 and +4 sits inside the range
of ordinary steps rather than outside it.

### This does not touch the stream results

The obvious worry is that if effective rank partly reads the weight scale, then
every rank number in this project is in doubt, including the 6 November result
that the fall across a stream is set by which task is learned first. It is not.

| Run | Norm at t=0 | Norm at t=8 | Norm change | Rank at t=0 | Rank at t=8 | Rank fall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| order 01 seed 0 | 273.36 | 293.88 | +7.51% | 7.4249 | 4.0169 | 45.9% |
| order 01 seed 1 | 274.19 | 286.34 | +4.43% | 4.5626 | 2.9708 | 34.9% |
| order 01 seed 2 | 271.87 | 285.22 | +4.91% | 9.1852 | 3.5510 | 61.3% |
| order 02 seed 0 | 273.36 | 291.46 | +6.62% | 7.4135 | 2.0706 | 72.1% |
| order 03 seed 0 | 273.36 | 291.34 | +6.58% | 7.5014 | 5.6165 | 25.1% |

The norm change is nearly the same across the five runs, 4.4 to 7.5 percent,
while the rank fall ranges from 25 to 72 percent. Orders 02 and 03 start from
the identical generator and finish with norm changes of 6.62 and 6.58 percent,
which are the same to within a twentieth of a percent, yet their rank falls are
72.1 and 25.1 percent. The weight scale cannot be producing that.

The reason the norm matters at the unlearn and not in the stream is the size
and the direction of the move. Learning raises the norm by 4 to 7 percent
spread across eight tasks, roughly half a percent per task. The unlearn lowers
it by 8.4 percent in 100 steps, about fifteen times a task's move and the other
way, ending below the untrained generator's 274.19.

### What this changes, and what it does not

The 7 November reading called the unlearn result the strongest evidence yet
against the indicators and said the move was far larger than anything late
learning produced. **The size claim is withdrawn.** Two thirds of it is the
weight scale, and the remainder is the size of an ordinary step.

**The result against the indicators stands, for a different reason.** They were
asked whether they can detect damage when it is genuinely present. On a model
whose task 3 had just fallen to chance, all three moved the way they call
healthy. Learning why does not undo that. A measure that reports improvement
because the damaging operation happened to shrink the weights is still not
reporting damage. What has changed is that the failure now has a mechanism: on
this architecture, effective rank and srank are partly a readout of how large
the generated weights are.

That mechanism is worth stating plainly, because it is a property of the
measurement rather than of this one run. Both measures divide by the sum of the
singular values, so neither can be moved by rescaling a feature matrix. But the
features are not reached by a rescale. They pass through ReLU and through batch
normalisation held at fixed statistics, so shrinking the weights moves units
across their firing threshold and changes which directions carry the signal.
Any comparison of these indicators between two models whose generated weight
norms differ by several percent is partly a comparison of those norms.

### Limits

One seed, one unlearned task, one pair of models. The residual is reported as
a residual, not as a finding. Dormant units were not put through this control;
they moved by 3 to 5 percent of the stream's total on 7 November, which is
small enough that the same test would not separate anything. EuroSAT was left
out because the local machine has no copy, so the control covers the in-stream
and probe sets only, which is where the recorded effect was largest.

### What this makes worth doing next

The direct probe has still never been run on a damaged model, and
`boundary_9.pt` is sitting in R2 waiting for it. That measurement does not care
about weight scale, because it trains a new task code and counts accuracy, so
it answers the question this control cannot: whether the unlearned model is
actually worse at learning something new. It is also the natural first point of
the partial-return grid, since the unlearn turns out to move the generated
weights back toward their starting values.
