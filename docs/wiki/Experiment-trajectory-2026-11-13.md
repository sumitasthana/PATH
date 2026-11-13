# Experiment trajectory: 13 November 2026

Back to the [index](Experiment-trajectory).

## The learning-rate sweep: 5e-4 wins, but not by enough to call it a win

Phase 2a of the revised study is done. Runs 1 to 5 of the thirteen, the
learning-rate sweep, all finished on one Colab session with an L4.

### Settings

Five short runs, one per candidate rate, each learning the first three tasks of
`order_A` at seed 0 and then stopping. Everything else held fixed: five epochs
per task, batch size 64, ResNet50 written by the generator, beta 0.01,
hyperfan-in initialization, twenty classes per task. 785 optimizer steps per
task, which is 10,000 training images in batches of 64, five times over.

The generator holds 134,076,432 parameters and writes 23,541,332 into the
target network. Peak GPU memory was 5.08 GiB of the L4's 22, so the batch size
was never near the limit.

### Result

| Learning rate | Mean over 3 tasks | Task 0 | Task 1 | Task 2 | Minutes |
| --- | --- | --- | --- | --- | --- |
| 3e-4 | 48.80% | 43.5 | 51.5 | 51.4 | 14.6 |
| **5e-4** | **49.33%** | 39.8 | 52.3 | 55.9 | 14.0 |
| 1e-3 | 34.60% | 11.1 | 43.2 | 49.5 | 14.1 |
| 2e-3 | 41.23% | 23.6 | 48.9 | 51.2 | 14.0 |
| 4e-3 | 39.07% | 31.1 | 40.8 | 45.3 | 14.0 |

Chance accuracy is 5 percent, because a task now holds twenty classes. Every
value is far above it, so the model learns at all five rates. The rejected
values are kept here because the methodology reports the tuning runs in full.

5e-4 is chosen, by the rule set before the sweep ran: highest mean accuracy
over the three tasks.

### Why the win does not mean much on its own

5e-4 beats 3e-4 by 0.53 percentage points. The methodology's own noise floor,
the figure the five-seed confirmation has to come in under, is 3 points. The
gap between first and second place is about a sixth of that, on one seed each.
These two values are tied as far as this sweep can tell.

The shape of the results says the same thing a second way. The response is not
monotonic: 1e-3 scores 34.60, below both its neighbours, 5e-4 at 49.33 and
2e-3 at 41.23. A smooth response to the learning rate does not dip in the
middle and come back up. With one seed per value, the most likely reading is
that seed-to-seed variation here is worth several points, which is large enough
to have produced the ordering of the top two by itself.

So the sweep supports a weaker claim than a single chosen number suggests: the
two low rates, 3e-4 and 5e-4, are clearly better than the three higher ones,
and the sweep cannot separate the two low rates from each other.

### What actually separates the candidates

Almost all of the spread is in task 0, the first task learned:

| | Range across the five rates |
| --- | --- |
| Task 0 | 32.4 points, from 11.1 to 43.5 |
| Task 1 | 11.5 points, from 40.8 to 52.3 |
| Task 2 | 10.6 points, from 45.3 to 55.9 |

Task 2 is measured right after it is learned and barely moves. Task 0 has to
survive two later tasks, and that is where the learning rate decides the
outcome. The learning rate's main effect in this setting is on how much of the
first task is still there at the end, not on how well a task is learned when it
arrives.

At 1e-3, task 0 falls to 11.1 percent, which is close to the 5 percent floor.
That value is the default the earlier version of this work used and never
tuned. It is the worst of the five tested here.

### The nine-task runs will cost about twice what was projected

The sweep's first run printed a projection of 5.1 hours for the seven nine-task
runs. That was wrong, and the notebook has been corrected.

A task costs more the later it arrives, because the term that holds earlier
tasks still has one more task to hold at every optimizer step. The first run's
three tasks took 0.244, 0.361 and 0.470 seconds per step, so the cost per task
rises by roughly 89 seconds for each extra task being held. The old projection
multiplied a flat per-task mean by 63 and so assumed the ninth task costs what
the first one did.

Fitting a line to the rise and extending it to nine positions gives about 1.4
hours for one nine-task run and **about 9.6 hours for the seven of them**. The
fit rests on three points, so treat that as a floor rather than an estimate.

### Decision

Go to runs 6 to 10 at 5e-4: five seeds of the full nine-task `order_A` stream.
The gate is unchanged. The five seeds must finish within 3 percentage points of
each other on mean accuracy over the nine tasks, or the value is not good
enough to measure against and tuning continues instead.

If 5e-4 fails that gate, 3e-4 is the fallback and needs no new sweep. It is the
value this sweep could not separate from the winner, and it holds task 0 best
of the five at 43.5 percent.

### Decision: the unlearn step runs on all seven stream runs, not one

The methodology first called for one unlearn, a tenth step on one finished run.
The run notebooks were written to unlearn on every run instead, each on the
task its ordering learned first: task 0 in order A, task 4 in order B, task 8
in order C. That is seven damaged models rather than one.

The user has chosen the seven, and `docs/plasticity/methodology.md` has been
updated to match rather than left to disagree with the code.

It costs almost nothing. An unlearn is 100 update steps against the 7,065 a
nine-task run spends learning, so the whole arm adds under 2 percent. What it
buys is a floor. The five order A runs differ only in their seed, so the damage
test now has a spread of its own, and the rescaling control in section 3.8 has
seven before-and-after pairs instead of the single pair it used to rest on. A
difference smaller than the spread across those five is not an effect, which is
the same rule the rest of the study already applies and which the one-run
version of this arm could not state at all.

The reading after the unlearn was also renumbered in writing. Section 3.5 said
"the tenth boundary", which dates from when the stream was eight tasks long.
With nine tasks the stream ends at boundary 9 and the unlearn is boundary 10,
the eleventh reading counting the fresh generator as boundary 0. The notebooks
already number it that way. This is the specification catching up with the
code, and it is worth having unambiguous on paper: the 7 November record shows
a boundary off by one already cost this study a run once.

`docs/plasticity/run-plan.md` still carried the one-run wording in two places:
the summary of the arms, and a paragraph headed "unlearn settings for run 13".
There is no run 13 any more, because the unlearn is a step inside each of the
seven runs rather than a run of its own. Both are corrected. The cost section
of that file already said seven.

### The eleventh boundary breaks the report's idea of the end

`indicator_report.endpoints` takes the smallest and the largest boundary number
in a record and reports the change between them. Every earlier record stopped
at the last task learned, so the largest number was the end of learning.

These records do not stop there. Boundary 10 is the model after the unlearn, so
the largest number is now a model damaged on purpose, and the headline change
for every measure would read from the fresh generator to the damaged model.
Section 4.1 asks what nine tasks of learning do to effective rank. That number
would quietly mix the damage into the answer.

Nothing is wrong in either run notebook, and nothing needs changing in the
library. The report has to be run twice: over boundaries 0 to 9 for the
learning claim, and over all eleven for the damage claim. Recorded here because
the wrong version would look entirely reasonable in the output.

### Gaps in this record

The per-run `environment` files record the code revision as `None`, because the
run was started from a Colab clone where the notebook's own revision lookup did
not reach the saved environment record. The revision is in
`sweep_summary.json`, which the notebook writes itself, so the runs are not
unidentified, but the two records disagree and the environment file is the one
that is wrong.
