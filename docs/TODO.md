# TODO

Working backlog, sequenced. Last updated 23 October 2026, after batch 1 ran.

Experiment results live in [the experiment trajectory](wiki/Experiment-trajectory.md).
The implementation plan lives in [PLAN.md](PLAN.md). This file is only what is next.

## Status at a glance

| Item | Status |
| --- | --- |
| Batch 1, recovery arms, notebook 32 | Done. Nothing recovered |
| Batch 2, harder cases, notebook 32 rerun | Partly superseded, see below |
| Batch 3, three seeds, notebook 30 | Done |
| Batch 4, tell the group | Note written, not sent. Notebook 33 not run |
| Batch 5, plasticity infrastructure | Not started, still blocked |
| Write batch 1 into the experiment record | **Pending, do this first** |

## The goal has changed

The old goal was to show that the method's own stopping point leaves the task recoverable.
Batch 1 falsified the recoverable half. The weights had barely moved at step 20, and the task
was gone anyway, and no multiplication brought any of it back.

The new goal, which is what the measurements actually support:

**Show that the usual evidence for unlearning does not establish what it claims, and that
this method does all its removing in the first 6 steps of 100.**

Four results stand behind it, all from seed 0 unless noted:

- The task reaches chance at step 6. The method runs 100 steps. So 94 steps remove nothing.
- Chance accuracy is not evidence of removal. From step 12 the model answers two classes, and later one, which scores chance on a balanced set whatever the model knows.
- Effective rank, the measure the adapter study uses, read 214.7 before and 225.2 after. It did not notice the task being destroyed.
- The stored per-task running averages are load bearing and unlearning never touches them. Replacing them kills a healthy task outright.

## Already done

- Direction measurement during unlearning, three seeds. Notebook 30. Weights turn 49 to 54 degrees and lose 43 to 54 percent of their length over 100 steps. Seed 0 is the mildest.
- Accuracy joined to direction. Notebook 31. Join validated to 0.13 percent.
- Recovery arms at step 20, seed 0. Notebook 32. Every arm sits at chance.
- The accuracy trace during unlearning, every two steps. Crossing located at step 6.
- `measure.direction_change` in the library, with tests.
- Tiny ImageNet download falls back to a second host, after the Stanford one went down.

## Do first: write batch 1 into the record

The batch 1 result is not in `docs/wiki/` yet. It changes the headline, so it should go in
before anything else runs. It needs to carry the failure of my own hypothesis, not only the
numbers.

## Batch 1 follow-ups, in order

- **Recover at step 6, not step 20.** Notebook 32 with `STOP_AT = 6`. The task has just reached chance there and the weights have moved even less. About 7 minutes. If rescaling fails here too, the result is as strong as it can be made.
- **Relearn instead of rescale.** The real recovery test. Give the unlearned model a few examples of task 3 and a fixed small budget, and compare against a model that never learned task 3. Faster relearning means the information is still present. Rescaling was the cheapest possible attempt and it failed; this is the one that decides the privacy question. **Notebook 32 is not this test and notebook 26 is not either.** The notebook already exists: `24_recovery_probe.ipynb`, built 13 October, with four arms that separate task 3's code from its BatchNorm statistics. It now also runs both stopping points, 6 steps and 100, in one session. It has never been run. It needs an A100 for about thirty minutes and runs on Colab, so it is not blocked on RunPod. Do not build anything new first.
- **Fix or drop the neutral statistics arm.** As built it breaks healthy tasks, so it cannot isolate the forgotten one. The control arm caught this. Either compare against statistics from another task, or drop the arm and test the statistics separately.

## Batch 2: harder cases

- **Task 14 instead of task 3.** Still worth running. Task 3 started at 35.60 percent because it is first in the order. A task at high accuracy may not die in 6 steps. Notebook 32 with `FORGET_TASK = "14"`.
- **Step 100 instead of step 20.** Downgraded. Rescaling already failed at a 7 degree turn, so it will certainly fail at 49. Run only if a reviewer asks for the contrast.

## Batch 3: more than one seed

Done, 23 October. The direction trajectory holds on all three seeds. At step 20 the direction
is 0.9833 to 0.9919. By step 100 it is 0.5864 to 0.6542. No channel lost its weights on any
seed, and the control held on all three.

Still one seed on the accuracy side. Notebook 27 and notebook 32 both ran seed 0 only, so
every accuracy statement rests on seed 0 while the weight statements rest on three.

The rescaling factor differs by seed: 1.174, 1.255 and 1.290 at step 20. Notebook 32 computes
it per seed, and measured 1.190 for seed 0 at step 20.

## Batch 4: tell the group

- **Run notebook 33.** Not yet run. Seconds, reads only.
- **Send the note.** Written at `ops-docs/andrii-hypernetwork-plasticity-note.md`, not sent. It needs one addition now: effective rank read 214.7 before unlearning and 225.2 after, while the task was destroyed. That is a second, independent case of the measure failing to report what it claims, from our architecture rather than theirs.

## Batch 5: plasticity infrastructure, blocked

Every item is blocked behind the first, and every item is unblocked by it.

- Retrain the stream saving a checkpoint after every task. `checkpoint.save` already takes a path, so this needs a per-task path and nothing else.
- Then: mid-stream probe points, orders 02 and 03, trained reference points for the probes, per-task weight geometry across the stream, and the EuroSAT external probe.

Do not start until the unlearning line is written up.

## Parked

- Experiment B baseline and Experiment C code-only arm, from the 17 October record.
- Put the Tiny ImageNet archive in R2, so the notebooks stop depending on outside hosts. About 240 MB against the 9 GB already there.
- Anything needing a GPU, until RunPod is ready. Its token was rejected on 23 October and needs re-authenticating. Nothing in batches 1 to 4 needs it.
