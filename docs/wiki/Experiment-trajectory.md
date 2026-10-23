# Experiment trajectory

Updated 23 October 2026. This is the index for the dated experiment record. Each page keeps the questions, settings, results, evidence links, and decisions together.

## Latest findings

**The task is gone by step 20, and the other 80 unlearning steps only make it unrecoverable.** Accuracy on the removed task reaches chance at step 20 and never moves again. At that point the weights are 99.19 percent aligned with where they started, a turn of 7.3 degrees, having kept 85 percent of their length. By step 100 they have turned 49 degrees, which no rescaling can undo. So the unlearning length past step 20 buys irreversibility, not forgetting.

**A model at chance accuracy is not evidence that a task was removed.** From step 40 the model answers the same single class for every test image, which scores exactly 10.00 percent on a balanced ten-class test set, the same as chance. Input sensitivity is the measurement that still works: it falls by a factor of 5.8 million across the run, including through the windows where accuracy is flat.

No channel lost its weights, so the 58.9 percent of channels that went silent on 21 October went silent in activations while keeping their weights. The removed task started at 35.60 percent rather than full accuracy, because it is first in the order and had already lost most of its accuracy to ordinary forgetting. One seed, CPU, notebooks 30 and 31. Details in the [23 October record](Experiment-trajectory-2026-10-23).

That same record reviewed our plasticity work against the adapter study in `reference/adapter-plasticity-study`, and then corrected itself. The review's main claim, that comparing against an untrained generator cannot answer a plasticity question, was wrong: at t = 0 the full probe already reaches about 52 percent on ten-class tasks against 10 percent chance. The [22 October EuroSAT plan](Experiment-trajectory-2026-10-22) is planned again, not deferred.

**What the t = 0 comparison measures is learnability relative to initialization.** It does not separate transfer from plasticity changes, and it does not rule out deterioration relative to an earlier trained checkpoint. The fix is to add trained reference points at t = 1, 2, 4 and 6 when the sequence is next trained, saving the generator after each task, rather than to replace t = 0. A `relative_contribution` docstring that misdescribed its own denominator has been corrected.

The [20 and 21 October record](Experiment-trajectory-2026-10-21) finishes the plasticity probe on three seeds.

**The generator learns an unseen task better after eight tasks than before any task,** by 4.74 points on average. Read this as: after eight tasks the generator performed better on the tested probes than a freshly initialized generator, under the chosen training budgets. It does not rule out deterioration relative to an earlier trained checkpoint, and it does not separate transfer from changes in plasticity.

Two measurements agree. Test accuracy on three unseen tasks rises by 6.37, 3.01 and 4.86 points, a mean of +4.74. Two of the three are bigger than their own noise floor; the third is just inside it. Training loss on randomly labelled images falls by 0.70, 0.76 and 0.64, against floors of 0.29, 0.42 and 0.18, so all three pass. Those three values differ from each other by only 0.06, which is very consistent.

That second measurement worked for the first time. On 17 October it was void, because the model never memorised anything. These numbers replace the 17 October reading of +0.8 points, which used one new task, one learning rate and a borrowed noise floor.

Two further results. Training a task's 32 numbers buys nothing. Leave them random and train only the generator. The differences are then -1.48, +1.01 and +1.27 points, against floors of 5.53, 3.23 and 2.28. Separately, a frozen generator with a fresh task code sits at chance on all three seeds. That confirms on three seeds what one seed had shown.

Comparing the three seeds gave two observations, each from three data points. First, one task is far noisier than three tasks averaged. Task 15 spans 16.4 points across the seeds, while the three averaged probe tasks span 2.13. That is why the 17 October spread was 5.5 and this one is 1.68. Second, the seed with the weakest control gained the most from unlearning: controls of 50.8, 44.0 and 34.4 pair with gains of 0.0, +3.2 and +13.0, in that order. Seed 2 supplies most of the +5.40 mean, and its control sits below its own fresh-model baseline. Rerunning that control would settle it.

One defect is recorded. The resume check compared two spellings of the same arm name, so two of three arms were never skipped. They ran again every session. That cost about 3.6 hours and left 276 repeated runs, which are now dropped when the file is read. The main result uses the unaffected arm.

The paper says unlearning restores plasticity. That claim is not supported here, because there is no plasticity loss to restore. The planned beta-zero control arm is no longer needed.

The [17 October record](Experiment-trajectory-2026-10-17) adds the first plasticity measurement and closes it. Plasticity means the ability to learn a new task, which is a separate question from forgetting. Notebook 26 supplies the baseline the study never had: a generator that has learned nothing, taught task 15 alone. Across three seeds with two runs each, the baseline reaches 44.5%, 43.2% and 39.0%, against 50.8%, 44.0% and 34.4% for the same task learned after the eight tasks of order 01. The paired differences are +6.3, +0.8 and -4.6 points, a mean of +0.8 with a sample SD of 5.5. **There is no plasticity change at this probe**, in either direction. An earlier seed-0-only reading of +13.2 points did not survive the other two seeds and is superseded. The planned beta-zero control arm is therefore not needed, because there is no plasticity loss for the protection term to explain. Two further observations: repeated runs of one identical configuration span 8.4 points across sessions, so the 3-point noise floor carried over from the replay work is too optimistic for this measurement; and the eight-task model's new-task accuracy varies three times more across seeds than the fresh baseline's does, which is recorded as an observation and not a claim. This result removes a candidate explanation rather than producing one, and the study's main finding is unchanged.

A teammate's adapter plasticity study asks the same question on different models, and a [read-only snapshot of it](https://github.com/sumitasthana/PATH/tree/main/reference/adapter-plasticity-study) is now in this repository for context. That study trains small adapters, called LoRA and OFT, on a frozen vision transformer and probes whether they keep learning new tasks over 20 tasks. Its provisional answer is that they do, at the learning rates that give the best accuracy. The two projects meet at one claim: the UnCLe paper states that unlearning restores plasticity, no plasticity probe has tested that claim, and the mechanism the paper gives for it, returning a task to a random initialisation, is contradicted by the measurements recorded here. Section 9 of the snapshot's handoff document proposes the shared experiment and records an open decision, namely whether hypernetworks are in scope for the adapter study at all. That decision has not been made, and no shared experiment has been run. The snapshot is a copy of a folder that is not under version control, so it will go stale.

The [10 October record](Experiment-trajectory-2026-10-10) and [completed selected-study report](../reports/Paired-study-2026-10-10-draft.md) now include all six verified comparisons across seeds 0, 1, and 2. Both forgotten tasks finish at 10% accuracy in every seed, with no recorded relapse above the 12% threshold during task-15 learning. The three-seed mean task-15 change is +5.40 points for U3/L15 and -0.20 points for U14/L15. The task-3 mean includes a large +13.0-point seed-2 effect; results vary substantially by seed. Mean retained-task changes are -1.29 and -1.02 points, and the largest individual losses range from 2.6 to 14.2 points. All 12 selected jobs and all 39 requests are complete. All six saved CPU-summary exports and eleven session logs are verified in R2. Recorded runner time totals 15 h 10 m 20 s. These results do not establish information erasure or recovery resistance. A [four-slide HTML presentation with an interactive experiment tree](../reports/Paired-study-2026-10-10-presentation.html) is available for presenting and inspecting the nine continuations. A concise [mentor-facing public summary](https://sumitasthana.github.io/PATH/paired-study.html) presents the scores, averages, and limits in tables and a labeled chart.

The [9 October pilot record](Experiment-trajectory-2026-10-09) contains user-reported seed-0 results for both selected comparisons. U3/L15 matched the learning-only control on task 15; U14/L15 was 2.6 percentage points lower. Mean retained-task differences were -1.086 and -1.6 points, but individual tasks fell by up to 11.0 and 14.2 points. The original multi-seed matrix remains incomplete. Forgotten-task accuracies were not included in the supplied excerpt, so successful forgetting and its persistence are not established by these figures.

Study-model preparation has started: the user reported seed 0 completing shared L0 at 51.4% validation accuracy, with its checkpoint and report verified in R2. The next stage prepares the never-learned-task-3 sequence from that checkpoint. Recovery remains untested.

In the earlier controlled comparison, task-3 accuracy fell from 27.4% to 10.0%, the chance score for ten classes, and stayed there during one later lesson. Other tasks showed mixed changes. New task 15 scored 49.0% after unlearning versus 54.4% with learning alone.

Old-task protection helped in the beta trials, but repeated-session variation remains unexplained. Chance accuracy does not establish information erasure. Recovery, privacy, and results across multiple training seeds remain untested.

Read the [7 October research summary](Experiment-trajectory-2026-10-07#research-summary-as-of-7-october-2026) for the current assessment and comparison with the paper. The [illustrated HTML report](https://github.com/sumitasthana/PATH/blob/main/site/index.html) presents the findings with six charts.

## Record by date

The [8 October setup record](Experiment-trajectory-2026-10-08) describes reserved recovery images, matched-model preparation, completed R2 cleanup, and the resumable paired-study notebooks. R2 now holds 0.56 GB after historical-model cleanup; the paired study keeps approximately 4.84 GB of source models by default. The user has reported shared L0 completion for seed 0; no recovery or paired-generalization result has been reported.

| Reporting date or period | What is covered |
| --- | --- |
| [23 October 2026](Experiment-trajectory-2026-10-23) | First direction measurement: unlearning turns the weights 49 degrees as well as shrinking them, seed 0 on CPU. Also the review against the adapter study and the correction withdrawing its main claim. |
| [22 October 2026](Experiment-trajectory-2026-10-22) | EuroSAT external-probe plan, rationale, controls, measurements, and compute budget. Not yet run. |
| [20 and 21 October 2026](Experiment-trajectory-2026-10-21) | Plasticity probe completed on three seeds: no loss, a small gain, and the task code shown to add nothing. |
| [17 October 2026](Experiment-trajectory-2026-10-17) | Plasticity probe: the missing fresh-generator baseline for task 15, three seeds, no plasticity change found. |
| [10 October 2026](Experiment-trajectory-2026-10-10) | All selected training and CPU review complete; six verified comparisons, three-seed averages, and readable report. |
| [9 October 2026](Experiment-trajectory-2026-10-09) | Live R2 progress check for paired generalization: two completed jobs, one resumable job, and 42 pending jobs. |
| [8 October 2026](Experiment-trajectory-2026-10-08) | Recovery setup and seed-0 shared L0 completion; verified R2 cleanup; paired-study preparation, resume, and CPU review notebooks. |
| [7 October 2026](Experiment-trajectory-2026-10-07) | R2 continuation, beta batch and repeat, learning sequence, paired unlearning comparison, and reflection against the paper. |
| [5 and 6 October 2026](Experiment-trajectory-2026-10-05-to-06) | Session replay, learning-only runs, Hyperfan pilot, task-17 continuation, and implementation reviews. |
| [Through 4 October 2026](Experiment-trajectory-2026-10-04) | Earlier reproduction stages, parameter studies, full sequence, replay variation, and weight audit. |
| [Through 29 September 2026](Experiment-trajectory-2026-09-29) | Detailed historical forgetting diagnostics, loss and gradient measurements, and decisions at the time. |

Older pages preserve historical interpretations and plans. They are not instructions to resume those experiments. Date ranges are used where the original record spans several days or does not give an exact run date.

## Keeping the record current

Add new results to the page for their reporting date, then update this index's latest findings. Preserve failed and interrupted attempts, actual settings, evidence, and limitations. Publish the index and dated pages together on Wednesdays and Saturdays at 6 p.m. America/New_York, or when explicitly requested.
