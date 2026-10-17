# Experiment trajectory

Updated 17 October 2026. This is the index for the dated experiment record. Each page keeps the questions, settings, results, evidence links, and decisions together.

## Latest findings

The [17 October record](Experiment-trajectory-2026-10-17) adds the first plasticity measurement. Plasticity means the ability to learn a new task, which is a separate question from forgetting. Notebook 26 supplies the baseline the study never had: a generator that has learned nothing, taught task 15 alone. On seed 0 that baseline reaches 37.6% with the chunk embeddings frozen, against 50.8% for the same task learned after the eight tasks of order 01. The difference of +13.2 points exceeds the 3-point noise floor and points away from plasticity loss: the generator learns a new task better after eight tasks than it did fresh. The protection term handicaps the eight-task arm rather than helping it, so the planned beta-zero control is not needed while the sign stays positive. This rests on one seed, one probe task and one order. The recorded seed spread for the comparison arm is 16.4 points, larger than the effect, so seeds 1 and 2 are required before this is a finding.

The [10 October record](Experiment-trajectory-2026-10-10) and [completed selected-study report](../reports/Paired-study-2026-10-10-draft.md) now include all six verified comparisons across seeds 0, 1, and 2. Both forgotten tasks finish at 10% accuracy in every seed, with no recorded relapse above the 12% threshold during task-15 learning. The three-seed mean task-15 change is +5.40 points for U3/L15 and -0.20 points for U14/L15. The task-3 mean includes a large +13.0-point seed-2 effect; results vary substantially by seed. Mean retained-task changes are -1.29 and -1.02 points, and the largest individual losses range from 2.6 to 14.2 points. All 12 selected jobs and all 39 requests are complete. All six saved CPU-summary exports and eleven session logs are verified in R2. Recorded runner time totals 15 h 10 m 20 s. These results do not establish information erasure or recovery resistance. A [four-slide HTML presentation with an interactive experiment tree](../reports/Paired-study-2026-10-10-presentation.html) is available for presenting and inspecting the nine continuations. A concise [mentor-facing public summary](https://sumitasthana.github.io/CARK/paired-study.html) presents the scores, averages, and limits in tables and a labeled chart.

The [9 October pilot record](Experiment-trajectory-2026-10-09) contains user-reported seed-0 results for both selected comparisons. U3/L15 matched the learning-only control on task 15; U14/L15 was 2.6 percentage points lower. Mean retained-task differences were -1.086 and -1.6 points, but individual tasks fell by up to 11.0 and 14.2 points. The original multi-seed matrix remains incomplete. Forgotten-task accuracies were not included in the supplied excerpt, so successful forgetting and its persistence are not established by these figures.

Study-model preparation has started: the user reported seed 0 completing shared L0 at 51.4% validation accuracy, with its checkpoint and report verified in R2. The next stage prepares the never-learned-task-3 sequence from that checkpoint. Recovery remains untested.

In the earlier controlled comparison, task-3 accuracy fell from 27.4% to 10.0%, the chance score for ten classes, and stayed there during one later lesson. Other tasks showed mixed changes. New task 15 scored 49.0% after unlearning versus 54.4% with learning alone.

Old-task protection helped in the beta trials, but repeated-session variation remains unexplained. Chance accuracy does not establish information erasure. Recovery, privacy, and results across multiple training seeds remain untested.

Read the [7 October research summary](Experiment-trajectory-2026-10-07#research-summary-as-of-7-october-2026) for the current assessment and comparison with the paper. The [illustrated HTML report](https://github.com/sumitasthana/CARK/blob/main/site/index.html) presents the findings with six charts.

## Record by date

The [8 October setup record](Experiment-trajectory-2026-10-08) describes reserved recovery images, matched-model preparation, completed R2 cleanup, and the resumable paired-study notebooks. R2 now holds 0.56 GB after historical-model cleanup; the paired study keeps approximately 4.84 GB of source models by default. The user has reported shared L0 completion for seed 0; no recovery or paired-generalization result has been reported.

| Reporting date or period | What is covered |
| --- | --- |
| [17 October 2026](Experiment-trajectory-2026-10-17) | First plasticity probe: the missing fresh-generator baseline for task 15, measured on seed 0. |
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
