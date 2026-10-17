# Experiment trajectory: 7 October 2026

[Back to the experiment index](Experiment-trajectory)

Results reported or reviewed on 7 October. Experiments are paused. Earlier plans in the detailed record are superseded by the research summary below.

## On this page

- [Research summary as of 7 October 2026](#research-summary-as-of-7-october-2026)
- [Current findings and paused experiments](#current-findings-and-paused-experiments)
- [Reflection against the paper's claims](#reflection-against-the-papers-claims)
- [R2 continuation reported on 7 October](#r2-continuation-reported-on-7-october)
- [Beta batch completed on 7 October](#beta-batch-completed-on-7-october)
- [Beta 0.1 repeat read from R2](#beta-01-repeat-read-from-r2)
- [Learning sequence 1, 7, 14 completed](#learning-sequence-1-7-14-completed)
- [Paired U3 and L15 completed](#paired-u3-and-l15-completed)
- [Evidence and maintenance](#evidence-and-maintenance)

## Research summary as of 7 October 2026

We began by asking why older tasks lost accuracy. We checked repeated sessions, the learning implementation, and the strength of old-task protection. We then measured ordinary forgetting during successive lessons. Finally, we compared learning alone with unlearning followed by the same learning. This order matters: a loss caused by ordinary learning should not automatically be blamed on unlearning.

| Question | What we observed | What we learned |
| --- | --- | --- |
| Does the same saved model always give the same result? | Two beta-0.01 continuations had matching recorded starting models and settings, but task 9 finished at 32.6% and 50.0%. A beta-0.1 repeat also varied. | Repeated-session variation remains unexplained. Small differences from one trial are not enough for a reliable ranking. |
| Does old-task protection help? | Without protection, the four old tasks lost 26.5 points on average. Nonzero beta trials lost much less. | Protection helped in these trials. We have not established a stable best beta. |
| Can an average hide forgetting? | During lessons 1, 7, and 14, the original five-task mean ended unchanged at 41.76%. Task 17 nevertheless fell by 5.8 points. | Report individual tasks alongside the average. |
| Can we suppress the requested task? | In the matched U3/L15 pair, task 3 fell from 27.4% to 10.0%, the chance score for ten classes. It stayed at 10.0% through five later task-15 epochs. | One controlled pair supports functional forgetting through one subsequent lesson. |
| Does unlearning leave other tasks and new learning unchanged? | The seven retained tasks finished 0.34 points lower on average than the learning-only branch, with larger individual differences. New task 15 scored 49.0% versus 54.4% in the control. | Retention was mixed. This pair did not show improved new-task learning after unlearning. |
| Has task-3 information been erased? | No controlled recovery or membership inference test has been run. | Chance accuracy does not answer whether useful information remains in the model. |

The main result is a bounded observation: task-3 accuracy reached chance and stayed there during one later lesson, while other tasks changed by different amounts. This partly agrees with the paper's functional-forgetting claim. It does not reproduce the full paper or establish privacy, permanent forgetting, or results across multiple training seeds. The detailed paper comparison below explains the differences in settings and metrics.

**Decision today:** close the beta diagnostic, the learning-only sequence, and the first paired unlearning comparison. Preserve their evidence and checkpoints. Do not start another beta or gamma search. The next research question, when experiments resume, is whether task-3 information can be recovered under a small, fixed training budget with suitable controls. The recovery experiment is not yet specified or run.

The requested illustrated write-up is complete and committed: [HTML report source](https://github.com/sumitasthana/PATH/blob/main/site/index.html). It includes six labeled charts, data tables, and measurement links. Chart values, desktop and mobile layouts, and printing were checked. A manual GitHub Pages deployment workflow is included. Pages was disabled at the last check; no live deployment is claimed. This report records existing results and adds no new training results.

## Current findings and paused experiments

The user paused further experiments to reflect on agreement with the paper. No recovery run or new training is authorized by that reflection request. Completed evidence is preserved. The comparison below separates limited support from untested claims; the proposed recovery probe remains future work.

The user requested a separate HTML publication of the existing findings. `site/index.html` presents this evidence snapshot in plain English with six bar and line figures, accessible data tables, and links to the original measurements. A manual GitHub Pages workflow publishes only the static site folder. GitHub Pages was disabled when checked; publication still requires enabling the Actions source and running that workflow. This writing task does not resume experiments or change the wiki's Wednesday/Saturday publication schedule.

Learning a new task can lower older-task scores even without an unlearning request, but the size of that loss varies between the two reported beta-0.01 continuations. The earlier Drive run learned task 17 to 53.4% and lost 6.25 percentage points on older tasks on average. The later R2 run learned task 17 to 52.2% and lost 0.05 points on older tasks on average. Task 5 still fell by 3.8 points in the later run. Task 9 finished at 32.6% in the earlier run and 50.0% in the later run. Neither result establishes consistent retention.

The CPU report comparison is complete. The user reported matching starting-model hashes, starting scores, training settings, lesson, and task sizes. The only reported computer or software difference was the Git revision: `6a59648` for the earlier run and `dd7beb1` for the later run. This establishes matching recorded setup; it does not explain the accuracy difference or prove that every source-code path and runtime state matched. The comparison was checked in R2 at `uncle/learning_loss_diagnostic/report_comparison/20261007_drive_vs_r2_01.json`.

The five-value beta batch is complete. With beta 0, task 17 reached 54.0%, but the older-task average fell by 26.5 points. With beta 0.1, task 17 reached 50.8%, the older-task average rose by 1.0 point, and the largest individual old-task drop was 0.4 points. This supports old-task protection as a useful mechanism in this continuation. It does not establish a repeatable optimum.

The beta-0.1 repeat is complete and was read directly from R2, together with both individual reports. It learned task 17 to 55.8%, but the older-task average fell by 2.1 points and task 9 lost 5.6 points relative to the starting model. The first beta-0.1 trial had a 1.0-point average gain and a 0.4-point largest drop. Both reports have matching source hashes, starting scores, full configurations, code versions, and recorded environment dictionaries. The repeat does not confirm near-zero forgetting or explain the session variation.

The beta diagnostic and three-lesson learning-only sequence are now complete. Beta 0.1 was the provisional setting used for the subsequent pair. The sequence's original five-task average finished unchanged, but task 17 lost 5.8 points and task 1 lost 7.8 points during the last lesson. The paired comparison is also complete, as recorded below. Multiple training seeds and checkpoints remain necessary for a generalized claim.

The paired experiment is complete. From the saved model after learning 14, branch A learned 15; branch B unlearned 3, then learned 15. Direct R2 review independently recomputed all eight matching-condition checks from the branch reports. Task 3 fell from 27.4% to 10.0% and stayed at 10.0% through all five later learning epochs. Common retained tasks finished 0.34 points lower on average in B than A, while new task 15 finished 5.4 points lower. Individual retained-task differences were larger than the mean. This is one controlled observation of functional forgetting without observed relapse through one subsequent lesson, not evidence of information erasure.

The new paired runner resolves the recorded representation difference explicitly: its noise loss uses actual generated target parameters, including output scales and offsets, matching the representation used by preservation. It averages summed squared distances to fresh unit Gaussian vectors, as in Algorithm 2 and equation 3. Historical `UnCLe.forget()` retains its raw-output objective. This is a new unlearning implementation for this pair, not a reinterpretation of old results. The source's fixed settings are beta 0.1, gamma 0.01, 100 unlearning steps, ten noise samples, learning and unlearning LR 0.0001, five learning epochs, and batch size 64. Moving U3 to this later checkpoint and using the pilot learning settings makes this a diagnostic variant, not an exact paper reproduction. [Paper, unlearning](https://arxiv.org/html/2509.17530v1#S3.SS1.SSS2)

The completed pair measured task 3 immediately after unlearning and after learning 15, and compared changes on the seven common retained tasks against branch A. Chance accuracy alone does not prove information erasure. Matching recorded conditions does not exclude all GPU numerical variation in a single pair. The notebook saved reports during the run and three new models: A final, B after unlearning, and B final, approximately 1.61 GB total. Existing files were preserved. Local small-model tests checked the objective's value and gradient, control equivalence to ordinary learning, frozen state, absence of old training-data reads, RNG isolation, checkpoint history, invalid-pair rejection, and R2 publication. These tests are separate from the A100 measurements.

Decision after the completed pair: close this first functional-unlearning comparison without another gamma or beta search. Preserve the source, after-unlearning, and final branch checkpoints for a bounded recovery probe. The next residual-knowledge question is whether task-3 performance can be recovered from the saved unlearned model under a fixed adaptation budget and suitable controls. Faster recovery than a control can indicate reusable information, but transfer from retained tasks must be distinguished from task-3-specific residual knowledge. No recovery experiment has been run, and the present accuracy result cannot answer that question.

The completed sequence used notebook 15, section 9, with `MODE = "SEQUENCE"`. It loaded the completed beta-0.1 repeat model after tasks 3, 0, 9, 5, and 17, then learned 1, 7, and 14 in that order, carrying each final model forward. Task 17 was not trained again. This follows the next learning operations in Table 4's Tiny ImageNet sequence 1 while keeping this diagnostic learning-only. Beta stayed 0.1; other settings came from the saved repeat model.

The same three lessons tested four hypotheses: forgetting accumulates across lessons; earlier-learned tasks lose more; a mean hides individual task damage; and learning/protection gradients conflict in sampled parameter groups. The saved reports contain a fixed five-task average relative to the sequence's start, per-task changes, losses from best recorded scores within this sequence, and a first-batch gradient sample from each epoch. Task age and gradient observations are associations, not causal comparisons. The final report and per-lesson checkpoints were saved to R2. Local verification with a small model checked the checkpoint chain and all three lessons; it is separate from the A100 measurements.

An R2 inventory taken after the repeat found 177 objects totaling 13.98 GB. No Tiny ImageNet dataset objects appeared. Most space is in model checkpoints of approximately 537 MB each. The unused beta-0, beta-0.001, and beta-1 final models are initial cleanup candidates totaling 1.61 GB. Older paired-L9, session-replay, and learning-only folders total approximately 7.00 GB including their reports; their historical models need an archive decision before deletion. Preserve all reports, the post-task-5 source, both beta-0.1 models, and active sequence checkpoints. Three new sequence checkpoints will add approximately 1.61 GB. Cleanup is a proposal only: nothing has been deleted and training cells do not delete older evidence.

The earlier cross-session replay problem remains unexplained. Its first recorded mismatch was at training update 1, after matching recorded inputs. A single new beta trial should therefore be treated as diagnostic evidence rather than a settled ranking.

## Reflection against the paper's claims

Overall assessment: the completed diagnostics partly agree with the paper. They support reducing the requested task to chance and maintaining that low accuracy through one subsequent lesson. They do not establish broad retention, improved later learning, privacy protection, or information erasure. This is not a full reproduction.

The paper reports chance-level forget accuracy, low spill, resistance to relapse, and better learning after unlearning in longer sequences. It also reports membership inference results. Its primary tables average three seeds. [Paper, discussion and Tables 1 through 3](https://arxiv.org/html/2509.17530v1#S4.SS4)

| Claim or mechanism | Our observations | Assessment |
| --- | --- | --- |
| Protection reduces ordinary forgetting | Beta 0 lost 26.5 points on older tasks on average. Nonzero beta trials lost much less; beta 0.1's two task-17 trials ranged from a 1.0-point gain to a 2.1-point loss. | Mechanism supported descriptively; optimum and repeatability unresolved. |
| Requested task falls to chance | In the corrected paired runner, task 3 fell from 27.4% to 10.0%. | Supported for this one request. |
| Forgotten task stays low after later learning | Task 3 stayed at 10.0% through all five L15 epochs. | Supported through one lesson, not a long permanence test. |
| Other tasks remain stable | U3 caused a 1.54-point mean retained-task loss at its end, with a 3.2-point largest final drop. Task 17 temporarily lost 21.8 points at update 1. The later paired final mean difference was -0.34 points, masking larger task-level differences. | Partial retention, not consistently stable individual accuracy. |
| Unlearning improves new-task learning | Task 15 reached 54.4% in A and 49.0% in B. | Not supported in this pair; does not refute a longer-sequence saturation claim. |
| Privacy or absence of residual information | No membership inference, recovery, or matched never-learned control has been tested. | Not assessed. Chance accuracy cannot establish erasure. |

Use the paper's definitions when comparing metrics. Its spill is the sum of absolute old-task accuracy changes immediately after an unlearning request. Applying that definition to our seven retained tasks yields 10.8 percentage points. Our 1.54-point mean retained loss and 0.34-point paired final difference are different quantities. Our relapse over the one subsequent lesson is 0.0 points. These single-request results must not be equated with the paper's averages over full sequences and multiple seeds. [Paper, equations 4 and 5](https://arxiv.org/html/2509.17530v1#S4.SS4)

Reasons this does not settle agreement or disagreement: our detailed pair uses seed 0, changes when U3 occurs, uses beta 0.1 rather than the paper's Tiny ImageNet beta 0.01, and uses constant LR 0.0001 rather than the stated 0.001 with a scheduler. Hyperfan classifier, bias, and BatchNorm choices remain documented implementation choices. Earlier raw-output unlearning runs also used a different noise representation from the corrected pair. There is no direct ablation attributing the later success solely to that correction. Full 30-request, multiple-seed results and baseline comparisons have not been reproduced. [Paper, implementation and Appendix C](https://arxiv.org/html/2509.17530v1#S4.SS1)

The defensible research statement is: in one controlled diagnostic, task-3 accuracy was suppressed to chance and remained there during one later lesson, while retained-task effects varied and later-task accuracy was lower than the learning-only control. Whether task-specific information remains recoverable is still unanswered. Further experiments are paused at the user's request.

## R2 continuation reported on 7 October

The user supplied notebook 15's output for one beta value, 0.01. The default selected source is the confirmed four-task Hyperfan model above. The screenshot does not include the source hash, actual configuration, or environment metadata; these must be read from the saved reports before calling this an exact repeat. The receipt date is 7 October, while the experiment name contains 6 October.

Reported final model: `uncle/learning_loss_diagnostic/20261006_learning_loss_r2_01/beta_0_01/checkpoint.pt` in R2. Each epoch report and the final model were reported as checked in R2. The final history-check and GPU-release cell was still pending in the screenshot.

| Epoch | New-task loss | Beta times protection | Task 3 (%) | Task 0 (%) | Task 9 (%) | Task 5 (%) | Task 17 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.8008 | 0.8181 | 20.6 | 29.4 | 41.4 | 37.2 | 40.2 |
| 2 | 1.4441 | 0.4781 | 31.2 | 31.6 | 49.6 | 52.8 | 39.6 |
| 3 | 1.2688 | 0.4108 | 22.6 | 31.0 | 36.4 | 45.8 | 39.6 |
| 4 | 1.1410 | 0.4133 | 26.4 | 34.2 | 56.2 | 48.2 | 54.6 |
| 5 | 0.9835 | 0.4025 | 28.8 | 31.4 | 50.0 | 51.0 | 52.2 |

| Task | Confirmed source score (%) | Earlier Drive final (%) | Later R2 final (%) | Later change from source (points) |
| --- | --- | --- | --- | --- |
| 3 | 27.0 | 24.6 | 28.8 | +1.8 |
| 0 | 31.2 | 28.0 | 31.4 | +0.2 |
| 9 | 48.4 | 32.6 | 50.0 | +1.6 |
| 5 | 54.8 | 51.2 | 51.0 | -3.8 |
| 17 | Not supplied | 53.4 | 52.2 | Not calculated |

The later mean old-task score is 40.30%, against the source's 40.35%. The notebook printed the change rounded to one decimal place as `-0.0 points`; the unrounded change is -0.05 points. An unchanged average does not mean that every task was preserved.

Task 9's final score differs by 17.4 points between the runs, while task 17 differs by 1.2 points. This is a repeatability concern for interpreting smaller beta effects. Moving files to R2 is not a demonstrated cause of the retention improvement. The training, checkpoint, configuration, dataset, and learning-diagnostic files have no Git changes between notebook 14's pin `6a59648a5512724f94917ed9b97a12807c68809b` and notebook 15's pin `dd7beb173fb957467116c71b6ffe39e562799016`. Runtime versions and actual report metadata have not yet been compared.

The next check uses the already saved `report.json` files to compare `source_sha256`, `starting_accuracies`, `config`, `code_version`, and `environment`. This is a CPU metadata check, not another training run or a request for extended kernel tracing. Gradient samples are not shown in either supplied output, so loss magnitudes alone do not explain the discrepancy.

## Beta batch completed on 7 October

The user supplied the saved batch comparison JSON. All five rows record the same source SHA-256, `9741c921a34ff68a8e15f2f80469adacf06cd0c72b9c3b8fb65da14e92d1586f`, for the confirmed model after tasks 3, 0, 9, and 5. The batch records code revision `dd7beb173fb957467116c71b6ffe39e562799016`, PyTorch `2.11.0+cu130`, CUDA `13.0`, and an NVIDIA A100-SXM4-40GB. Four new trials shared that runtime. Beta 0.01 was reused from an earlier session.

| Beta | Task 17 accuracy (%) | Mean old-task change (points) | Largest old-task drop (points) | Result source |
| --- | --- | --- | --- | --- |
| 0 | 54.0 | -26.50 | 41.6 | New batch |
| 0.001 | 56.8 | -1.55 | 5.8 | New batch |
| 0.01 | 52.2 | -0.05 | 3.8 | Earlier-session reference |
| 0.1 | 50.8 | +1.00 | 0.4 | New batch |
| 1 | 52.4 | -1.30 | 3.8 | New batch |

| Stage or beta | Task 3 (%) | Task 0 (%) | Task 9 (%) | Task 5 (%) |
| --- | --- | --- | --- | --- |
| Starting model | 27.0 | 31.2 | 48.4 | 54.8 |
| 0 | 17.8 | 14.4 | 10.0 | 13.2 |
| 0.001 | 22.4 | 31.6 | 52.2 | 49.0 |
| 0.01 | 28.8 | 31.4 | 50.0 | 51.0 |
| 0.1 | 29.0 | 30.8 | 49.8 | 55.8 |
| 1 | 25.4 | 32.6 | 44.6 | 53.6 |

Analysis: nonzero protection substantially reduced forgetting relative to beta 0 in these observations. Beta 0.001 had the highest new-task score; beta 0.1 had the smallest worst old-task drop. Increasing beta to 1 did not improve retention further. Each value has only one result in this table, and the earlier beta-0.01 continuation showed substantial run-to-run variation. The batch therefore identifies a candidate setting, not a statistically established ranking or a general reproduction of the paper.

The aggregate file reports completed trial results. Individual epoch reports and checkpoints have not been independently inspected in this workspace. The file alone does not confirm that Colab released the GPU afterwards.

Evidence: [original batch comparison JSON](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_beta_batch_comparison.json). R2 key: `uncle/learning_loss_diagnostic/20261007_beta_batch_01/beta_batch_comparison.json`.

## Beta 0.1 repeat read from R2

Local credentials enabled direct read access to the private R2 bucket. The comparison file and both individual reports were downloaded and verified against their R2 SHA-256 metadata. No model checkpoint was downloaded or independently loaded for this review.

| Measurement | First beta 0.1 | Repeat beta 0.1 |
| --- | --- | --- |
| Task 17 accuracy (%) | 50.8 | 55.8 |
| Mean old-task change (points) | +1.00 | -2.10 |
| Largest old-task drop (points) | 0.4 | 5.6 |
| Task 3 final accuracy (%) | 29.0 | 23.6 |
| Task 0 final accuracy (%) | 30.8 | 30.8 |
| Task 9 final accuracy (%) | 49.8 | 42.8 |
| Task 5 final accuracy (%) | 55.8 | 55.8 |

Both runs used seed 0, beta 0.1, five epochs, batch size 64, and learning rate 0.0001. Their complete recorded configurations and environment dictionaries match. Each report records 395 updates and status `complete`. The repeat's task-9 score is 7.0 points below the first trial and 5.6 points below the source model. The new-task score is 5.0 points higher. Losses and scores already differ in the first epoch summary; these reports do not record the first individual update where the runs diverged.

Analysis: protection remains associated with much less forgetting than the beta-0 trial in the batch, but this repeat weakens the initial impression that beta 0.1 preserves every old task almost exactly. Two sessions with one saved source model do not establish a stable optimum, different-seed generalization, or longer-sequence retention. The comparison does not show a recorded setup mismatch that explains the differences. Final reports alone do not confirm that the GPU was released afterwards.

Evidence: [repeat comparison](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_beta01_repeat_comparison.json), [first trial report](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_beta01_first_report.json), and [repeat report](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_beta01_repeat_report.json). R2 comparison key: `uncle/learning_loss_diagnostic/20261007_beta01_repeat_01/repeat_comparison.json`.

## Learning sequence 1, 7, 14 completed

The user supplied the R2 key for the completed sequence comparison. Direct review downloaded that file and all three individual lesson reports with SHA-256 verification. Each report has status `complete`, five epochs, matching final scores and source fingerprint, and a remote checkpoint metadata fingerprint matching the aggregate. The checkpoint chain matches the selected repeat model, followed by the outputs of lessons 1 and 7. Models were not downloaded or independently loaded during this review.

| Stage | Task 3 (%) | Task 0 (%) | Task 9 (%) | Task 5 (%) | Task 17 (%) | Task 1 (%) | Task 7 (%) | Task 14 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Starting repeat model | 23.6 | 30.8 | 42.8 | 55.8 | 55.8 | Not learned | Not learned | Not learned |
| After learning 1 | 25.8 | 27.6 | 47.6 | 56.8 | 55.4 | 54.2 | Not learned | Not learned |
| After learning 7 | 26.2 | 32.6 | 50.6 | 56.4 | 52.4 | 56.8 | 35.2 | Not learned |
| After learning 14 | 27.4 | 31.6 | 45.6 | 54.2 | 50.0 | 49.0 | 31.8 | 47.6 |

The fixed original five-task mean was 41.76% initially, 42.64% after task 1, 43.64% after task 7, and 41.76% after task 14. The largest drop relative to those tasks' sequence-start scores increased from 3.2 to 3.4 to 5.8 points. Task 17 declined after every new lesson. Task 1 lost 7.8 points during learning 14, from 56.8% to 49.0%, and finished 5.2 points below its own first completed lesson. Task 7 lost 3.4 points during learning 14, from 35.2% to 31.8%. Its largest loss from a within-sequence epoch peak was 13.0 points; this includes variation during its own training and is not solely forgetting caused by subsequent tasks.

Hypothesis findings:

- Cumulative forgetting: observed for task 17, but not as a falling fixed-cohort average. No original task collapsed toward 10% in this sequence.
- Earlier task age: no simple pattern supports greater damage to the earliest tasks. Tasks 3, 0, and 9 improved relative to the starting repeat model, while later task 17 declined. Task identity and age remain confounded.
- Averages hiding damage: supported descriptively. The unchanged final original-task mean hides losses of 5.8 points on task 17 and 1.6 points on task 5.
- Gradient conflict: 15 first-batch samples were recorded, one per epoch. The protection gradient is zero at each lesson's first update, so each group has four defined cosine samples per lesson. Across the 12 defined samples, the shared trunk has four negative cosines; the weight head has none. Mean trunk cosines by lesson are approximately 0.032, -0.001, and 0.006. These samples do not show persistent strong opposition or establish the cause of individual score drops. Per-group gradient sizes and ratios remain in the evidence.

Decision: close this learning-only diagnostic. It supplies observed ordinary-forgetting measurements and a saved eight-task model for a subsequent paired unlearning test. Protection helps, but retention is not exact and no generalized result is established. The next test should compare a learning-only branch with an unlearning-plus-learning branch from the same checkpoint. The precise unlearning request and protection implementation need to be fixed before training; moving an unlearning request to this later checkpoint would be a diagnostic variant of the paper's sequence, not an exact reproduction.

Evidence: [sequence comparison](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_sequence_comparison.json), [task 1 report](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_sequence_task_1_report.json), [task 7 report](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_sequence_task_7_report.json), and [task 14 report](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_sequence_task_14_report.json). R2 aggregate: `uncle/learning_loss_diagnostic/20261007_learning_sequence_beta01_01/sequence_comparison.json`. Final checkpoint: the same prefix followed by `/lesson_03_task_14/checkpoint.pt`.

## Paired U3 and L15 completed

The user supplied `uncle/learning_loss_diagnostic/20261007_U3_L15_pair_01/paired_comparison.json`. The comparison and two full branch reports were read directly from R2 and verified against their SHA-256 metadata. Source hash `4b8849bece1e82defd7c0393dd4c5f9acefe4caa87a30fba932e5179f300a980` matches the preceding learning sequence's final checkpoint. Runner revision is `bda6040bff7d3927d78f8289f3ecd584e529508f`. Both branches completed 395 learning updates on the same A100 runtime. Branch B completed 100 unlearning updates first.

All eight controls were independently recomputed from the individual reports and passed: starting model, starting scores, CPU RNG, CUDA RNG, new-task code, batch order, input values, and update count. Reported frozen-state fingerprints also match before and after both branches. Remote metadata for A final, B after unlearning, and B final matches each recorded checkpoint fingerprint. Models were not independently downloaded or loaded during this review. The reports alone do not verify that Colab released the GPU afterwards.

| Task | Start (%) | After U3 (%) | A: L15 final (%) | B: U3 then L15 final (%) | B minus A (points) |
| --- | --- | --- | --- | --- | --- |
| 3, forget target | 27.4 | 10.0 | 35.2 | 10.0 | -25.2 |
| 0 | 31.6 | 30.0 | 36.8 | 34.2 | -2.6 |
| 9 | 45.6 | 42.4 | 47.0 | 41.4 | -5.6 |
| 5 | 54.2 | 53.0 | 46.0 | 55.6 | +9.6 |
| 17 | 50.0 | 50.0 | 45.0 | 47.8 | +2.8 |
| 1 | 49.0 | 46.8 | 52.0 | 46.6 | -5.4 |
| 7 | 31.8 | 30.8 | 29.4 | 30.4 | +1.0 |
| 14 | 47.6 | 46.0 | 45.8 | 43.6 | -2.2 |
| 15, new task | Not learned | Not learned | 54.4 | 49.0 | -5.4 |

Task 3 first reached 10.0% at the recorded step-20 evaluation and stayed there at every subsequent unlearning evaluation through step 100. It stayed at 10.0% at each of the five L15 epoch evaluations. Branch A's task-3 score instead rose to 35.2%. This supports functional suppression and absence of observed accuracy relapse over this one later lesson. It does not demonstrate erasure, permanence across a longer sequence, or a repeatable different-seed effect.

For the same seven retained tasks, mean changes relative to the starting checkpoint were -1.54 points after U3, -1.11 points after A's L15, and -1.46 points after B's U3 and L15. The paired final difference is -0.34 points. This small mean difference is descriptive and should not be treated as a resolved statistically reliable effect. Task-level changes do not cancel scientifically merely because they cancel in the average: task 9 is 5.6 points worse in B, while task 5 is 9.6 points better. New task 15 is 5.4 points worse in B.

There was substantial temporary retained-task damage. The first unlearning update lowered task 17 from 50.0% to 28.2%, a 21.8-point drop. Much of the retained performance recovered during the remaining unlearning updates. The initial preservation penalty is zero at the starting snapshot; this observation does not establish the cause or prove that the noise term dominates all gradients. Its large scalar loss includes the high-dimensional Gaussian squared-distance contribution and cannot by itself quantify gradient dominance.

Evidence: [paired comparison](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_paired_comparison.json), [learning-only report](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_pair_learning_only_report.json), and [unlearn-then-learn report](https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/20261007_pair_unlearn_then_learn_report.json).

## Evidence and maintenance

Original checkpoints and GPU report files remain at their reported Drive paths; they were not downloaded or independently inspected here. Screenshots and pasted outputs are labelled as user-reported results. Missing gradients, environment details, timings, and fingerprints are not reconstructed.

The tracked measurements are in [registry.json](https://github.com/sumitasthana/PATH/blob/main/docs/experiments/registry.json), [manifest.json](https://github.com/sumitasthana/PATH/blob/main/docs/experiments/manifest.json), and the CSV and text evidence under [docs/experiments](https://github.com/sumitasthana/PATH/tree/main/docs/experiments). Figures retain their dated source data. The registry covers the older archived runs; the newly reported results above have not been added to that structured registry.

Record subsequent experiments on the page for their reporting date and update the main index. Keep the question, starting model, actual settings, observed results, limitations, and next decision together. Keep measurements in their evidence files and link them here. Git history retains previous versions of the page.
