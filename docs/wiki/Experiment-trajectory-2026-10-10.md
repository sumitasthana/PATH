# Experiment trajectory: 10 October 2026

## Seed 1 comparisons complete

The user supplied the notebook-19 progress panel and session record `20261010T023852185516Z`. Both order-01, seed-1 branches, U3/L15 and U14/L15, completed. All four selected seed-1 jobs and all 13 of its requests are complete. The panel reports 8 of 12 jobs, 4 of 6 comparisons, and 26 of 39 requests complete across the narrowed study. Only seed 2 remains: its eight-lesson source, L15-only control, U3/L15 branch, and U14/L15 branch. This is four jobs, two comparisons, and 13 requests. The other orders and task-5 branches remain deferred in the original study plan.

A read-only R2 check at 12:21 a.m. America/New_York on 10 October independently verified the new session log. It records both seed-1 branches complete and a duration of 6,073.955492 seconds, approximately 1 h 41 m 14 s. The session began on 9 October at 22:38:52 Eastern daylight time and continued past midnight. The user-supplied panel gives the latest verified boundary as `2026-10-10T04:19:57.282775+00:00`. No seed-1 accuracy results were included in this update.

Seven verified R2 session logs now total 33,067.539814 seconds, rounded to 9 h 11 m 8 s. This includes the interrupted upload session and excludes earlier notebook setup, gaps, and sessions without an uploaded final log. It is runner wall time, not a complete measure of billed GPU time. The latest local audit is `outputs/r2-review/session-times-2026-10-09/sessions.json`; its internal check timestamp identifies this refreshed snapshot.

The requested Excel tracker was updated as `outputs/experiment-tracker/Paired_Study_Experiment_Tracker_2026-10-10.xlsx`. It adds the seventh session, marks both seed-1 comparisons complete, updates the grand total and remaining work, and leaves seed-1 accuracy fields blank because no paired effects were supplied. Workbook checks confirmed seven session rows, six comparison rows, correct completion labels, and the duration total. The dated 9 October workbook is preserved.

Next: change only SEED to 2 in notebook 19 and run all on A100. Keep SEED 2 on subsequent sessions until its four jobs complete. Use notebook 20 on CPU to inspect and save multi-seed effects when available. No new GPU training was started by this record or workbook update. Wiki publication remains on its scheduled days.

## Seed 2 source saved and tracker refreshed

The user supplied session `20261010T042423429787Z` and a progress panel showing the seed-2 source resumable after four completed source lessons, with Learn 17 active. The panel reports 30 of 39 requests complete across the narrowed study; jobs remain at 8 of 12 and comparisons at 4 of 6. Nine requests are unfinished. The latest boundary reported by the panel is `2026-10-10T06:12:02.796895+00:00`. Its active epoch and step were not supplied. Keep SEED 2 and run all on A100 to resume.

A read-only R2 session audit at 1:09 p.m. Eastern on 10 October independently verified the eighth log, recording the seed-2 source as resumable with elapsed time 6,466.819694 seconds (1 h 47 m 47 s). All eight logs total 39,534.359508 seconds, rounded to 10 h 58 m 54 s. The detailed request count above remains user-reported; the session audit did not reread the active progress pointer. The October-10 Excel tracker now includes all eight sessions, the correct seed-2 label, the updated duration total, and the user's 30/39 progress snapshot. Workbook checks passed. No new accuracy result was supplied.

## Full planned-job tracker

The user requested that the Excel tracker include all planned work so remaining jobs are visible. A read-only R2 audit at 1:14 p.m. Eastern on 10 October checked all 45 progress pointers and validated their referenced report/checkpoint metadata. Eight jobs are complete, one is resumable, and 36 are pending. The seed-2 source has six completed learning requests and an active Learn 7 request saved at epoch 0, step 0. The narrowed study therefore has 32 of 39 requests complete, seven unfinished requests, and four unfinished jobs. Its two seed-2 comparisons remain unfinished.

The Excel tracker now includes a Planned jobs sheet with all 45 original jobs. Each row shows selected versus deferred scope, order, seed, source sequence, planned requests, purpose, saved status, planned/completed/remaining request counts, active saved request, timestamp, and job ID. Twelve jobs belong to the narrowed study and 33 are deferred. Overview separates these totals. Start dates and durations are not invented for pending work; session durations remain in the Sessions sheet. Checks confirmed 45 job rows, 12 selected jobs, 39 selected requests, and 135 requests across the original plan. No new accuracy effect was assessed.

## Seed 2 source and control complete

The user supplied session `20261010T163836495156Z`, with status saved and elapsed time 6,584.078991 seconds (1 h 49 m 44 s). The seed-2 source and L15 control completed. U3/L15 returned resumable with Unlearn 3 active; the exact saved step was not supplied. U14/L15 remains pending. The panel reports 10 of 12 jobs, 4 of 6 comparisons, and 35 of 39 requests complete. Its latest verified boundary is `2026-10-10T18:28:12.115066+00:00`. This update records user-supplied output; remote state was not independently reread here.

Next: reconnect to A100, keep SEED 2 and the same study ID/settings, and run all. The completed source and control are skipped, U3/L15 resumes, and U14/L15 follows if the session budget permits. Two branch jobs and four requests remain unfinished; U3 already has partial progress. Once the panel shows 12/12 jobs and 6/6 comparisons complete, stop GPU execution and run notebook 20 on CPU. Its verified R2 exports provide the three-seed means, variation, retained-task changes, and forgotten-task gate/relapse checks. Do not start deferred orders or targets for this narrowed study.

## Requested draft report and verified seed-1 effects

The user explicitly requested a report using available results and placeholders for unfinished comparisons. A fresh CPU review read verified R2 reports and checkpoint metadata without downloading models. Four selected comparisons pass all recorded pairing checks; seed-2 U3/L15 remains resumable and U14/L15 pending. In all four completed comparisons, the target accuracy falls from above chance to 10.0% after unlearning and remains 10.0% after L15. The fixed 12% forgetting gate is met, with no recorded relapse above that gate during the evaluated L15 epochs. These results establish observed classification suppression over one new lesson; they do not establish erasure or recovery resistance.

| Seed | Branch | L15 control | L15 branch | L15 difference | Mean retained difference | Largest final retained drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | U3/L15 | 50.8 | 50.8 | 0.0 | -1.086 | 11.0 |
| 0 | U14/L15 | 50.8 | 48.2 | -2.6 | -1.600 | 14.2 |
| 1 | U3/L15 | 44.0 | 47.2 | +3.2 | -1.571 | 11.6 |
| 1 | U14/L15 | 44.0 | 47.8 | +3.8 | -1.743 | 14.0 |

Accuracy columns are percentages; differences and drops are percentage points. Provisional task-15 means and sample SD across seeds 0 and 1 are +1.60 ± 2.26 points for U3/L15 and +0.60 ± 4.53 for U14/L15. These are partial summaries; final three-seed means remain pending. The CPU review independently verifies the previously user-supplied seed-0 values and fills the missing seed-1 effects and forgotten-task scores. Nine recorded sessions total 46,118.438499 seconds, rounded to 12 h 48 m 38 s.

The requested draft is [Paired study draft](../reports/Paired-study-2026-10-10-draft.md), with a committed evidence snapshot, PNG/SVG figure, and rebuild script. It separates actual diagnostic settings, validated results, provisional averages, limitations, and placeholders for both seed-2 branches and final runtime/conclusions. Figure and numeric checks passed. No GPU training was performed to create the draft.

The user requested an HTML copy for easier reading. The [HTML draft](../reports/Paired-study-2026-10-10-draft.html) contains the same report with readable tables, section links, and an embedded figure. It can be opened as a standalone local file without a server or an image download. Browser printing can save a PDF. The report builder now produces both Markdown and HTML so future result updates keep them aligned. HTML checks confirmed six tables, the embedded figure, portable evidence links, and the seed-2 placeholders. This is a local and main-repository artifact, not a website deployment.

## Report readability revision

The user found the numbers, language, and chart confusing. The Markdown and HTML drafts now separate three questions: forgetting the requested task, learning task 15, and changes to other tasks. Each result table defines its measure and gives a reading example. A directly labeled bar chart shows absolute task-15 accuracy separately for each completed seed. The main report reserves final averages for the completed three-seed study and marks the two unfinished comparisons as pending. Technical settings and individual-task differences appear after the findings. This revision changes presentation only; it uses the same verified four-comparison snapshot and introduces no new training results.

## Seed 2: only the final learning request remains

The user reported session `20261010T201745955185Z` saved after 6,384.282779 seconds (1 h 46 m 24 s). The seed-2 U3/L15 branch is complete. The U14/L15 branch is resumable at Learn 15. The selected study now has 11/12 jobs, 5/6 comparisons, and 38/39 requests saved complete. Seed 2 has 3/4 jobs, 1/2 comparisons, and 12/13 requests complete. The latest verified R2 save reported by the notebook is 2026-10-10T22:03:47.034314+00:00. No new accuracy scores were included, so the report's numeric result tables retain their earlier four-comparison evidence snapshot. Progress counts are updated separately.

The session used max_jobs 4, a 120-minute soft session limit, and a 15-minute save reserve. Its recorded plan SHA-256 is `a51f342dfdf1faef9dd22f06426db5ad1e5ac7800a6b9318df055400cf310171`. Adding this user-supplied duration to the nine reviewed session logs gives 52,502.721278 seconds, rounded to 14 h 35 m 3 s across ten recorded sessions. The latest session has not been independently reviewed from R2 in this update. This is recorded runner time, not billed GPU time.

Next: reconnect to A100, keep SEED 2 and the same study ID, and run all. Completed jobs are skipped. The last Learn 15 request resumes from its verified save. When the panel reaches 12/12 jobs, 6/6 comparisons, and 39/39 requests, run notebook 20 on CPU and save its R2 summaries. Review the final scores before filling the report's remaining numeric placeholders.

## Selected paired study training complete

The user reported session `20261010T220801006167Z` completing the seed-2 U14/L15 branch. All selected work is now saved complete: 12/12 jobs, 6/6 comparisons, and 39/39 requests across seeds 0, 1, and 2. Seed 2 has 4/4 jobs, 2/2 comparisons, and 13/13 requests complete. The notebook reports its latest verified R2 save at 2026-10-10T22:42:48.286523+00:00. Its generic instruction to run all again does not imply unfinished selected training.

The session records 2,117.425159358 seconds (35 m 17 s), max_jobs 4, a 120-minute soft budget, and a 15-minute reserve. Plan SHA-256 remains `a51f342dfdf1faef9dd22f06426db5ad1e5ac7800a6b9318df055400cf310171`. Adding both latest user-supplied session durations to the nine previously reviewed logs gives 54,620.146437 seconds, rounded to 15 h 10 m 20 s across eleven sessions. This is recorded runner time, not billed GPU time. The final two session records have not yet been independently reviewed from R2.

No seed-2 accuracy scores accompanied this completion panel. Training completion does not establish a successful forgetting threshold or the final paired effects. The report now marks all training complete while retaining its verified four-comparison numeric snapshot. Next: run notebook 20 on CPU with the same study ID and SAVE_SUMMARY_TO_R2 enabled, review all six comparison reports, and fill the three-seed means, variation, forgetting checks, and final conclusions. No further A100 session is needed for the selected scope; deferred orders remain deferred.

## Final CPU review and saved summaries verified

The user confirmed running notebook 20. A read-only R2 review verified all six selected comparisons and all recorded pairing checks, without downloading models or training. All six notebook-20 exports passed their SHA-256 checks under `uncle/paired_generalization/paired_generalization_v2/summaries/20261011T004607328665Z/`: summary.json, study.json, comparisons.csv, retained_tasks.csv, paired_effects.png, and paired_effects.svg. The saved summary's comparison rows and plan hash match the independent review. The export timestamp is 11 October UTC, corresponding to 10 October in America/New_York.

Seed 2 completes the missing numeric results. Its baseline task-15 accuracy is 34.4%. U3/L15 scores 47.4%, a +13.0-point difference, with a -1.20-point mean change on retained tasks and a largest retained loss of 4.2 points on task 0. U14/L15 scores 32.6%, a -1.8-point difference, with a +0.29-point mean retained change and a largest loss of 2.6 points on task 9. Target task 3 starts at 38.4%; target task 14 starts at 41.8%. Both reach 10.0% after unlearning and remain 10.0% after learning task 15. Across all six comparisons, no saved L15 evaluation exceeds the 12% forgetting threshold after the gate is met.

| Request | Three-seed mean task-15 difference | Sample SD | Three-seed mean retained difference | Sample SD |
| --- | ---: | ---: | ---: | ---: |
| U3/L15 | +5.40 | 6.77 | -1.29 | 0.25 |
| U14/L15 | -0.20 | 3.49 | -1.02 | 1.13 |

Differences and standard deviations are in percentage points. Means use paired branch-minus-baseline effects separately for each target. The task-3 average is strongly influenced by seed 2's +13.0 points. The task-14 mean is near zero, with both positive and negative seed effects. The largest individual retained loss ranges from 2.6 to 14.2 points. Three seeds, shared controls, and one learning order limit generalization. These results support observed classification suppression during one subsequent task, not erasure, privacy, or recovery resistance.

All eleven session logs were independently read and verified from R2. Their durations sum to 54,620.146437 seconds, rounded to 15 h 10 m 20 s. The Markdown and standalone HTML report now show all three seeds, separately calculated means and standard deviations, a three-run labeled accuracy chart, and no remaining numeric placeholders for the selected study. Deferred orders and targets remain outside this result.

## Five-minute presentation and interactive experiment tree

The user requested a visual presentation centered on accuracy across three request sequences and three seeds. The [standalone HTML presentation](../reports/Paired-study-2026-10-10-presentation.html) has four slides with labeled task-15 bars, forgotten-task line charts at three measured stages, retained-task change bars, and findings with limitations. Presenter notes, fullscreen, keyboard navigation, and landscape PDF printing are included. The Explore experiments tab lists all nine continuation IDs grouped by seed. Selecting a list entry or branch node highlights the matching branch and displays its results, baseline identity, requested-task stages, and individual retained-task changes. Source task nodes identify their learning position; unavailable intermediate scores are stated directly.

The presentation embeds the committed verified evidence and works offline. Browser checks exercised all nine list selections, source nodes, branch selection, tabs, arrow navigation, notes, fullscreen, and print visibility. PDF validation confirmed four pages with their final takeaway text present. Numeric checks cover chart scores, paired effects, and retained-task exclusions. This adds a presentation, not new experiment evidence or a website deployment.

## Mentor-facing public summary

The user requested a concise additional GitHub Pages report, written for a non-native English-speaking freshman undergraduate. The [mentor summary](https://sumitasthana.github.io/PATH/paired-study.html) presents the same verified three-seed evidence using one labeled bar chart and five compact tables for settings, task-15 scores, paired averages, forgetting stages, and individual retained-task losses. It states the seed-2 contribution, variation, limited scope, and absence of an erasure claim. The homepage links to this additional page, which also links to the four-slide presentation and interactive tree. Desktop and mobile checks confirmed all nine bars, five tables, working local presentation link, and no page-level horizontal overflow. The user explicitly authorized publishing this page; it adds no new experiment results.

## Public report URL correction

The user reported a 404 at the shared root report URL after publication. GitHub API inspection confirmed Pages uses the legacy main-branch repository-root source, while the manual workflow uploads only site/. The site-folder report remained available at /CARK/site/paired-study.html. A generated root-level report now preserves /CARK/paired-study.html and links to the site-folder homepage and presentation. The correction relies on the configured branch deployment and does not change Pages settings.

## Original public report updated with the completed extension

The user requested updating /CARK/site/index.html after completion of the larger study. The page now leads with the verified three-seed extension, all nine task-15 scores, paired averages, retention-check definitions, and links to the concise report and interactive tree. Header counts and conclusions reflect six verified comparisons. Earlier one-seed diagnostics remain labeled historical and are excluded from the new averages. The paper-assessment table now distinguishes mixed task-15 effects and incomplete retention from successful observed forgetting. No new measurements are introduced.
