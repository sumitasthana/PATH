# Experiment records and publication

Keep `docs/wiki/Experiment-trajectory.md` as a short index with the latest findings. Keep detailed experiment narratives in dated `docs/wiki/Experiment-trajectory-YYYY-MM-DD.md` pages. Use an explicit date range for historical records spanning several days; do not invent missing run dates. Record results as the user shares them, including failed and interrupted runs, intended and actual settings, starting checkpoints, missing evidence, interpretations, and the next decision. Commit these working records to the main repository. Do not create separate experiment reports or setup-help wiki pages unless the user explicitly requests them.

Publish accumulated updates to the GitHub wiki's Experiment trajectory index and dated pages on Wednesdays and Saturdays at 6 p.m. America/New_York. Other-day publication requires an explicit user request. The local Windows task `CARK-Experiment-Trajectory-Publish` runs `scripts/publish_experiment_trajectory.py --publish`; it publishes the committed main-repository records and their figures. It does not read private chat history or discover new experiment results. Keep the working records current during the conversation.

# Notebooks

A notebook change is not finished until the link is in the user's hands. After creating or updating any notebook in `notebooks/`, commit it, push it, and give the user its Colab link in the reply:

```
https://colab.research.google.com/github/sumitasthana/PATH/blob/main/notebooks/<file>.ipynb
```

Colab serves the pushed file, so an unpushed commit gives a link that opens the old version or nothing at all. Do all three steps in the same turn as the change.

# General

Use the user's configured Git identity. Add no assistant, tool, or coauthor attribution. Preserve unrelated working changes. Write plain English for a non-native freshman reader. Do not invent missing results, metadata, citations, or conclusions. Do not use U+2014 in newly written prose.
