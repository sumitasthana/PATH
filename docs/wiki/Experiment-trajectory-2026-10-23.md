# Experiment trajectory: 23 October 2026

[Back to the experiment index](Experiment-trajectory)

## Review of the EuroSAT plan, and a correction to that review

No training was run today. This page first recorded a review of the [22 October plan](Experiment-trajectory-2026-10-22) against the adapter study in `reference/adapter-plasticity-study`. The main claim of that review was wrong. This page now records both the claim and its correction, because the wrong version was committed and the record should show what happened.

### What the review claimed

That the 22 October plan compares a generator at t = 0, which has random weights, against one that has learned eight tasks, and that this comparison cannot answer a plasticity question. The reasoning was that a random generator has never produced working convolution filters, so it would learn poorly from 800 images in 250 steps, and the t = 8 model would win for a reason unrelated to plasticity.

On that basis the review recommended deferring the EuroSAT plan and restating the 20 and 21 October finding.

### Why it was wrong

The 20 and 21 October record already contains the numbers that settle it. At t = 0 the full probe reaches 51.87, 53.09 and 52.51 percent on ten-class tasks. Chance is 10 percent.

The untrained generator learns the probe task well inside the budget. The gap to the eight-task model is about 5 points, not the collapse the review assumed. The claim was a prediction presented as a fact, and the measurement contradicting it was already in our own record.

The review also leaned on a second mistake. It cited the 21 October observation that a frozen generator with a fresh task code sits at chance, and treated that as evidence that the reference model is weak. That observation is about the code-only control arm, where the generator cannot change. It says nothing about the full probe, which trains the generator. The two arms have to be read separately.

The recommendation to defer the EuroSAT plan does not follow and has been withdrawn. The plan's status is planned again.

### A related error in uncle/probe.py

The warning on `relative_contribution` said that because the control at t = 0 sits near chance, G_0 is close to zero and the ratio divides by almost nothing.

That does not follow. G_0 is the gap between the two arms, not the control's own score:

```text
G_0 = full-probe performance - code-only performance
```

With the measured values, G_0 is about 51.9 minus 10.8, or 41 points. Nothing is unstable. The 21 October record confirms it: the ratio computed to 1.174, 1.076 and 1.124.

The real limitation is different, and that record states it correctly. Our control is pinned near chance at every point, so the gap tracks the full probe's accuracy almost exactly and the ratio repeats that accuracy in different units. In the adapter study the control is a linear probe on frozen pretrained features, which scores well and moves as the features change, so their ratio carries information their raw accuracy does not.

The docstring has been corrected.

### What the t = 0 comparison actually measures

The comparison measures learnability relative to initialization. It does not separate transfer from plasticity changes, and it does not rule out deterioration relative to an earlier trained checkpoint.

Two limits follow. Useful transfer can hide deterioration, because a model that has become harder to train may still score higher on a related probe. And a rise early in the sequence followed by a fall later cannot be seen when only the endpoints are measured.

The correct reading of the existing result: after eight tasks the generator performed better on the tested probes than a freshly initialized generator, under the chosen training budgets. This does not rule out deterioration relative to an earlier trained checkpoint, and it does not separate transfer from changes in plasticity.

### What survives from the review

Adding trained reference points. Not because t = 0 is invalid, but because one trained reference does not fix the problem either. A single task does not guarantee a capable reference, and transfer can affect a comparison between any two trained states. Keep t = 0 and add t = 1, 2, 4 and 6 where practical, so the shape across the sequence is visible instead of one difference between endpoints.

This needs the sequence trained again. Notebook 29 records that no mid-stream checkpoints were kept. The review wrongly blamed `uncle/checkpoint.save` for this: that function writes to whatever path it is given, so per-task files need only a per-task path. The states were not saved; the function did not prevent saving them.

Reading the two probe arms separately, as described above.

Logging diagnostics between probe points, which we do not do at all. The 22 October page now lists them with their real costs. Two corrections to the review's version: the fraction of silent channels needs a forward pass over a defined input set, not weights alone, and relative weight change across the sequence needs the earlier states, which an endpoint checkpoint cannot reconstruct.

These diagnostics are exploratory. The condition number of a generator head is not an established equivalent of the adapter study's LoRA factor conditioning. And the review's claim that a small weight movement would make any plasticity experiment on this stream pointless is wrong: weight distance does not determine learning ability, and geometry cannot replace the probe.

The revised compute estimate. With five repetitions at the reference, two at each later state, and both arms, two model states come to 3 x (5 + 2) x 3 x 2 = 126 probe runs, not the 90 of the original design. Each further state adds 36. The plan now also states which repetitions are paired when the counts differ.

### What does not port from the adapter study

Two of its mechanisms have no counterpart here. One concerns the Cayley map used by orthogonal finetuning, the other the two factor matrices in LoRA. We have neither.

Its argument that capacity cannot shrink also does not carry over. That holds there because the backbone is frozen and the adapter is a rotation or a low-rank term, so the set of reachable weights is fixed by construction. Our generator is the model and can lose expressivity, so we cannot assume that a slowdown means bad coordinates.

On re-anchoring, the defensible statement is narrower than the review's. We currently have no defined and validated equivalent of the adapter study's re-anchoring intervention. Their operation folds the adapter into the frozen backbone and restarts it at identity; we have no frozen backbone to fold into. That rules out the specific operation, not every function-preserving reparameterization of a hypernetwork.

What does port is the split between the part carrying the history and the part new to the probe: the generator against the fresh 32-number task code. `uncle/probe.py` implements both arms through `code_only` and `train_code`.

### The adapter study's own result, for context

At the learning rates that maximise accuracy, that study found no plasticity loss in either LoRA or orthogonal finetuning across 20 tasks. Its measurable effect appeared only when the adapter learning rate was pushed past the accuracy optimum, and then only for orthogonal finetuning.

After 20 tasks its mean rotation angle is 3.0 degrees, with a 99th percentile of 11.5 degrees. At that angle the slowdown factor it predicts is 0.99, so its own mechanism is inactive at realistic settings. Its informative arm was the one that pushed the learning rate, not the one at the best settings.

### Decisions

The EuroSAT plan stands. Its deferral is withdrawn.

Add intermediate measurement points when the sequence is next trained, saving the generator after each task.

Correct the `relative_contribution` docstring. Done.

### Open question, not yet decided

Whether plasticity remains the headline of the work, or becomes a supporting measurement while the unlearning results take the main line.

The argument for moving it is weaker than this page first claimed, since the plasticity measurements are sound and only their interpretation needed narrowing. It rests instead on what the two threads can deliver. The adapter study found no loss over a longer and better-powered stream, and our own eight-task result shows improvement rather than decline. The unlearning thread has a published claim that has never been tested, a mechanism the 21 October measurements contradict, and untouched per-task BatchNorm running statistics, which are both a route to recovery and a gap in any privacy claim.

The three steps at the end of the 21 October record belong to that thread, and the first is one extra line in a measurement already written.

Recorded as a recommendation. No decision has been made.
