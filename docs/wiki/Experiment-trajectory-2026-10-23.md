# Experiment trajectory: 23 October 2026

[Back to the experiment index](Experiment-trajectory)

## Review of the EuroSAT plan, and what it changed

No training was run today. This is a review of the [22 October plan](Experiment-trajectory-2026-10-22) against the adapter study in `reference/adapter-plasticity-study`, and the decisions that followed.

### The reference model was untrained

The 22 October plan compares a generator at t = 0 with a generator at t = 8. Its t = 0 is a generator with random weights, before any task.

That comparison cannot answer a plasticity question. A random generator has never produced working convolution filters, so it will learn EuroSAT poorly from 800 images in 250 steps, and the t = 8 model will beat it for a reason that has nothing to do with plasticity.

The adapter study avoids this because its t = 0 is a frozen ViT-B/16 pretrained on ImageNet-21k with the adapter zeroed. That model is already strong. Ours is not.

`uncle/probe.py` already states this in the note on `relative_contribution`.

### The same fault affects a published finding of ours

The 20 and 21 October record reports that the generator learns an unseen task better after eight tasks, by 4.74 points on average, with the random-label loss falling as well.

Those measurements are correct. The reading placed on them is not safe. Both compare against the same untrained generator, so the expected result was a gain whatever the generator's plasticity. The record also notes that a frozen generator with a fresh task code sits at chance on all three seeds, which is direct evidence that the reference model is weak.

The finding should be restated as: a generator that has learned eight tasks is a better starting point for a new task than a generator that has learned nothing. That is not in doubt and it is not a plasticity result.

### The fix needs the stream trained again

Using t = 1 as the reference would fix it. One task is enough to give the generator a working visual system, so the comparison becomes about the length of the history.

We cannot do this with the models we hold. Notebook 29 records that no mid-stream checkpoints were kept, and `uncle/checkpoint.save` replaces any earlier checkpoint.

So a sound reference point, the mid-stream probe points the 21 October record asked for, and the orders 02 and 03 question are all the same retraining job.

### What does not port from the adapter study

Three parts of that study have no equivalent here, and the record should say so plainly rather than leave it implied.

Re-anchoring. There it means folding the adapter into the frozen backbone and restarting the adapter at identity. The model is unchanged and only its coordinates reset. We have no frozen backbone to fold into, because the generator is the model.

The claim that capacity never shrinks. That holds there because the backbone is frozen and the adapter is a rotation or a low-rank term, so the set of reachable weights is fixed by construction. Our generator is the model and can genuinely lose expressivity. We cannot assume that a slowdown means bad coordinates.

The two chart mechanisms. One is about the Cayley map used by orthogonal finetuning, the other about the two factor matrices in LoRA. We have neither.

What does port is the split between the part that carries the history and the part that is new to the probe. Here that is the generator against the fresh 32-number task code. `uncle/probe.py` already implements both arms through `code_only` and `train_code`.

### The adapter study's own result, for context

At the learning rates that maximise accuracy, that study found no plasticity loss in either LoRA or orthogonal finetuning across 20 tasks. Its measurable effect appeared only when the adapter learning rate was pushed past the accuracy optimum, and then only for orthogonal finetuning.

After 20 tasks its mean rotation angle is 3.0 degrees, with a 99th percentile of 11.5 degrees. At that angle the slowdown factor the study predicts is 0.99, so its own mechanism is inactive at realistic settings.

This matters for us. A null result on an eight-task stream with three seeds would repeat a null that a longer and better-powered study already found. Two null results do not combine into a positive one.

### Missing diagnostic: nothing is logged between probe points

That study logs adapter geometry after every one of its 20 tasks, not only at its six probe points, because those quantities come from the weights alone and cost nothing. That is how it can tie a change in learning to a change in the adapter's shape.

We log nothing between probe points. The 22 October page now lists the quantities worth logging and where they come from. None needs a GPU or a forward pass, and all can run on checkpoints we already hold.

### Decisions

The EuroSAT probe is deferred, not cancelled. Its data splits, validation-based rate selection, pilot, noise floor, and interpretation table are sound and are kept. Its baseline is not, and fixing the baseline requires the retraining job.

The geometry logger is the next piece of work. It needs no GPU and no retraining, it runs on existing checkpoints, and it answers whether the generator moves enough across the stream for any plasticity experiment on this stream to show anything.

### Open question, not yet decided

Whether plasticity remains the headline of the work, or drops to a supporting measurement while the unlearning results become the main line.

The argument for moving it: the plasticity thread currently has two null results behind it, one of them from a longer and better-powered study. The unlearning thread has a published claim that has never been tested, a mechanism that the 21 October measurements contradict, and the untouched per-task BatchNorm running statistics, which are a route to recovery and a gap in any privacy claim.

The three steps listed at the end of the 21 October record belong to that thread, and the first of them is one extra line in a measurement that is already written.

This is recorded as a recommendation. No decision has been made.
