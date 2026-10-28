# Findings

What I set out to answer about UnCLe, what the paper claims, what I measured, and how I reached the conclusion.

Paper: arXiv 2509.17530, "An Unlearning Framework for Continual Learning", Adhikari, Kumaravelu and Srijith.

Last updated 28 October 2026.

A short note on the method, so the questions below make sense. A hypernetwork produces the weights of a second network instead of being the classifier itself. Each task has a code, a learned vector of 32 numbers. The code goes in, a full set of classifier weights comes out. Forgetting a task means editing that mapping.

---

### 1. What does the forgetting step actually change?

Claimed: the paper presents forgetting as removing a task from the model.

Found: it freezes the task code and retrains the shared generator, so that this one code stops producing a working classifier. The code itself is never touched.

How: the code path shows the optimiser is handed only the trunk and the heads, with everything else frozen, so no gradient ever reaches the task code. A run then confirms what that implies. After forgetting I tried to relearn the task twice, once keeping its original code and once with a brand new code. The arm with the original code produced no response to its input for the first 38 steps. The arm with a new code responded immediately, within 2 steps. A second seed gave 50 to 60 steps against 0.

The split follows the code and nothing else. That is what you would see if the generator had been edited so that this one input now maps to dead weights, and it is not what you would see if the code itself had been scrambled, since a scrambled code would behave like any other new code.

---

### 2. Does the forgetting objective make the weights random?

Claimed: yes. The paper describes forgetting as driving the generated weights toward random noise.

Found: no. The objective is smallest when the weights are exactly zero.

How: first the arithmetic. The loss measures the squared distance between the generated weights and fresh samples of unit Gaussian noise. Those samples average to zero, so the expected distance works out to the squared length of the weights plus a constant that does not depend on them. Minimising that drives the weights to zero. Fresh noise is drawn every step, so there is no fixed random target to move toward.

Then the check that distinguishes the two outcomes. A model with random weights still reacts strongly to its input, just wrongly. A model with zero weights barely reacts at all. I measured how much the model's output moves across the test set, and after 100 forgetting steps it was 2.280e-06, which is effectively nothing. Random weights cannot give that number. Zero weights can.

---

### 3. What does the forgotten model actually output?

Claimed: a uniform spread across all classes, scored as maximum confusion.

Found: one single class, for every image. Exactly 10.00 percent on a balanced ten-class test set.

How: I ran the forgotten model over the whole test set and counted how many distinct classes it ever predicted. The answer was 1 out of 10, with 100 percent of predictions in that one class.

The accuracy is the giveaway. A model guessing uniformly would use all ten classes and would land near 10 percent, but not on exactly 10.00, and the figure would wobble between runs. Exactly 10.00 percent with one class used is what you get when the model names the same class every time and happens to be right on that class's share of a balanced set. Seen on two tasks of different starting strength, across several runs.

---

### 4. Do the paper's success tests show the task was removed?

Claimed: three tests confirm forgetting. Accuracy on the forgotten task at chance, output uniformity at 100, and a membership inference attack at 50.

Found: the constant-answer model from question 3 passes all three. So the tests show the model went quiet. They do not show the information is gone.

How: I took the model that question 3 produced and walked each test against it. Accuracy at chance is measured directly, and it is 10.00 percent, so that test passes. Uniformity scores the spread of the output values, and near-zero weights give near-equal values, so that test passes too. The membership attack works by spotting that the model responds differently to data it was trained on, and a model whose output barely moves with its input cannot respond differently to anything, so it passes as well.

All three are satisfied by a model that has stopped answering. None of them separates a model that forgot from a model that went silent. This point is not mine alone; Xu et al., arXiv 2505.16831, makes a version of it.

---

### 5. How many forgetting steps are actually needed?

Claimed: 100 steps, annealed down to 20 over repeated operations.

Found: the task is already at chance well before 100. The later steps do not remove more. They make the change harder to undo.

How: I forgot the same task three times from the same starting checkpoint, stopping at 6, 20 and 100 steps, then learned a later task and measured three things.

| Steps | Retained tasks after forgetting | Forgotten task after learning a later task | Input sensitivity after forgetting |
| ---: | ---: | ---: | ---: |
| 6 | 40.1% | 12.8% | 6.051e+00 |
| 20 | 39.4% | 10.0% | 3.943e-01 |
| 100 | 40.0% | 10.0% | 2.280e-06 |

Retained tasks stood at 40.4 percent before forgetting.

Read the three columns together. The first column barely moves, so the extra steps are not buying safety for the other tasks. The second column does move: at 6 steps the forgotten task climbs back to 12.8 percent once the model learns something else, which is above the 12 percent relapse limit, while at 20 and 100 steps it stays flat at chance. The third column falls by six orders of magnitude, so the model goes from fully reading its input to not reacting at all.

So the extra steps are not removing more of the task. They are driving the model further into silence, where it can no longer drift back.

A separate measurement agrees. The generator's weights turn only 7.3 to 10.5 degrees from where they started by step 20, but 49.1 to 54.1 degrees by step 100, across three seeds. Most of the movement happens after the task has already stopped working.

---

### 6. Does forgetting damage the other tasks?

Claimed: the preservation term protects them.

Found: it does. No measurable damage at any step budget.

How: the same three runs as question 5. The seven retained tasks averaged 40.4 percent before forgetting, and 40.1, 39.4 and 40.0 percent after it, at the three budgets.

Every gap is under one point, and I treat anything under three points as no difference, because repeated runs of the same setting vary by about that much. There is no trend with the step count either, so this is noise and not a small real cost.

---

### 7. Is the skill gone, or only blocked?

Claimed: removed.

Found: I could not determine this, and I do not think the question is reachable in my model.

How: two attempts, and the second is the more useful failure.

The first was a relearning test. If the skill is still in there, feeding the forgotten model a handful of images should bring it back faster than teaching the task to a model that never saw it. I gave both 1, 5 and 20 images per class and compared their best accuracy. On the first seed the forgotten model was ahead by 1.9 points, which is inside the three-point noise band, so it counts as a tie rather than a win. On the second seed it was behind by 2.8, 1.0 and 15.4 points at the three image budgets.

No advantage, then. But this test cannot prove absence either, because both sides train the generator, and training can rebuild a skill from scratch. A real gain would have been hard to see even if the skill were there.

The second was a reading test, designed to avoid that problem by training nothing. The idea: forgetting blocks one code, so give the model a different, unused code and see whether the task is still readable through it. I tried five random codes on the forgotten task.

The result was 10.0 percent, which is chance. But the control is what matters. I ran the same five codes on the model before any forgetting, when the task was sitting at 42.6 percent through its own code, and they read 9.5 percent. A task the model definitely knows reads as nothing through a random code.

So the probe cannot detect a skill that is present. It was dead before forgetting ever ran, which means the reading after forgetting carries no information at all. The test is void, not negative.

Worth noting that the authors' own appendix reports 15.54 percent on Tiny ImageNet for a random replacement code, above chance. My model does not reproduce that, and I do not yet know why.

---

### 8. How much of the method is new?

Claimed: a framework combining continual learning with unlearning.

Found: the architecture and the forgetting family are both cited prior work. What remains as the paper's own piece is the specific weight-space forgetting loss, which is the piece in question 2.

How: I matched each component of the method against the paper's own reference list.

The learning half is the hypernetwork continual-learning setup of von Oswald et al., ICLR 2020: a chunked generator, a small learned code per task, and a penalty that keeps old codes producing the weights they used to produce. The authors' released config uses the same shape, 200 chunks and a 32-number code.

The forgetting half belongs to an existing line of work that defines unlearning as making a model look untrained, cited here as Chundawat et al. 2023a and 2023b. The idea of combining continual learning with unlearning is cited to Chatterjee et al. 2024, by overlapping authors.

Subtracting those leaves the weight-space forgetting loss as the new contribution. That is the part question 2 shows collapses to zero. Chundawat et al. 2023a defines forgetting on the model's outputs rather than its weights, which cannot collapse the same way, and UnCLe does not use it.

---

## Still open

Whether the task's information survives forgetting. Question 7 was meant to settle it and could not.

Why a random replacement code reads above chance in the authors' model and at chance in mine.

Whether an output-based forgetting objective avoids the collapse inside this architecture. Not yet run.

---

## How to read these numbers

Most results here are one seed unless stated. Where a second seed exists it is noted.

The relapse limit is 12 percent. A forgotten task scoring above it after later learning has come back.

Differences under 3 points are treated as no difference, because repeated runs of the same setting vary by about that much.

Input sensitivity is the widest spread any single output score shows across the test set. It separates a dead model from one that is guessing but still reading its input.

Detailed dated records, including the runs that failed or were withdrawn, are in [docs/wiki/Experiment-trajectory.md](wiki/Experiment-trajectory.md).
