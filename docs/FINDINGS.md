# Findings

What I set out to answer about UnCLe, what the paper claims, what I measured, and how.

Paper: arXiv 2509.17530, "An Unlearning Framework for Continual Learning", Adhikari, Kumaravelu and Srijith.

Last updated 28 October 2026.

A short note on the method, so the questions below make sense. A hypernetwork produces the weights of a second network instead of being the classifier itself. Each task has a code, a learned vector of 32 numbers. The code goes in, a full set of classifier weights comes out. Forgetting a task means editing that mapping.

---

### 1. What does the forgetting step actually change?

Claimed: the paper presents forgetting as removing a task from the model.

Found: it freezes the task code and retrains the shared generator, so that this one code stops producing a working classifier. The code itself is never touched.

How: read `forget()` in `uncle/trainer.py` in my reimplementation, then checked it against the authors' released `main.py`, which builds its optimiser over the hypernetwork parameters only.

---

### 2. Does the forgetting objective make the weights random?

Claimed: yes. The paper describes forgetting as driving the generated weights toward random noise.

Found: no. The objective is smallest when the weights are exactly zero.

How: worked through the loss. It measures the squared distance between the generated weights and fresh samples of unit Gaussian noise. Those samples average to zero, so the expected distance works out to the squared length of the weights plus a constant that does not depend on them. That is minimised at zero. Confirmed in the paper's equation 3, in my implementation, and in the authors' `toolbox/regularisation.py`.

---

### 3. What does the forgotten model actually output?

Claimed: a uniform spread across all classes, scored as maximum confusion.

Found: one single class, for every image. Exactly 10.00 percent on a balanced ten-class test set, which is what a constant answer gives.

How: recorded the model's predictions directly after forgetting, and counted how many distinct classes it used. One of ten. Seen on two tasks of different starting strength, across several runs. `notebooks/21_forgetting_collapse_check.ipynb`.

---

### 4. Do the paper's success tests show the task was removed?

Claimed: three tests confirm forgetting. Accuracy on the forgotten task at chance, output uniformity at 100, and a membership inference attack at 50.

Found: a model that answers one fixed class for every image passes all three. So the tests show the model went quiet. They do not show the information is gone.

How: measured the forget accuracy directly and observed it at chance. The other two follow from near-zero outputs. This point is not mine alone; Xu et al., arXiv 2505.16831, makes a version of it.

---

### 5. How many forgetting steps are actually needed?

Claimed: 100 steps, annealed down to 20 over repeated operations.

Found: the task is already at chance well before 100. The later steps do not remove more. They make the change harder to undo.

How: two measurements. First, accuracy and input sensitivity traced through a 100-step run, `notebooks/27_weight_change_and_sensitivity.ipynb`. Second, how far the generator's weights turn: 7.3 to 10.5 degrees by step 20, against 49.1 to 54.1 degrees by step 100, over three seeds, `notebooks/30_unlearning_direction.ipynb`.

A direct comparison at 6, 20 and 100 steps, on task 3, seed 0:

| Steps | Retained tasks after forgetting | Forgotten task after learning a later task | Input sensitivity after forgetting |
| ---: | ---: | ---: | ---: |
| 6 | 40.1% | 12.8% | 6.051e+00 |
| 20 | 39.4% | 10.0% | 3.943e-01 |
| 100 | 40.0% | 10.0% | 2.280e-06 |

Retained tasks stood at 40.4 percent before forgetting. At 6 steps the model is still fully reading its input, and the forgotten task comes back above the 12 percent relapse limit. At 100 steps it is dead. `notebooks/23_learn_after_unlearn.ipynb`.

---

### 6. Does forgetting damage the other tasks?

Claimed: the preservation term protects them.

Found: it does. No measurable damage at any step budget.

How: the same run as question 5. Retained tasks were at 40.4 percent before forgetting and 40.1, 39.4 and 40.0 percent after, at the three budgets.

---

### 7. Is the skill gone, or only blocked?

Claimed: removed.

Found: I could not determine this, and I do not think the question is reachable in my model.

How: two attempts. First, relearn the forgotten task from a handful of images and compare against a model that never saw it. The forgotten model was no faster, and on the second seed it was clearly slower. `notebooks/24_recovery_probe.ipynb`. Second, try to read the task through a different, unused code, on the theory that forgetting blocks one route and leaves the skill behind. That test came back void: on an untouched model, five random codes already read 9.5 to 10.0 percent against 10 percent chance, so the instrument was dead before forgetting ever ran. `notebooks/34_route_probe.ipynb`.

Worth noting that the authors' own appendix reports 15.54 percent on Tiny ImageNet for a random replacement code, above chance. My model does not reproduce that, and I do not yet know why.

---

### 8. How much of the method is new?

Claimed: a framework combining continual learning with unlearning.

Found: the architecture and the forgetting family are both cited prior work. What remains as the paper's own piece is the specific weight-space forgetting loss, which is the piece in question 2.

How: read the reference list. The hypernetwork continual-learning architecture, chunked generator and task codes, is von Oswald et al., ICLR 2020. The forgetting side cites Chundawat et al. 2023a and 2023b, and the combination framing cites Chatterjee et al. 2024, by overlapping authors. Chundawat et al. 2023a defines forgetting on outputs rather than weights, which cannot collapse the way question 2 describes, and UnCLe does not use it.

---

## Still open

Whether the task's information survives forgetting. Question 7 was meant to settle it and could not.

Why a random replacement code reads above chance in the authors' model and at chance in mine.

Whether an output-based forgetting objective avoids the collapse inside this architecture. Not yet run.

---

## How to read these numbers

Most results here are one seed unless stated. Where a second seed exists it is noted.

The relapse limit is 12 percent. A forgotten task scoring above it after later learning has come back.

Differences under 3 points are treated as no difference.

Input sensitivity is the widest spread any single output score shows across the test set. It separates a dead model from one that is guessing but still reading its input.

Detailed dated records are in [docs/wiki/Experiment-trajectory.md](wiki/Experiment-trajectory.md).
