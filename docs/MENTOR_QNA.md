# Mentor QnA

Questions from mentors, quoted as asked, with our answers. Newest first. Answers cite measurements recorded in the [experiment trajectory](Experiment-trajectory).

---

## Q1. Is the hypernetwork needed?

> Get back to Andrii on whether the hypernetwork is needed to cause forgetting and learning
>
> Andrii suggested using plain continual learning like Matas's setup, or hypernetwork-generated adapters on a frozen model. That would let the two papers merge.

**Answer.** It is needed for the unlearning. It is the weak part of the learning.

The unlearning method only works because a hypernetwork exists. It removes a task by corrupting the mapping from that task's code to its weights. That is why it needs no training data. It needs only the code and a noise generator. A plain network has no code to corrupt.

So Matas's setup removes the mechanism the paper is about.

But the hypernetwork is also where our accuracy goes.

| Model | Learning rate | Accuracy on one task |
| --- | ---: | ---: |
| Plain ResNet50 | 0.001 | 53.8% |
| Hypernetwork | 0.001 | 14.4% |
| Hypernetwork | 0.0001 | 32.0% |

The remaining gap is unexplained. Two recent measurements narrow down where it is not.

**The generator carries everything. The task code carries nothing.**

| Arm | What trains | Result over 3 seeds |
| --- | --- | --- |
| Code only | The 32-number task code | 10.0 to 10.9%, chance is 10% |
| Generator only | The generator, code left random | No measurable difference from full |
| Full | Both | 51.9, 53.1, 52.5% |

**The generator is not getting stiff.** The fork-and-discard probe ran on three seeds. After eight tasks it learns an unseen task 4.74 points better than a generator that has learned nothing. The random-label measurement agrees, and its three seeds differ from each other by only 0.06.

So this is not a plasticity problem. It is a weaker learner from the start.

**Which option to take.**

| Option | Keeps the unlearning mechanism | Fixes learning quality | Merges the two papers |
| --- | --- | --- | --- |
| Plain continual learning, Matas's setup | No | Yes | Yes |
| Hypernetwork-generated adapters, frozen model | Yes | Yes | Yes |
| Current setup | Yes | No | No |

The second option is the one to take.

**Open before we commit.** We have not tested whether unlearning still drives a task to chance when only adapters are generated and the backbone is frozen. An adapter is a much smaller target than a full ResNet50.

---

## Q2. Why does the forgotten task stop responding?

> Explain why the forgotten task stops responding, with activation visualizations
>
> Andrii wants a detailed explanation of what's happening, including which neurons activate and which don't. This tells us whether it's real plasticity loss, temporary or not. Due before the next meeting.

**Answer.** The generated network stops responding to its input. We know why. The neuron-level picture is not measured yet.

**What the method does.** The unlearning objective compares the generated weights against fresh random numbers. It draws new ones at every step. Those draws are centred on zero. The best single answer to many random demands is their average, which is near zero. So the weights collapse toward zero instead of becoming random.

The paper's wording and the paper's formula disagree here. The wording says randomise. The formula, with averaging, says go to zero. Both stop the task working, so accuracy looks the same either way.

**What we measured.**

| Measurement | Result |
| --- | --- |
| Task 3 accuracy after unlearning | 10.0%, chance for 10 classes |
| Relapse over the next lesson | 0.0 points, stays at 10.0% |
| Input sensitivity | Falls by about 7 factors of ten |
| Relative weight change during unlearning | 0.629 |
| The other seven tasks | 40.4% to 40.0% mean, none near chance |

Input sensitivity is the widest spread any single output score shows across the test set. A working model reads 10.8 to 13.9. After unlearning it reads far below that. The network is switched off, not randomised.

**What is missing.** Which units still fire, and which have gone silent. That is new analysis. It is cheap. Notebook 27 rebuilds a post-unlearning model in under a minute, because no post-unlearning checkpoint is saved.

**Two corrections to the question.**

1. This is not plasticity loss. Plasticity means the ability to learn a **new** task. We measured that on three seeds. It is unimpaired, and slightly improved. A forgotten task going silent is a different thing.

2. Activation pictures cannot say whether it is temporary. That is the recovery question: can the task be brought back under a fixed training budget. It is the largest open item in the study and it is still not specified. It needs its own experiment.

---

## Related

- [Experiment trajectory](Experiment-trajectory), the index of all measurements quoted here.
- [20 and 21 October record](Experiment-trajectory-2026-10-21), the plasticity probe results.
- [Concepts and processes](Concepts-and-processes), definitions of the terms used above.
