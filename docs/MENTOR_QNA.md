# Andrii's Questions - 18/Oct/2026


Questions from mentors, quoted as asked, with our answers. Newest first. Answers cite measurements recorded in the [experiment trajectory](Experiment-trajectory).

## Words used on this page

Every term here is defined once, in one line.

| Term | What it means |
| --- | --- |
| Hypernetwork, or generator | The part that produces the weights of a classifier, instead of being the classifier. |
| Task code | 32 numbers belonging to one task. They tell the generator which classifier to produce. |
| Backbone | The large model that does the main work of reading an image. |
| Frozen | None of that part's weights are allowed to change during training. |
| Adapter | A small set of trainable weights added beside a frozen backbone. The backbone stays fixed and the adapter changes what it computes. |
| Hypernetwork-generated adapters | The generator produces the adapter weights from a task code, instead of the adapter being trained directly. |
| Matas's setup | Plain continual learning: train the model on one task after another, with no hypernetwork. |
| Plasticity | The ability to learn a **new** task. Not the same as forgetting, which is old tasks getting worse. |
| Fork-and-discard probe | Copy the model, teach the copy a task it has never seen, measure, then throw the copy away. |
| Input sensitivity | The widest spread any single output score shows across the test set. A dead model reads near zero. |
| Chance | The score a model gets by guessing. With 10 classes it is 10 percent. |

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

**On the two options, we have no data.** We have not run either one. What we can say is only this:

- Matas's setup has no task code. So the unlearning method as written cannot be applied to it.
- Hypernetwork-generated adapters keep a task code, so the method could in principle be applied. Whether unlearning still drives a task to chance there is untested. An adapter is a much smaller target than a full ResNet50.

Testing the second option on one task is a short run. That would tell us more than any argument here.

---

## Q2. Why does the forgotten task stop responding?

> Explain why the forgotten task stops responding, with activation visualizations
>
> Andrii wants a detailed explanation of what's happening, including which neurons activate and which don't. This tells us whether it's real plasticity loss, temporary or not. Due before the next meeting.

**Answer.** The generated network is turned down everywhere, and over half its channels stop firing. Only the forgotten task is affected. Measured 21 October, seed 0, task 3 removed.

**What the method does.** The unlearning objective compares the generated weights against fresh random numbers. It draws new ones at every step. Those draws are centred on zero. The best single answer to many random demands is their average, which is near zero. So the weights collapse toward zero instead of becoming random.

**What we measured.**

| Measurement | Result |
| --- | --- |
| Task 3 accuracy after unlearning | 10.0%, chance for 10 classes |
| Relapse over the next lesson | 0.0 points, stays at 10.0% |
| Input sensitivity | Falls by about 7 factors of ten |
| Relative weight change during unlearning | 0.629 |
| The other seven tasks | 40.4% to 40.0% mean, none near chance |

Input sensitivity is the widest spread any single output score shows across the test set. A working model reads 10.8 to 13.9. After unlearning it reads far below that. The network is switched off, not randomised.

### Which neurons still fire

We measured every neuron twice. Once before we asked the model to forget task 3, and once after. Both times we showed it the same 256 pictures.

We did the same for task 0, which we kept. That is our **control**: something we leave alone, so we can see whether our change hit only what we aimed at.

**How strongly the neurons still react.** A ResNet reads a picture through five parts, one after another. The stem comes first, then layer1, layer2, layer3, layer4.

| Part of the network | Forgotten task 3 | Kept task 0 |
| --- | ---: | ---: |
| stem | 29% | 100% |
| layer1 | 14% | 99% |
| layer2 | 13% | 98% |
| layer3 | 16% | 97% |
| layer4 | 19% | 97% |

Read each number as the share of the reaction that is left. 14% means the neurons there still react, but only about one seventh as strongly as before. 100% means nothing changed.

**How many neurons stopped reacting.** We call a neuron **silent** when its strongest reaction drops below one hundredth of what it used to be.

| | Silent neurons, out of 22,720 |
| --- | ---: |
| Forgotten task 3 | 58.9% |
| Kept task 0 | 0.0% |

**What these two tables tell us.**

- **We hit only what we aimed at.** The kept task lost nothing. Not one of its neurons went silent, and every part of its network reacts within 3 percent of before.
- **The forgotten task does not break in one place.** Every part got weaker by a similar amount. If one layer were blocking the signal, that layer would read near zero and the parts before it would read normal. They do not.
- **It is weakened, not erased.** About 41 percent of neurons still react, at roughly one sixth of their old strength. This is what we expect: our forgetting method pushes the task's weights toward zero, and small weights give small reactions.

**One thing we noticed and cannot yet explain.** The stem lost less than the rest, 29 percent against 13 to 19 percent. There is a possible reason in our code. The forgetting step and the protecting step compare numbers measured on different scales, and those scales differ from one part of the network to the next. So the forgetting may press harder on some parts. We have not tested this, so treat it as a guess with a reason behind it, not a result.

**What this does not tell us.** Whether task 3 can be brought back. These measurements describe the network as it is now. Recovery is a separate experiment that we have not yet designed.
---
