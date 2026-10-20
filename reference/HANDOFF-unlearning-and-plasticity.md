# Unlearning work: handoff into the adapter plasticity study

Written 14 October 2026. Everything below is either measured or quoted. Where a number
rests on one seed, it says so.

This document exists so the adapter plasticity study can pick up a line of work on
machine unlearning without rerunning any of it. It covers what was measured, what the
numbers mean, which ideas are already dead, where the code and data live, and the one
experiment that connects the two projects.

---

## 1. Why this is in front of you

The adapter plasticity study asks whether LoRA and OFT keep their ability to learn new
tasks. Its own notes record a soft spot:

> "At the selected rates the adapters barely move (OFT's mean rotation after 20 tasks is
> 3°), so these runs cannot test H1."

To make anything happen, the learning rate has to be pushed to 3e-3, where LoRA diverges
in some seeds. A reviewer can fairly say the effect only appears in a regime nobody would
deploy. The study needs a **principled** reason for a model to move a long way.

Separately, a published continual-unlearning method claims in writing that unlearning
**restores plasticity**:

> "UnCLe's unlearning objective restores the learned task-specific classifier parameters
> to a randomly initialized state, akin to a Kaiming He initialization. This restores the
> hypernetwork's plasticity, allowing it to learn new tasks again."
> (UnCLe, page 7)

Nobody has tested that with a plasticity probe. Their evidence is final retained accuracy
(their Table 3), which mixes plasticity with forgetting and transfer. The adapter study
exists precisely to separate those.

The work summarised here shows the stated mechanism is false: the unlearned task is not
returned to a random initialisation, it is switched off. So if plasticity is restored, it
is not for the reason given.

**That is the bridge.** Section 9 sets out the experiment.

---

## 2. The method under study, in brief

**UnCLe** performs data-free unlearning in a continual setting. Three pieces matter:

| piece | what it is | changed by unlearning? |
| --- | --- | --- |
| **task code** | 32 numbers naming one task | **no**, frozen during unlearning |
| **generator** (hypernetwork) | shared network mapping a code to classifier weights | yes |
| **generated weights** | ~23.5M weights the generator outputs for that code | yes, they shrink |

Learning trains the generator and that task's code. Unlearning trains **only** the
generator, pushing the generated weights for one code toward Gaussian noise, while a
protection term holds the other tasks' weights near where they were.

Verified directly ([§11](#11-reproducing-the-core-claims) has the script): after a
forgetting request the task code changes by exactly 0.0, the generator moves, and the
generated weights shrink.

---

## 3. What was measured, and how firmly

### 3.1 The unlearned task gives one constant answer: SOLID, six runs

Source: notebook 21, R2 `diagnostics/20261012T174517Z`, all six completed comparisons,
run through the study's own code path.

| seed | forget | before | after | distinct classes | class chosen | weights left | input sensitivity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 35.6% | 10.0% | 1 | 0 | 57.2% | 2.28e-06 |
| 0 | 14 | 42.6% | 10.0% | 1 | 6 | 59.3% | 2.68e-06 |
| 1 | 3 | 40.4% | 10.0% | 1 | 0 | 48.2% | 8.94e-08 |
| 1 | 14 | 40.4% | 10.0% | 1 | 1 | 56.0% | 2.98e-07 |
| 2 | 3 | 38.4% | 10.0% | 1 | 0 | 45.7% | 7.45e-08 |
| 2 | 14 | 41.8% | 10.0% | 1 | 7 | 49.4% | 1.49e-07 |

Every run ends giving **the same predicted class to all 500 test images**. The validation
set holds 50 images of each of 10 classes, so one fixed answer is correct 50 times out of
500: exactly 10.0%, with no variance left. A genuinely guessing model would scatter
roughly between 7.3% and 12.7% (binomial, n=500, p=0.1, sd 1.34 points).

The chosen class varies across runs (0, 1, 6, 7), which rules out an argmax tie-breaking
artefact.

**Input sensitivity** is the measurement that separates the two states. Definition: for
each of the ten output scores, the range it spans across the whole test set; report the
largest of the ten. A working model scored 13.2. After unlearning, about 1e-6 to 1e-7.

### 3.2 A never-trained network is not in that state: SOLID in direction, noisy in size

Measured three times on freshly built networks: **38.2, 95.0, 309.4**. It varies because
each is a different random network and logits have no fixed scale. Use it as an order of
magnitude, never as a target. Against 1e-6 the gap is roughly a hundred-million-fold
either way.

This is what contradicts "akin to a Kaiming He initialization". An untrained network still
reacts to its input. This one does not.

### 3.3 Unlearning is specific; the other tasks survive: SOLID, six runs

Same model, measured immediately before and after the forgetting request:

| run | mean before | mean after | change | worst single task |
| --- | ---: | ---: | ---: | --- |
| seed 0, forget 3 | 40.4% | 40.0% | -0.4 | task 9: -2.4 |
| seed 0, forget 14 | 39.4% | 39.1% | -0.3 | task 5: -1.4 |
| seed 1, forget 3 | 38.8% | 40.0% | +1.2 | task 14: 0.0 |
| seed 1, forget 14 | 38.8% | 39.1% | +0.3 | task 5: -1.6 |
| seed 2, forget 3 | 43.6% | 43.0% | -0.7 | task 5: -3.0 |
| seed 2, forget 14 | 43.1% | 42.1% | -1.0 | task 5: -3.4 |

The method does what it claims on this axis. Do not write otherwise.

### 3.4 Recovery: PROVISIONAL, one seed each

Two protocols, two different answers. This matters, because the authors used the first.

**Generator frozen** (their protocol: freeze the hypernetwork, train only a fresh 32-number
embedding). R2 `diagnostics/20261014T162221Z`, 300 steps:

| images total | old code + old stats | old code only | fresh code + old stats | fresh code |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 10.4% | 10.4% | 18.8% | 12.4% |
| 50 | 10.2% | 10.2% | 16.0% | 21.0% |
| 100 | 10.0% | 10.0% | 14.2% | 15.2% |
| 200 | 10.0% | 10.0% | 14.6% | 11.0% |

Nothing approaches the original 35.6%. **On the authors' own protocol their
irrecoverability claim holds.** Two further readings:

- The **old task code is dead**. It sits at 10.0 to 10.4% for 300 steps at every budget. A
  fresh code does better. Unlearning does not merely remove the task, it makes that
  identifier unusable. Clean across four budgets, no exceptions.
- The old BatchNorm statistics are worth a little: 14 to 19% against 10%.

**Generator trainable** (a looser threat model). R2 `diagnostics/20261014T020144Z`:
task 3 reaches 33.6% with 200 images. But tasks 2, 4 and 6, which the model has **never
seen**, reach 37.2% on average under the same budget. So the recovery reflects how easily
the generator learns anything, **not** leftover knowledge of task 3.

**Conclusion: there is no privacy result here.** Do not revive it without new evidence.

### 3.5 Unlearning does not block later learning: SOLID enough to build on

The same unseen-task arms show the post-unlearning model learning tasks it has never seen
to 31.8 to 41.8% with 200 images. Task 3 originally scored 35.6%. **This is the preliminary
plasticity signal, and it is the reason this document exists.**

---

## 4. Ideas that are dead: do not spend time on these

| idea | why it is dead |
| --- | --- |
| "The objective drives the weights to zero" | In the paper's own Theorem (their Appendix), and restated in a reviewer's summary. Not a finding. |
| "Use one fixed noise target instead of fresh draws" | The paper already tested it (Alternative Noising Strategies) and reports it "hampers the retain-task accuracy". Our run diverged: weights grew from 4706 to 79,528 and the arithmetic overflowed. |
| "The output need not be uniform" | A reviewer already asked exactly this. |
| "Unlearning destroys all tasks" | False. See §3.3. This came from measuring the wrong code path. |
| "The task's code is damaged, so recovery is slower" | At 100 steps the gap was 9 to 13 points; by 1000 steps it was 4.4, inside the noise. A delay, not damage. |
| Any privacy or leakage claim | §3.4. The unseen-task control is negative. |

---

## 5. Two traps that cost real time

**Trap 1: there are two forgetting implementations and they are not the same.**

- `uncle/trainer.py:276` `forget()` aims at the generator's **raw output**, before scaling.
- `uncle/study_training.py:80` `run_request()` aims at the **generated weights**, after
  scaling, via `uncle/paired_diagnostic.py:43` `generated_noise_loss()`.

**The study runs the second.** Measuring the first produced a completely different and
much harsher picture (all tasks collapsing to chance), which contradicted the study's own
saved evidence. Always go through `run_request`. Do not reimplement the loop.

**Trap 2: measure in evaluation mode.**

In training mode BatchNorm normalises using each batch's own statistics, which rescales a
nearly dead signal back to normal size and hides the collapse entirely. Measured on a
ResNet-18: the same collapsed model reads 0.918 in training mode and 0.034 in evaluation
mode, 27 times healthier. On the study's ResNet-50 the distortion was about a million
times. Always `hypernet.eval(); target.eval()` before measuring, and restore the mode
after.

---

## 6. Where the code is

**Repository:** `C:\LangChain\Algoverse\continual-learning\hypernetwork`
**Remote:** https://github.com/sumitasthana/CARK

**Pinned experiment revision:** `65e2e7c850fd6f1852a99d9e81780c27f799c926`
("Keep paired study reports and source models by default", 8 October 2026)

Every notebook from 18 to 24 detaches the checkout to this commit and verifies that
`uncle/` matches it exactly. **Nothing may be added to `uncle/`** or all of them refuse to
run. New measurement code goes in notebooks or `scripts/`.

### Files that matter

| path | why |
| --- | --- |
| `uncle/study_training.py:80` | `run_request`, the study's learn and forget path. Use this. |
| `uncle/paired_diagnostic.py:43` | `generated_noise_loss`, the real unlearning objective |
| `uncle/trainer.py:276` | `forget`, a **different** objective; see trap 1 |
| `uncle/trainer.py:72` | `preserve`, the protection term, divided by the number of protected tasks |
| `uncle/hypernet.py` | `task_codes`, `weights_from_code`, per-layer `scales` and `offsets` |
| `uncle/paired_study.py:110` | `jobs()`, job identifiers; `StudyStore.load` fetches a checkpoint |
| `scripts/dashboard.py` | progress panel; patches the repo's two bar factories. `python -m` style self-check inside |
| `docs/reports/evidence/paired-study-2026-10-10.json` | the study's own saved results and pairing checks |
| `ops-docs/why-ten-percent/` | long-form write-up plus three CPU scripts. **Gitignored, local only** |

### Notebooks

| notebook | what it does |
| --- | --- |
| `21_forgetting_collapse_check.ipynb` | measures all six completed runs: accuracy, per-class prediction counts, input sensitivity, weight size, and the seven protected tasks. **The one that carries the solid result.** |
| `22_fixed_noise_forgetting.ipynb` | fixed-target comparison. Dead end, kept for the record |
| `23_learn_after_unlearn.ipynb` | forgetting, then learning a new task, measured at three stages |
| `24_recovery_probe.ipynb` | recovery arms, the unseen-task control, and `FREEZE_GENERATOR` |

All four open in Colab from `notebooks/README.md`, need an A100 and three R2 secrets
(`R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`).

### Commits from this work

`d8567fb` through `426661d`, 19 commits. `git log --oneline 9d3be65..426661d`.

---

## 7. Data and artefacts

**Source models are kept in R2 and never need retraining.** Branch models are deleted
after verification; sources are not. A source is 8 learned tasks, about 4.8 hours on A100,
so reuse them.

- Bucket `uncle-experiments`, prefix `uncle/paired_generalization/paired_generalization_v2`
- Source job ids: `sources/order_01/seed_{0,1,2}`
- Diagnostics written by this work: `.../diagnostics/<UTC timestamp>/`

| timestamp | what |
| --- | --- |
| `20261012T174517Z` | notebook 21, six runs, correct code path. **The headline data** |
| `20261013T211143Z` | 1000-step fixed-target comparison |
| `20261014T020144Z` | recovery, generator trainable, unseen control |
| `20261014T162221Z` | recovery, generator frozen (the authors' protocol) |

Credentials live in the repo's local `.env`, which is gitignored.

---

## 8. The study's configuration

| setting | value |
| --- | --- |
| dataset | Tiny ImageNet, 10 classes per task, partition seed 42 |
| per task | 5,000 train / 500 validation images |
| backbone | ResNet-50, 3×3 stride-1 stem, 200 generated chunks |
| generator | input 64, hidden 128/256/512, three output heads, hyperfan-in init |
| codes | 32 numbers for task and chunk |
| optimiser | Adam, lr 1e-4 for both learning and forgetting |
| learning | 5 epochs, batch 64 |
| forgetting | 100 steps, 10 fresh noise samples per step |
| loss weights | beta 0.1 (learning protection), gamma 0.01 (noise term) |
| order 01 | tasks 3, 0, 9, 5, 17, 1, 7, 14, then learn 15 |
| seeds | 0, 1, 2 |
| forget tolerance | 2.0 points above chance |

**Important caveat for any write-up:** beta is 0.1 here, against the paper's 0.01 for Tiny
ImageNet. This is **not** the paper's configuration and nothing here reproduces their
reported numbers. Only one learning order of the three planned was completed, so task
identity and sequence position are perfectly confounded.

---

## 9. The experiment that connects the two projects

**Question.** Does unlearning a task restore a model's ability to learn new ones, as UnCLe
claims, and if so by what mechanism?

**Why it is open.** Their evidence is final retained accuracy, which mixes plasticity with
forgetting and transfer. No probe has been run. And the mechanism they give for it,
returning the task to a random initialisation, is measurably false (§3.1, §3.2).

**Why it is affordable.** The unseen-task arms in notebook 24 are already a fork-and-discard
probe in miniature, and the source models are saved. The pipeline exists.

### Design, following the adapter study's protocol

- **Probe points:** before unlearning, immediately after, and after the next task is learned.
- **Probe tasks:** held out from the stream. Order 01 uses 0, 1, 3, 5, 7, 9, 14, 17 and then
  15, so 2, 4, 6 and others are untouched. Notebook 24 already loads 2, 4 and 6.
- **Fork and discard:** snapshot, train a copy, measure, restore. Notebook 24 does this.
- **Head-only control:** train only the classifier head on frozen generated features, to
  separate the generator's contribution from feature quality.
- **Normalisation:** report each point against its own t=0 measurement, per seed.
- **Noise floor and decision rule:** reuse the adapter study's, including the paired
  bootstrap over seeds.
- **Seeds:** 0, 1, 2. All three source models exist.

### What each outcome gives you

| outcome | what it means |
| --- | --- |
| Plasticity rises after unlearning | Their claim holds, their stated mechanism does not. Supply the correct one. |
| Plasticity does not rise | Their Table 3 effect comes from reduced interference, not restored capacity. An explicit claim is wrong. |
| It rises only for the unlearned task's own slot | Connects directly to the dead-code result in §3.4. The most interesting of the three. |

### What it does for the adapter study

It supplies a second, non-artificial way to make a model move a long way, instead of
inflating the learning rate to 3e-3. And it ties a null result on LoRA and OFT to a
positive published claim: *capacity is not lost at sensible rates, and the one method
claiming to restore capacity does not do so for the reason it states.*

### Decide this first

Check with whoever owns the adapter study whether hypernetworks are in scope, or whether
it is strictly adapters on a frozen backbone. That decides whether this is a contribution
or a distraction. Do not start before that answer.

---

## 10. Status of the source paper

**UnCLe: An Unlearning Framework for Continual Learning.** Adhikari, Kumaravelu, Srijith
(IIT Hyderabad / Deakin). arXiv [2509.17530](https://arxiv.org/abs/2509.17530).
OpenReview [pFjzF7dIgg](https://openreview.net/forum?id=pFjzF7dIgg), submission 13520.

**Rejected from ICLR 2025.** Four reviews: 5, 5, 8, 5. Soundness and presentation rated
good; contribution fair. The meta review rejects it for significance, not correctness:

> "It is incremental rather than opening a new direction ... this paper lacks of insights
> behind empirical observations."

Reviewers kept their scores after rebuttal.

**This matters for framing.** A critique whose subject is a rejected submission is not a
contribution. Anything built from this must lead with the general point about how
unlearning is evaluated, with this method as one worked example.

### What the reviewers did and did not say

Checked across all four reviews and every author response.

**Raised already** (so not novel):
- One reviewer restated that the objective "reduces the generated parameter values to zero"
  and treated it as intended behaviour.
- One asked why the output must be uniform.
- One proposed "discarding the embedding or setting θ_f to zero" as trivial baselines the
  method should beat.
- One asked what recovering the task embedding implies for recovering task performance.
  **The authors never answered it for their own method.**

**Never raised, by anyone** (searched the whole page, zero hits): that the unlearned model
gives a constant output, ignores its input, or anything resembling input sensitivity.

### The authors' own claims, from the rebuttal

> "**Irrecoverability: Unlearning with UnCLe is irrecoverable.**"

Listed as one of four reasons the work matters. The recovery experiment in the paper is for
a naive baseline, and they state: *"this figure is not reflective of our method UnCLe."* The
equivalent test on UnCLe does not appear anywhere. §3.4 is that test. Under their protocol
the claim survives.

Their recovery protocol, for reference: *"The hypernetwork is frozen and a new task
embedding is randomly initialised ... and then optimized with respect to the data subset."*
40 epochs, with 10, 20, 50 and 100 images.

---

## 11. Reproducing the core claims

Three scripts in `ops-docs/why-ten-percent/` run on CPU in seconds and need no GPU, no R2
and no checkpoints:

| script | shows |
| --- | --- |
| `variation_check.py` | what chance-level accuracy should look like: sd 1.34 points, 7.3 to 12.7% usual range, 5.9% chance of landing on exactly 10.0% |
| `resnet_zero_check.py` | the study's ResNet-50 with generated weights at zero: identical logits for every image, exactly 10.0% |
| `collapse_check.py` | the repository's own forgetting loop, with the prediction histogram collapsing to one class |

To rerun the real measurements: open notebook 21 in Colab on an A100, add the three R2
secrets, Run all. About 30 minutes for all six comparisons. Nothing needs retraining.

---

## 12. If you write this up

**Supported by six runs:**
- The unlearned task does not become a randomly initialised network. It stops responding to
  its input entirely.
- No metric in the paper separates those states. Retain accuracy, forget accuracy, Spill,
  Relapse, Uniformity and membership inference all pass a switched-off model. Their
  uniformity criterion in fact **rewards** it, since a constant output is perfectly uniform
  by construction.
- Unlearning is specific: the retained tasks survive it.

**Supported by one seed, say so explicitly:**
- The unlearned task's own code cannot be retrained afterwards.
- Under the authors' frozen protocol the task does not recover.

**Not supported, do not claim:**
- That the forgotten information survives or leaks. The unseen-task control is negative.
- That the authors' reported numbers are wrong. Different configuration, one order, three
  seeds.

**One sentence that holds up:**

> A continual unlearning method reports a task as forgotten once its accuracy reaches
> chance. We show it reaches chance by making the model stop responding to its input, not
> by returning it to an untrained state as claimed, that standard measures including output
> uniformity cannot distinguish these and in fact reward the former, and that the
> unlearned task's identifier cannot be retrained afterwards.

---

## 13. Papers referred to

**The method under study**
- Adhikari, Kumaravelu, Srijith 2025. *UnCLe: An Unlearning Framework for Continual
  Learning.* [arXiv:2509.17530](https://arxiv.org/abs/2509.17530).
  [OpenReview](https://openreview.net/forum?id=pFjzF7dIgg). Rejected, ICLR 2025.
  Local copy: `2509.17530-adhikari-uncle-unlearning-cl.pdf` in the CARK repository root.

**Cited by it, and relevant here**
- von Oswald et al. 2020. *Continual Learning with Hypernetworks.*
  [arXiv:1906.00695](https://arxiv.org/abs/1906.00695). The learning and protection scheme.
- Chang, Flokas, Lipson. *Principled Weight Initialization for Hypernetworks.*
  Cited in the code as [arXiv:2312.08399](https://arxiv.org/abs/2312.08399) at
  `uncle/hypernet.py:136`, whose Table 1 gives the hyperfan-in scheme the study uses. This
  is the source of the Kaiming comparison the UnCLe paper leans on, so check the exact
  identifier before citing it yourself.
- Shokri et al. 2017. *Membership Inference Attacks Against Machine Learning Models.*
  [arXiv:1610.05820](https://arxiv.org/abs/1610.05820). Their privacy metric.
- Kurmanji et al. 2023. *Towards Unbounded Machine Unlearning* (SCRUB).
  [arXiv:2302.09880](https://arxiv.org/abs/2302.09880). One of their baselines.
- Fan et al. 2024. *SalUn.* [arXiv:2310.12508](https://arxiv.org/abs/2310.12508). Baseline.
- Foster et al. 2024. *Fast Machine Unlearning Without Retraining Through Selective Synaptic
  Dampening* (SSD). [arXiv:2308.07707](https://arxiv.org/abs/2308.07707). Baseline.

**Shared with the adapter plasticity study**
- Dohare et al. 2024. *Loss of Plasticity in Deep Continual Learning.* Nature.
- Lyle et al. 2023. *Understanding Plasticity in Neural Networks.*
  [arXiv:2303.01486](https://arxiv.org/abs/2303.01486). Random-label trainability probe.
- Kumar et al. 2021. *Implicit Under-Parameterization.*
  [arXiv:2010.14498](https://arxiv.org/abs/2010.14498). Effective rank as a diagnostic, the
  closest existing relative of input sensitivity.
- Hernandez-Garcia, Figliolia, Millidge 2026. *Can Scale Save Us From Plasticity Loss in
  Large Language Models?* [arXiv:2606.24752](https://arxiv.org/abs/2606.24752).
  Fork-and-discard probe. §9's design follows this.
- Alain and Bengio 2016. *Understanding Intermediate Layers Using Linear Classifier Probes.*
  [arXiv:1610.01644](https://arxiv.org/abs/1610.01644). The head-only control.

Everything in the adapter study's own reference list still applies and is not repeated here.

---

## 14. First week, in order

1. Settle the scope question in §9 before anything else.
2. Read §5. Both traps cost days.
3. Run `ops-docs/why-ten-percent/*.py` on CPU, five minutes, to see the core claim yourself.
4. Run notebook 21 once on A100 to confirm the pipeline works end to end for you.
5. Then build the plasticity probe from §9 on top of notebook 24, which already has the
   unseen-task arms, the fork-and-discard structure and the measurement code.
