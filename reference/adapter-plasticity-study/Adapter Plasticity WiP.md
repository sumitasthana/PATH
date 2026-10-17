# **Adapter Plasticity Study: Methodology**

The study tests whether LoRA and OFT adapters on a frozen ViT-B/16 keep their ability to learn new tasks over a 20-task class-incremental stream, measured with a fork-and-discard plasticity probe. So far, at the learning rates that maximise accuracy, neither adapter loses plasticity. When the adapter is pushed to larger updates, OFT's ability to learn each new task declines in proportion to the learning rate, while LoRA's holds until training diverges.

## **Question and hypotheses**

**Question.** When a parameter-efficient adapter is trained on one task after another, does it keep its ability to learn the next task (plasticity), and does the answer differ between a low-rank update (LoRA) and an orthogonal update (OFT)?

Loss of plasticity is the gradual decline in a network's ability to learn new tasks under continued training. It is well documented for fully trained networks ([Dohare et al. 2024](https://www.nature.com/articles/s41586-024-07711-7); [Lyle et al. 2023](https://arxiv.org/abs/2303.01486)) and much less studied for adapters on a frozen pretrained backbone. It is distinct from forgetting, the loss of performance on earlier tasks, and has to be measured separately.

**Hypotheses.**

* **H1 (chart saturation, OFT).** OFT parametrises each rotation through the Cayley map. As the learned rotation angle grows, the map's sensitivity to its parameters shrinks, so equal gradient steps rotate less. OFT should lose plasticity as its angles grow, and the loss should track the angle.  
* **H2 (conditioning, LoRA).** LoRA's update BA can become ill-conditioned as B grows along a few directions. LoRA should lose plasticity if, and only if, the condition number of BᵀB grows.  
* **H0.** Neither adapter loses plasticity over 20 tasks on a frozen backbone, because the backbone's features are untouched and the adapters stay near their initialisation.

**What separates them.** H1 predicts a decline for OFT that grows with learning rate and with rotation angle, absent in LoRA. H2 predicts a LoRA decline tied to conditioning. H0 predicts flat probe scores for both at every learning rate where training is stable.

## **Model and adapters**

Every run uses the same frozen ViT-B/16; only the adapter and a linear head are trained. The two adapters sit in the same places and have matched parameter budgets (within 2%).

| Component | Setting |
| ----- | ----- |
| Backbone | ViT-B/16 ([Dosovitskiy et al. 2021](https://arxiv.org/abs/2010.11929)), timm checkpoint `vit_base_patch16_224.augreg_in21k` (ImageNet-21k, AugReg). All backbone weights frozen. Output: pooled 768-d pre-logit feature |
| Input | 224 × 224, normalised with mean \= std \= 0.5 (the checkpoint's statistics) |
| Adapter placement | `attn.qkv` and `attn.proj` in all 12 blocks (regex \`.\*blocks.\\d+.attn.(qkv |
| LoRA ([Hu et al. 2022](https://arxiv.org/abs/2106.09685)) | Rank 8, alpha 8, no dropout. W' \= W \+ (alpha/r) BA; B starts at zero, A random (PEFT default). 442,368 trainable parameters |
| OFT ([Qiu et al. 2023](https://arxiv.org/abs/2306.07280)) | Block-diagonal rotation with block size 48, exact Cayley map (`use_cayley_neumann=False`), no dropout. W' \= RW with R \= (I − Q)(I \+ Q)⁻¹, Q skew-symmetric per block, starting at Q \= 0 (R \= I). 433,152 trainable parameters |
| Head | One linear layer 768 → 100 shared by all tasks, trained at learning rate 1e-3 in every run |
| Implementation | Hugging Face PEFT 0.21.2 (`LoraConfig`, `OFTConfig`), timm 1.0.30 |

Because B \= 0 and Q \= 0 at the start, both adapted models are exactly the pretrained ViT before training. LoRA's A differs between seeds; OFT's starting point is the same for every seed.

## **Continual stream**

The stream is Split CIFAR-100 in its standard class-incremental form: 20 tasks of 5 classes, one shared head, and no task label at test time. Registered in Mammoth as `seq-cifar100-224-20t`.

| Item | Setting |
| ----- | ----- |
| Source | CIFAR-100 ([Krizhevsky 2009](https://www.cs.toronto.edu/~kriz/cifar.html)): 100 classes, 500 training and 100 test images per class, 32 × 32 colour |
| Class order | One fixed order for every run and seed: `numpy.random.default_rng(1).permutation(100)`. Task k holds the classes at positions 5k to 5k+4 |
| Validation split | 10% of each class's training images (50 per class), held out by Mammoth's `--validation 10`, stratified by class, drawn with the run's seed. Used only to choose learning rates |
| Training images | 450 per class, 2,250 per task, 45,000 in total |
| Test images | Official CIFAR-100 test split, 100 per class. All reported accuracies use it |
| Training augmentation | RandomCrop(32, padding 4\) and horizontal flip at native size, then bicubic resize to 224 and normalisation. Replayed images get the same augmentation when drawn from the buffer |
| Test preprocessing | Bicubic resize to 224 and normalisation, no augmentation |
| Loss | Cross-entropy over all 100 head outputs |
| Evaluation after each task | Every task seen so far, on its test images. Class-IL: argmax over all classes seen so far. Task-IL: argmax within the task's own 5 classes |

CIFAR-100 with ViT-B at 224 px is a standard pretrained continual-learning benchmark, used by L2P, DualPrompt and InfLoRA. Twenty 5-class tasks give a longer stream than the common 10-task split, which plasticity measurement needs.

## **Methods**

The adapters are compared under two continual-learning methods: one that does nothing against forgetting, and the standard rehearsal baseline. Both are run inside [Mammoth](https://github.com/aimagelab/mammoth) ([Buzzega et al. 2020](https://arxiv.org/abs/2004.07211)).

**Plain sequential** (`peft-seq`). Train on each task in turn with nothing kept from earlier tasks. It isolates what the adapter itself does, and it is the setting most loss-of-plasticity studies use.

**Experience replay** (`peft-er`). The training step is Mammoth's ER, following [Chaudhry et al. 2019](https://arxiv.org/abs/1902.10486):

1. Take a batch of 64 new images.  
2. If the buffer is not empty, draw 64 stored images (augmented on retrieval) and join them to the batch, giving 128\.  
3. One cross-entropy loss over the joined batch, one optimiser step.  
4. Offer the 64 new images (unaugmented) to the buffer. Each image is offered once per epoch, as in Mammoth.

The buffer holds 500 images and is filled by **class-balancing reservoir sampling** (CBRS, [Chrysakis and Moens 2020](https://proceedings.mlr.press/v119/chrysakis20a.html), Algorithm 1). While the buffer has room, every image is stored. Once it is full, an image of a class that is not among the largest is stored by evicting a random image of the currently largest class. An image of a class that is (or ever was) the largest replaces a random image of its own class with probability m\_c / n\_c (stored over seen). The result is an equal share per class, and within each class a uniform random sample of that class's stream. On this stream every run ends with exactly 5 images for each of the 100 classes. Mammoth's built-in balanced reservoir (`balancoir`) does not keep classes balanced on this stream, so CBRS is implemented as a subclass of Mammoth's `Buffer` (`models/ap_utils/cbrs_buffer.py`, audited in `tests/test_buffers.py`).

**Joint training** (`--joint 1`). All 100 classes trained together as one task, giving each adapter's upper bound on the same data and schedule.

**Larger buffer.** ER with 2,000 images (20 per class), to check that the comparison does not depend on a starved buffer.

## **Training and software**

Every run is one `python main.py` call into a pinned copy of Mammoth, so a run is fully described by its command line and the versions below.

| Item | Setting |
| ----- | ----- |
| Optimiser | Adam (PyTorch defaults, no weight decay), two parameter groups: adapter at the run's learning rate, head at 1e-3. A fresh optimiser at the start of each task |
| Schedule | 5 epochs per task, batch 64 new images (plus 64 replayed under ER), no learning-rate schedule |
| Precision | Stream training with TF32 matrix multiplies (Mammoth `-O 1`); probes under bfloat16 autocast; all diagnostics in float64 |
| Seeds | `--seed` fixes PyTorch and NumPy, LoRA's A initialisation, data order, the validation split and the CBRS sampler. Grid: seeds 1 to 3\. Final: 1 to 5\. Dose: 1 to 3 |
| Mammoth | [aimagelab/mammoth](https://github.com/aimagelab/mammoth), commit `e75a491c69fd729edeb01431afb753d9157d9a81` |
| Libraries | PEFT 0.21.2, timm 1.0.30, plus Mammoth's `requirements.txt` and `open_clip_torch`; `torchao` removed (an old copy conflicts with PEFT's LoRA). Each run records its exact PyTorch, PEFT and timm versions |
| Hardware | One Google Colab Pro+ GPU, 4 data-loader workers, data copied to local disk before training |
| Run time | Without probes: about 17 min (plain) and 31 min (replay). With probes at six points: 72 to 98 min |
| Plugin code | `backbone/vit_peft.py`, `datasets/seq_cifar100_224_20t.py`, `models/peft_seq.py`, `models/peft_er.py`, `models/ap_utils/{cbrs_buffer,probes,hooks}.py`, all written into Mammoth by `ap_install.py` |
| Output | One JSON per run (`runs/<name>.json`), written after every task: configuration, per-task accuracy rows, buffer class counts, adapter diagnostics and probe results. A run is complete when its `complete` flag is set; finished runs are skipped on relaunch |

## **Learning-rate selection**

Each adapter is run at its own best learning rate, chosen separately for each method, so that neither adapter is judged at a rate that suits only the other. Replay is tuned on its own because every step sees twice as many images (64 new plus 64 replayed).

1. Run the grid: adapter learning rates 3e-5, 1e-4, 3e-4, 1e-3 and 3e-3, for each adapter and method, seeds 1 to 3, with no probes.  
2. Score each run by its class-IL accuracy on the validation split after task 20\.  
3. For each adapter and method, pick the rate with the highest mean score over seeds (`ap_run.py --phase select`, written to `selected_lrs.json`). Test data plays no part.

| Adapter | Method | Selected rate |
| ----- | ----- | ----- |
| LoRA | Plain sequential | 3e-5 |
| OFT | Plain sequential | 3e-5 |
| LoRA | Replay | 1e-4 |
| OFT | Replay | 1e-4 |

The head learns at 1e-3 throughout. Without replay, accuracy rises as the adapter rate falls, so 3e-5 is the lowest rate tested rather than an interior optimum. The grid runs also record per-task accuracy and adapter geometry at every rate, and these feed the dose design below.

## **Accuracy and forgetting metrics**

All accuracy metrics come from one accuracy matrix per run, built on the official test split, separately for class-IL and task-IL. A\_{i,j} is the test accuracy (%) on task j after training through task i, with T \= 20\.

```
\text{Final accuracy} = \frac{1}{T}\sum_{j=1}^{T} A_{T,j}
```

```
\text{Learning accuracy} = \frac{1}{T}\sum_{j=1}^{T} A_{j,j}
```

```
\text{Forgetting} = \frac{1}{T-1}\sum_{j=1}^{T-1}\Big(\max_{j \le i < T} A_{i,j} - A_{T,j}\Big)
```

```
\text{BWT} = \frac{1}{T-1}\sum_{j=1}^{T-1}\big(A_{T,j} - A_{j,j}\big)
```

* **Learning accuracy** ([Riemer et al. 2019](https://arxiv.org/abs/1810.11910)) is the accuracy on each task right after it is learned, a direct within-stream measure of plasticity.  
* **Forgetting** follows [Chaudhry et al. 2018](https://arxiv.org/abs/1801.10112); **backward transfer (BWT)** follows [Lopez-Paz and Ranzato 2017](https://arxiv.org/abs/1706.08840).  
* **New-task accuracy curve.** The diagonal A\_{t,t} plotted against t. Its trend is summarised by the slope of a least-squares line over t \= 1 to 20, reported per 10 tasks, with a 95% bootstrap interval over seeds (10,000 resamples). A falling curve means each new task is learned less well. The task-IL version is the cleaner read, because it removes the head's bias toward the newest classes.  
* **Joint upper bound.** Final class-IL accuracy of the joint run with the same adapter.

All numbers are reported as mean ± standard deviation over seeds.

## **Plasticity probe**

Plasticity is measured directly. At fixed points in the stream a copy of the current model is trained on tasks it has never seen. How well that copy learns, compared with the same measurement before task 1, is the plasticity score. The copy is then thrown away, so measuring never changes the run being measured.

**Where the design comes from**

* **Fork and discard.** [Hernandez-Garcia, Figliolia and Millidge 2026](https://arxiv.org/abs/2606.24752) probe plasticity in continually trained language models by training a copy of each checkpoint on a held-out task, discarding its updates, and normalising each model against its own first measurement. The protocol here is the same, applied to adapters.  
* **Random-label probe.** Fitting randomly labelled data measures raw capacity to change, with nothing to transfer. It is the standard trainability test of [Lyle et al. 2023](https://arxiv.org/abs/2303.01486).  
* **Natural probe tasks.** Small, unrelated image tasks in the VTAB-1k format ([Zhai et al. 2019](https://arxiv.org/abs/1910.04867)), the usual yardstick for how well a pretrained vision model adapts.  
* **Head-only control.** A linear probe on frozen features ([Alain and Bengio 2016](https://arxiv.org/abs/1610.01644)), so the adapter's own share of learning can be separated from the quality of the features it carries.  
* **Feature effective rank** ([Roy and Vetterli 2007](https://infoscience.epfl.ch/record/110188)), used as a plasticity diagnostic by [Kumar et al. 2021](https://arxiv.org/abs/2010.14498) and [Lyle et al. 2022](https://arxiv.org/abs/2204.09560).

**Why each choice**

* *Held-out tasks, not the next CIFAR task.* Accuracy on the next stream task mixes plasticity with transfer from similar earlier classes. Unrelated tasks isolate the ability to learn something new.  
* *Best of several learning rates.* A model late in the stream may need a different rate than at the start. A single fixed rate would confuse plasticity loss with a rate mismatch.  
* *Paired design.* Data order and head initialisation depend only on the probe task and the repetition, never on the measurement point. Differences between points therefore reflect the model's state, not sampling noise.  
* *No augmentation in probes.* The probe is a measurement, so it is deterministic given the model.

**Probe tasks**

| Task | Classes | Train / val / test images | Source of each split |
| ----- | ----- | ----- | ----- |
| [DTD](https://arxiv.org/abs/1311.3618) (textures) | 47 | 800 / 200 / 1,880 | Train and val drawn from the official train \+ val pool of partition 1 (seed 11); test \= official test of partition 1 |
| [EuroSAT](https://arxiv.org/abs/1709.00029) (satellite land use, RGB) | 10 | 800 / 200 / 2,000 | One disjoint draw from the 27,000-image pool (seed 21\) |
| [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) (flower species) | 102 | 800 / 200 / 2,000 | Train and val from the official train \+ val pool (seed 31); test \= 2,000 of the official 6,149 (seed 32\) |
| Random labels | 10 | 10,000 (final runs) or the first 2,000 (dose runs), no val or test | CIFAR-10 training images (seed 1019), each given a uniform random label over 10 classes (seed 1020), fixed for every run |

Natural-task images are resized (bicubic, shorter side 224), centre-cropped to 224 × 224 and stored as 8-bit tensors. Random-label images stay at 32 × 32 and are resized to 224 on the fly. Everything is normalised like the stream. The preprocessed set is cached once (`probe_data_224.pt`) and loaded by every run, so all runs see identical probe images.

**Procedure at each probe point t**

1. **Pause** after task t. Final runs probe at t \= 0, 2, 5, 10, 15, 20; dose runs at t \= 0, 5, 10, 20\. t \= 0 is before any task.  
2. **Snapshot** the adapter's trainable tensors.  
3. **Full probe.** For each probe task and repetition r: attach a fresh linear head (768 → number of classes) and a fresh Adam optimiser, then train adapter and head together for S steps at batch 64, sampling with replacement. Run this once at each adapter learning rate in the probe grid, with the head at 1e-3, and restore the snapshot after every run.  
4. **Select** the best rate: highest validation accuracy for natural tasks (ties broken by lower validation loss), lowest final training loss over all images for random labels.  
5. **Head-only control.** The same probe with the adapter frozen: features are computed once, and only a fresh head is trained, at each rate in the head grid. The best rate is chosen the same way.  
6. **Record** training loss and accuracy at S/4, S/2, 3S/4 and S; validation loss and accuracy at the same steps; test loss and accuracy at S (natural tasks); the effective rank of the features of the first 2,000 probe training images.  
7. **Discard and continue.** Restore the snapshot exactly and resume the stream.

Each probe is repeated twice at t \> 0 and five times at t \= 0, to estimate measurement noise. Because the adapter is at its initialisation at t \= 0, those results are cached per adapter, seed and probe configuration (`t0_cache/`) and shared by every learning rate and method.

**Probe settings**

| Setting | Final runs | Dose runs |
| ----- | ----- | ----- |
| Probe points | 0, 2, 5, 10, 15, 20 | 0, 5, 10, 20 |
| Steps, natural tasks | 250 | 250 |
| Random-label probe | 10,000 images, 400 steps (2.6 passes) | 2,000 images, 600 steps (19 passes) |
| Adapter learning rates (full probe) | 1e-4, 3e-4, 1e-3 | 3e-4, 1e-3, 3e-3 |
| Head learning rates (head-only control) | 1e-3, 3e-3, 1e-2 | 3e-4, 1e-3, 3e-3, 1e-2 |
| Repetitions | 2 (5 at t \= 0\) | 2 (5 at t \= 0\) |
| Probe seed | 12345 | 12345 |

The dose settings were set from the final runs: the full probe chose the top rate in most measurements and the head-only random-label control always chose the bottom rate, so both grids were widened in those directions. The random-label probe was made smaller and longer so that it can fit its labels well below chance loss (ln 10 \= 2.30).

**Endpoints**

* **E1, capacity to change:** final training loss on the random-label task. Plasticity loss \= a rise from t \= 0\.  
* **E2, useful learning:** test accuracy on the three natural tasks, averaged. Plasticity loss \= a drop from t \= 0\.

The change at point t, paired by seed, and the adapter's own contribution over the head-only control:

```
\Delta_t = m_t - m_0
```

```
G_t = \mathrm{acc}^{\mathrm{full}}_t - \mathrm{acc}^{\mathrm{head}}_t \;\;(\text{E2}), \qquad G_t = \mathrm{loss}^{\mathrm{head}}_t - \mathrm{loss}^{\mathrm{full}}_t \;\;(\text{E1}), \qquad R_t = G_t / G_0
```

R\_t is the fraction of the adapter's starting contribution that remains after t tasks: 1 means fully retained, 0 means the adapter adds nothing beyond retraining the head.

**Decision rule, fixed before the runs.** Plasticity loss at point t is declared only if both hold:

1. the 95% paired bootstrap interval over seeds (10,000 resamples) for Δ\_t excludes zero in the loss direction, and  
2. the mean |Δ\_t| exceeds the noise floor

```
\delta = 1.96\, s_{\mathrm{rep}} \sqrt{1/n_t + 1/n_0}
```

where s\_rep is the standard deviation of single repetitions at t \= 0 (within seed, averaged over seeds) and n\_t, n\_0 are the repetition counts at t and at 0\.

## **Adapter geometry**

After every task (and at t \= 0\) each run records how far, and in what shape, its adapter has moved. This is what ties a plasticity change to a mechanism.

| Adapter | Quantity | Definition |
| ----- | ----- | ----- |
| OFT | Rotation angles | For every Cayley block R, the eigenvalues are e^{±iθ}. The angles θ of all blocks in all layers are pooled; recorded are the mean, 90th and 99th percentiles and the maximum (radians) |
| LoRA | log10 cond(BᵀB) | Squared ratio of B's largest to smallest singular value, median over layers |
| LoRA | ‖B‖, ‖B‖/‖A‖ | Frobenius norms, median over layers |

All are computed in float64 from the live adapter weights (`adapter_diagnostics` in `hooks.py`).

**The Cayley-chart prediction (H1).** For one 2 × 2 block with skew parameter q, the Cayley map gives a rotation by θ \= 2 arctan q. Its sensitivity to the parameter is

```
\frac{d\theta}{dq} = \frac{2}{1+q^2} = 2\cos^2(\theta/2)
```

So a gradient step of fixed size turns the block less the further it has already turned: by 1% at 11°, 15% at 45° and 50% at 90°. If OFT's plasticity loss comes from its chart, it should appear only once angles reach tens of degrees, and its size should follow cos²(θ/2). LoRA's additive update has no such factor, so under H2 any LoRA loss should instead follow the conditioning of B.

## **Run sets**

Three sets of runs, each named by its prefix in `runs/`. All share the setup above and differ in adapter learning rate, seeds and probe settings.

| Set | Purpose | Arms | Seeds | Probes | Status |
| ----- | ----- | ----- | ----- | ----- | ----- |
| `grid_` | Choose each arm's learning rate; per-task accuracy and geometry at every rate | 2 adapters × {plain, replay} × 5 rates | 1–3 | None | Complete (OFT replay: seeds 1–2) |
| `final_` | Plasticity at each arm's selected rate | 2 adapters × {plain, replay} at the selected rates | 1–2 (LoRA plain: 1–3) | Final settings, t \= 0, 2, 5, 10, 15, 20 | Complete for these seeds |
| `dose_` | Plasticity as the adapter is pushed to move further | Plain sequential: LoRA at 1e-4 and 1e-3; OFT at 1e-4, 1e-3 and 3e-3 | 1–3 | Dose settings, t \= 0, 5, 10, 20 | Running |

**Why the dose set.** At the selected rates the adapters barely move (OFT's mean rotation after 20 tasks is 3°), so these runs cannot test H1, which needs large angles. The adapter learning rate is the dial that sets how far each adapter moves. The dose arms span it up to the highest rate at which training stays stable: 3e-3 for OFT, 1e-3 for LoRA, which diverges in some seeds at 3e-3. Plain sequential is used because it isolates the adapter and runs in half the time of replay.

Still to run: the joint upper bound and the 2,000-image buffer (`--phase joint`, `--phase buffer2k`).

## **Findings so far**

At the learning rates that maximise accuracy, neither adapter loses plasticity over 20 tasks, and what fails is memory, not learning. Pushed to larger updates, OFT starts to lose plasticity in proportion to its learning rate, and LoRA does not. The final runs have 2 seeds (LoRA plain: 3), so their numbers are provisional.

**1\. No plasticity loss at the selected rates** (`final_` runs). Every arm passes the decision rule at every probe point. At t \= 20:

| Arm | E1 Δ (random-label loss) | E1 noise floor | E2 Δ (natural test accuracy, pts) | E2 noise floor |
| ----- | ----- | ----- | ----- | ----- |
| LoRA plain, 3e-5 | \+0.007 | 0.030 | \+0.04 | 0.56 |
| LoRA replay, 1e-4 | −0.005 | 0.034 | \+0.19 | 0.73 |
| OFT plain, 3e-5 | −0.026 | 0.078 | \+0.25 | 0.75 |
| OFT replay, 1e-4 | −0.027 | 0.078 | \+0.19 | 0.75 |

The natural probes start at 88.4 to 88.7% and move by less than 0.7 points at any point. OFT gets slightly better at new tasks after training on the stream (+0.2 to \+0.7), and LoRA stays flat. Inside the stream the same holds: class-IL accuracy on each task right after learning it is 98.9 to 99.1% in every arm, and the task-IL new-task curve rises slightly (+0.17 to \+0.46 points per 10 tasks).

**2\. Class-incremental failure comes from forgetting.** Each task is learned to about 99%, but without replay final class-IL accuracy is 29.5 ± 0.2% for LoRA and 35.3 ± 1.0% for OFT, with forgetting of 73.0 and 67.1 points. Replay with 5 images per class lifts both to 79.0% and cuts forgetting to 21.1 points, with no difference between adapters. Without replay OFT forgets 3 to 6 points less than LoRA (grid: 69.8 vs 72.9; final: 67.1 vs 73.0).

**3\. At the selected rates the adapters barely move.** After 20 tasks OFT's mean rotation angle is 0.052 rad (3.0°) with replay and 0.020 rad (1.1°) without; the 99th percentile is 0.20 rad (11.5°) and 0.11 rad (6.4°). The Cayley factor cos²(θ/2) at 11.5° is 0.99, so H1 predicts no effect here, as observed. LoRA's ‖B‖ grows steadily (0.16 to 0.56 with replay) while its conditioning stays flat (log10 cond 1.4 to 1.6; without replay 1.3 to 2.1).

**4\. Pushed harder, OFT loses plasticity and LoRA does not.** Across the grid's learning rates, the trend of new-task accuracy shows a clean dose response for OFT:

For OFT the trend falls at every step up in rate and turns negative between 3e-4 and 1e-3, with intervals that exclude zero, with or without replay. At 3e-3, OFT also starts lower (95.1% on tasks 1–5, plain) and falls to 93.1% on tasks 16–20. LoRA's trend stays positive up to 1e-3. At 3e-3 LoRA does not decline gradually: some seeds diverge (new-task accuracy 50 ± 41%), so that rate is excluded. The two adapters therefore fail in different ways: OFT loses plasticity gradually as it is pushed, and LoRA holds until it breaks. The dose runs measure this with the probes and the rotation angles together, which is the direct test of H1.

## **Reproduction**

The whole study runs from two files: `ap_install.py` (installer, carrying the plugin, driver and analysis) and a GPU notebook. Paths below assume Google Colab with Drive mounted; any machine with a CUDA GPU works with `--mammoth` and `--out` pointed elsewhere.

1. **Install.** Clones Mammoth at the pinned commit, installs the pinned libraries, removes `torchao`, writes the plugin into Mammoth, and writes `ap_run.py` and `ap_analyze.py` to the output folder.

```sh
python ap_install.py --mammoth /content/mammoth --tools /content/drive/MyDrive/ap
```

2. **Prepare data once.** Downloads CIFAR-100 and the four probe sources and builds the probe cache.

```sh
python /content/drive/MyDrive/ap/ap_run.py --phase prepare
```

3. **Copy data to local disk** before each session, for speed.

```sh
cp -r /content/drive/MyDrive/ap/data /content/apdata
```

4. **Run the phases in order.** Each skips runs that are already complete, so any phase can be relaunched after a disconnect.

```sh
R=/content/drive/MyDrive/ap/ap_run.py
python $R --phase smoke  --data /content/apdata   # 2 short end-to-end checks
python $R --phase grid   --data /content/apdata   # 60 runs, about 21 h
python $R --phase select                          # writes selected_lrs.json
python $R --phase final  --data /content/apdata   # probes at selected rates
python $R --phase dose   --data /content/apdata   # 15 runs, about 16 h
python $R --phase joint  --data /content/apdata
python $R --phase buffer2k --data /content/apdata
```

5. **Analyse.** Writes `report/summary[_<prefix>].md` and figures from whatever runs are complete.

```sh
A=/content/drive/MyDrive/ap/ap_analyze.py
python $A                  # final_ runs: summary.md, plasticity.png, final_*.png
python $A --prefix grid    # summary_grid.md, grid_*.png
python $A --prefix dose    # summary_dose.md, plasticity_dose.png, dose_*.png
```

**Equivalent single run.** Every run is one Mammoth call; for example, the OFT dose arm at 1e-3, seed 2:

```sh
cd /content/mammoth && python main.py --dataset seq-cifar100-224-20t --model peft-seq \
  --optimizer adam -O 1 --n_epochs 5 --batch_size 64 --num_workers 4 --non_verbose 1 \
  --lr 1e-3 --seed 2 --validation 10 --ap_adapter oft \
  --ap_probe_points 0,5,10,20 --ap_probe_lrs 3e-4,1e-3,3e-3 \
  --ap_probe_head_lrs 3e-4,1e-3,3e-3,1e-2 --ap_probe_steps_rand 600 --ap_rand_n 2000 \
  --base_path /content/apdata/mammoth/ --ap_data_dir /content/apdata/probes \
  --ap_out /content/drive/MyDrive/ap/runs --ap_t0_cache /content/drive/MyDrive/ap/t0_cache \
  --ap_run_name dose_oft_seq_lr0.001_s2
```

Replay runs add `--model peft-er --buffer_size 500 --minibatch_size 64 --ap_buffer_strategy cbrs --ap_unique_admission 0`.

## **References**

* Alain and Bengio 2016\. [Understanding Intermediate Layers Using Linear Classifier Probes](https://arxiv.org/abs/1610.01644). Head-only control.  
* Buzzega et al. 2020\. [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://arxiv.org/abs/2004.07211). NeurIPS. [Mammoth](https://github.com/aimagelab/mammoth) framework.  
* Chaudhry et al. 2018\. [Riemannian Walk for Incremental Learning](https://arxiv.org/abs/1801.10112). ECCV. Forgetting measure.  
* Chaudhry et al. 2019\. [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486). Experience replay.  
* Chrysakis and Moens 2020\. [Online Continual Learning from Imbalanced Data](https://proceedings.mlr.press/v119/chrysakis20a.html). ICML. CBRS.  
* Cimpoi et al. 2014\. [Describing Textures in the Wild](https://arxiv.org/abs/1311.3618). CVPR. DTD.  
* Dohare et al. 2024\. [Loss of Plasticity in Deep Continual Learning](https://www.nature.com/articles/s41586-024-07711-7). Nature.  
* Dosovitskiy et al. 2021\. [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929). ICLR. ViT.  
* Helber et al. 2019\. [EuroSAT](https://arxiv.org/abs/1709.00029). IEEE JSTARS.  
* Hernandez-Garcia, Figliolia and Millidge 2026\. [Can Scale Save Us From Plasticity Loss in Large Language Models?](https://arxiv.org/abs/2606.24752) Fork-and-discard probe.  
* Hu et al. 2022\. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR.  
* Krizhevsky 2009\. [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html). CIFAR-10 and CIFAR-100.  
* Kumar et al. 2021\. [Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning](https://arxiv.org/abs/2010.14498). ICLR. Feature rank as a diagnostic.  
* Lopez-Paz and Ranzato 2017\. [Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840). NeurIPS. Backward transfer.  
* Lyle et al. 2022\. [Understanding and Preventing Capacity Loss in Reinforcement Learning](https://arxiv.org/abs/2204.09560). ICLR.  
* Lyle et al. 2023\. [Understanding Plasticity in Neural Networks](https://arxiv.org/abs/2303.01486). ICML. Random-label trainability probe.  
* Nilsback and Zisserman 2008\. [Automated Flower Classification over a Large Number of Classes](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). ICVGIP. Flowers102.  
* Qiu et al. 2023\. [Controlling Text-to-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/abs/2306.07280). NeurIPS. OFT.  
* Riemer et al. 2019\. [Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference](https://arxiv.org/abs/1810.11910). ICLR. Learning accuracy.  
* Roy and Vetterli 2007\. [The Effective Rank: A Measure of Effective Dimensionality](https://infoscience.epfl.ch/record/110188). EUSIPCO.  
* Zhai et al. 2019\. [A Large-scale Study of Representation Learning with the Visual Task Adaptation Benchmark](https://arxiv.org/abs/1910.04867). VTAB-1k.

