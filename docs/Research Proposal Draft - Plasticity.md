**Plasticity Lives in the Chart:**  
**Why Continually Trained Adapters Stop Learning**

# **Introduction**

**Motivation.** Many deployments adapt a foundation model continually. One parameter-efficient (PEFT) adapter is updated each time a new task or domain arrives. Networks trained this way for long enough lose the ability to learn new tasks. This is called loss of plasticity (LoP). It is documented for standard networks \[2, 12, 13\], vision transformers \[19\], and LLMs \[4\]. No one has measured it for adapters.

Two adapter families dominate. LoRA adds a low-rank update to the frozen weight \[5\]. Orthogonal finetuning (OFT, BOFT, OFTv2) multiplies the frozen weight by a learned rotation \[11, 15, 16\]. OFT is growing in use because it forgets less, and PEFT-Arena finds it has the best single-task trade-off between learning and retention \[14\].

A common assumption is that an adapter slows down because it runs out of capacity. For plain LoRA and OFT this is false. A rank\-$r$ LoRA can overwrite its own subspace at any time. OFT’s reachable weights form a group orbit. Composing rotations gives another rotation, so the reachable set is the same after 1 task or 1,000 tasks. Capacity therefore cannot explain plasticity loss in these adapters. The remaining candidate is the optimization itself.

Every adapter has two parts: a constraint set (the weights it can reach) and a chart (the coordinates gradient descent runs in). No prior work measures next-task learnability of a shared, sequentially trained adapter across adapter types. No prior work separates the effect of the constraint set from the effect of the chart.

**Contribution** We test the hypothesis that plasticity loss in adapters comes from the chart. Our contributions are:

1. **Capacity result.** We show that the reachable set of plain LoRA and OFT does not change with training history. Subspace-freezing methods such as O-LoRA and InfLoRA do lose capacity, by design \[10, 20\]. We treat them as a separate case.

2. **Chart mechanisms.** We derive how each chart degrades. In OFT, the Cayley map slows rotation by a factor of ${cos}^{2}\left({\varphi /2}\right)$ as angles approach $\pi$, and the Cayley–Neumann approximation used by OFTv2 diverges once $\parallel Q\parallel \geq 1$. In LoRA, the factor Gram matrices become ill-conditioned.

3. **Re-anchoring test.** We merge the adapter into the frozen weight and restart the chart at identity. This leaves the reachable set unchanged. Any plasticity it recovers was lost in the chart.

4. **Measurement.** We run the first parameter-matched measurement of next-task learnability for five adapter types over 50+ vision tasks and 23 language tasks.

# **Related Work**

*Prior work and the gap this proposal addresses.*

| Paper | Summary | Gap / Our Extension |
| :---- | :---- | :---- |
| Dohare et al. \[2\]; Lyle et al. \[12, 13\] | Show LoP in continual learning. Identify correlates: weight growth, dead units, rank collapse. Test mechanisms by intervention. | Study full networks trained from scratch. We study adapters on a frozen backbone, where capacity is fixed and adapter geometry is the variable. |
| Hernandez-Garcia et al. \[4\] | Measure LoP in 5M–314M LLMs with a probe task scored by AUC. Onset follows a sublinear scaling law. | Full-parameter pretraining only. We apply their probe protocol to adapters. |
| Parseval reg. \[1\] | Keeping weights near-orthogonal preserves plasticity in continual RL. | Predicts OFT should preserve plasticity. Not tested for adapters. Does not address the chart. |
| OFT / BOFT / OFTv2 \[11, 15, 16\]; PEFT-Arena \[14\] | Rotation-based finetuning that keeps the weight spectrum fixed. OFTv2 assumes $Q$ stays near zero. OFT is best on single tasks. | Single-task only. Long task streams break OFTv2’s small\-$\parallel Q\parallel$ assumption. We test that regime. |
| Shuttleworth et al. \[18\] | LoRA accumulates intruder dimensions over sequential finetuning. | Measures forgetting, not learnability. No OFT comparison. |
| scoRNN, expRNN, trivializations \[3, 7, 8\] | The Cayley map is singular near eigenvalue $-1$. Fixes: the exponential map and periodic re-anchoring. | Recurrent networks. We show the same problem in continual adapter training, caused by rotations that build up across tasks. |
| O-LoRA, InfLoRA, LiteLoRA \[10, 20\] | Assign each task its own orthogonal subspace. Plasticity drops as the space fills. | Capacity loss is built in. We show a single shared adapter has no such loss and study what remains. |

# **Methodology**

The method has four phases: (1) train one adapter across a task stream and probe it at each step, (2) separate the constraint set from the chart, (3) measure chart degradation, and (4) test re-anchoring.

## **Setup and Probe Protocol**

A frozen backbone with weights ${W}_{0}$ carries one adapter $\psi$. The adapter starts at identity ($B=0$ for LoRA, $Q=0$ for OFT). We train it on tasks ${T}_{1},\ldots ,{T}_{N}$ in order, with a learning rate tuned per adapter type. After each task $t$, we copy ${\psi }_{t}$, train the copy on a held-out probe task $P$ for $K$ steps with a fresh optimizer, record its accuracy curve ${a}_{t}^{P}\left({k}\right)$, and discard the copy. The reference curve ${a}_{0}^{P}\left({k}\right)$ comes from a fresh adapter on the original backbone.

## **Constraint Set and Chart**

For LoRA, the constraint set is ${M}_{r}=\{\Delta W:rank\Delta W\leq r\}$ and the chart is $\phi \left({B,A}\right)=BA$. For OFT with block size $b$, the constraint set is $M=\{{W}_{0}{R}^{\top }:R\in SO{\left({b}\right)}^{d/b}\}$ and the chart is $\phi \left({Q}\right)=Cayley\left({Q}\right)$.

**Proposition 1** (History invariance).  *For plain LoRA and OFT, the set of weights reachable from ${\psi }_{t}$ equals the set reachable from ${\psi }_{0}$ for every $t$. A drop in learnability therefore cannot come from lost capacity.*

## **Chart Degradation**

**OFT (Cayley map).** Let $R=\left({I+Q}\right){\left({I-Q}\right)}^{-1}$, and let $Q$ have eigenvalues $\pm i{\theta }_{j}$. The rotation angles are ${\varphi }_{j}=2arctan{\theta }_{j}$. The derivative of the map is

$$d {Cayley}_{Q}\left[{E}\right] = 2 {\left({I-Q}\right)}^{-1}E {\left({I-Q}\right)}^{-1}.$$

In the eigenbasis of $Q$, its $\left({j,k}\right)$ entry has magnitude $2\left|{cos\left({{\varphi }_{j}/2}\right)}\right|\left|{cos\left({{\varphi }_{k}/2}\right)}\right|$. Adam takes steps of size about $\eta$ per coordinate, so each step changes the angle by at most

$$\left|{\Delta {\varphi }_{j}}\right| ≲ 2\eta {cos}^{2}\left({{\varphi }_{j}/2}\right).$$

Undoing a rotation of angle $\varphi$ takes about $tan\left({\varphi /2}\right)/\eta$ steps in the current chart and about $\varphi /\left({2\eta }\right)$ steps from a fresh chart. OFTv2 replaces the inverse with a truncated series, ${R}_{k}=\left({I+Q}\right)\left({I+\sum\limits_{i=1}^{k}{Q}^{i}}\right)$. Its orthogonality error is $O\left({\parallel Q{\parallel }^{k+1}}\right)$, and the series diverges when $\parallel Q\parallel \geq 1$. At that point $R$ is no longer a rotation and the weight spectrum is no longer preserved.

**LoRA (factorization).** Let $G={\nabla }_{\Delta W}L$. One gradient step on $\left({B,A}\right)$ changes the product by

$$\Delta \left({BA}\right) \approx  -\eta \left({G{A}^{\top }A+B{B}^{\top }G}\right).$$

The step size in each direction depends on the Gram matrices ${A}^{\top }A$ and $B{B}^{\top }$. These become ill-conditioned as the factor norms drift apart and intruder directions accumulate \[18\].

## **Re-anchoring**

Every $m$ tasks we reset the chart. For OFT: ${W}_{0}​\leftarrow ​{W}_{0}{R}^{\top }$, $Q​\leftarrow ​0$. For LoRA: ${W}_{0}​\leftarrow ​{W}_{0}+BA$, $B​\leftarrow ​0$, and $A$ is re-initialized, as in ReLoRA \[9\]. In both cases we reset the optimizer state. The reachable set stays the same and the chart derivative returns to its value at identity. This is a dynamic trivialization in the sense of \[7\]. Controls: optimizer reset alone, OFT with the exponential map ($R=expQ$, which has no singularity at $-1$), and L2 regularization toward the initial weights \[6\].

## **Metrics** 

* **Normalized steps to threshold:** ${S}_{t}\left({X}\right)={\tau }_{t}^{P}\left({X}\right)/{\tau }_{0}^{P}\left({X}\right)$, where $\tau$ is the first step at which the probe reaches fraction $X$ of fresh-adapter final accuracy.

* **Probe AUC ratio:** ${\rho }_{t}={AUC}_{t}/{AUC}_{0}$.

* **Plasticity onset** ${t}^{*}$: the first task after which ${S}_{t}>1+\delta$ holds for three consecutive tasks.

* **Chart-recovery ratio:** the share of lost plasticity that re-anchoring recovers,

$${\Gamma }_{t} = \frac{{S}_{t}^{cont}-{S}_{t}^{anchor}}{{S}_{t}^{cont}-1}.$$

* $\Gamma =1$ means all lost plasticity was in the chart. $\Gamma =0$ means none of it was.

# **Experimentation** 

*Experimental setup.*

| Component | Details |
| :---- | :---- |
| Models | Primary: ViT-B/16 (ImageNet-21k, frozen). Secondary: Qwen2.5-0.5B and 1.5B. |
| Vision streams | S1: 50+ natural tasks (20-class ImageNet-1k splits, ImageNet-R, DomainNet). S2: 50 random-label memorization tasks. S3: synthetic rotation stream with known angles ${\varphi }_{max}\in \{\pi /4,\pi /2,3\pi /4,0.95\pi \}$. |
| Language stream | 15-task Long Sequence benchmark \[17\] plus 8 TRACE tasks (23 total). |
| Probes | Related (held-out classes), unrelated (distant domain; Vietnamese language modeling for LLMs \[4\]), random-label. |
| Adapters | LoRA, rsLoRA, OFT (exact Cayley), OFTv2 (Cayley–Neumann), OFT with exponential map, HRA, BOFT. Parameter counts matched within 6% at 3 budgets. |
| Interventions | None; re-anchoring every $m\in \{1,5,10\}$ tasks; optimizer reset; L2 toward initial weights. |
| Diagnostics | OFT: angle distributions, $\parallel Q\parallel$, derivative attenuation $∏{cos}^{2}$, series error $\parallel {R}^{\top }R-I\parallel$. LoRA: Gram condition numbers, $\parallel B\parallel /\parallel A\parallel$, intruder dimensions. All: probe gradient norm, dead units. |
| Validation | 3 seeds $\times$ 3 task orders. Learning rates swept per adapter on ${T}_{1}$, then fixed. Bootstrap confidence intervals. PEFT library version pinned. |

# **Ideal Results** 

1. Continually trained adapters of both types show a measurable onset ${t}^{*}$ on S1 and S2, with ${S}_{t}\left({0.8}\right)>1.5$ by task 50\.

2. Re-anchoring recovers most lost plasticity in both types ($\Gamma \geq 0.8$) without changing what the adapter can represent. This shows the loss is in the chart.

3. OFT with the exponential map does not slow down on S3. Exact Cayley and Cayley–Neumann slow down along the predicted $tan\left({\varphi /2}\right)/\varphi$ curve.

4. Chart diagnostics (angle quantiles, Gram conditioning) predict ${S}_{t}$ with Spearman correlation above 0.8 within each adapter type.

5. A practical rule: re-anchor every $m$ tasks. It adds no parameters and negligible compute.

**Fallback if results are negative.** We run the re-anchoring experiment first, in weeks 3–4, as a fixed decision point.

* If $\Gamma$ is high only for OFT, the paper narrows to OFT: *Rotations Don’t Run Out, Their Charts Do*. LoRA becomes the comparison case.

* If $\Gamma$ is low for both, we use a neutral framing: *Isometry or Saturation? Plasticity of Orthogonal and Low-Rank Adapters Under Long Task Streams*. The paper then reports when adapter plasticity loss occurs and whether spectrum preservation or chart saturation dominates.

* If no adapter loses plasticity on natural streams, S2 and S3 still give a positive result. The null result becomes a finding about how much information each task carries.

The experiments are the same in every case. Only the title and main claim change.

# **Potential Limitations** 

1. **Real tasks may not cause saturation.** OFT angles may stay far from $\pi$ on natural streams. The Cayley test would then rely on the synthetic S3 stream.

2. **Transfer can hide plasticity loss.** Features shared with earlier tasks can make the probe easier. Random-label probes reduce this problem but do not remove it.

3. **Parameter matching can be disputed.** OFT has $d\left({b-1}\right)/2$ effective parameters but stores $db$, and matching fails for non-square layers. We report both counts and restrict adapters to attention layers.

4. **Scale.** We stop at 1.5B parameters. Plasticity loss starts later in larger models \[4\], so the effect may be delayed at scale.

*Evaluation protocol.*

| Test | Method |
| :---- | :---- |
| Chart ablation ($2\times 3$) | Cayley, Cayley–Neumann, and exponential map, each with and without re-anchoring. Measure ${\Gamma }_{t}$ and ${S}_{t}$. |
| Synthetic check | On S3, regress $S$ on the known ${\varphi }_{max}$ and compare to the $tan\left({\varphi /2}\right)/\varphi$ prediction. |
| Transfer vs. plasticity | Compare ${\rho }_{t}$ across related, unrelated, and random-label probes. |
| Re-anchoring frequency | Re-anchor every $m\in \{1,5,10\}$ tasks. Report recovered plasticity and the cost to forward transfer. |
| Budget sensitivity | Repeat the main experiment at low, mid, and high parameter budgets. Check whether ${t}^{*}$ moves. |
| LLM replication | Qwen2.5-0.5B and 1.5B on the 23-task stream. 3 adapter types. Main comparisons only. |

# **References**

\[1\] W. Chung, L. Cherif, D. Meger, and D. Precup. Parseval regularization for continual reinforcement learning. In *NeurIPS*, 2024\.

\[2\] S. Dohare, J. F. Hernandez-Garcia, Q. Lan, P. Rahman, A. R. Mahmood, and R. S. Sutton. Loss of plasticity in deep continual learning. *Nature*, 632:768–774, 2024\.

\[3\] K. Helfrich, D. Willmott, and Q. Ye. Orthogonal recurrent neural networks with scaled Cayley transform. In *ICML*, 2018\.

\[4\] J. F. Hernandez-Garcia, P. Figliolia, and B. Millidge. Can scale save us from plasticity loss in large language models? *arXiv:2606.24752*, 2026\.

\[5\] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In *ICLR*, 2022\.

\[6\] S. Kumar, H. Marklund, and B. Van Roy. Maintaining plasticity in continual learning via regenerative regularization. *arXiv:2308.11958*, 2023\.

\[7\] M. Lezcano-Casado. Trivializations for gradient-based optimization on manifolds. In *NeurIPS*, 2019\.

\[8\] M. Lezcano-Casado and D. Martínez-Rubio. Cheap orthogonal constraints in neural networks: A simple parametrization of the orthogonal and unitary group. In *ICML*, 2019\.

\[9\] V. Lialin, S. Muckatira, N. Shivagunde, and A. Rumshisky. ReLoRA: High-rank training through low-rank updates. In *ICLR*, 2024\.

\[10\] When one adapter speaks for many: Discovering low-rank redundancy in continual fine-tuning. *arXiv:2606.28117*, 2026\.

\[11\] W. Liu, Z. Qiu, Y. Feng, Y. Xiu, Y. Xue, L. Yu, H. Feng, Z. Liu, J. Heo, S. Peng, Y. Wen, M. J. Black, A. Weller, and B. Schölkopf. Parameter-efficient orthogonal finetuning via butterfly factorization. In *ICLR*, 2024\.

\[12\] C. Lyle, Z. Zheng, E. Nikishin, B. Avila Pires, R. Pascanu, and W. Dabney. Understanding plasticity in neural networks. In *ICML*, 2023\.

\[13\] C. Lyle, Z. Zheng, K. Khetarpal, H. van Hasselt, R. Pascanu, J. Martens, and W. Dabney. Disentangling the causes of plasticity loss in neural networks. In *CoLLAs*, 2024\.

\[14\] PEFT-Arena: Understanding parameter-efficient finetuning from a stability–plasticity perspective. *arXiv:2605.28819*, 2026\.

\[15\] Z. Qiu, W. Liu, H. Feng, Y. Xue, Y. Feng, Z. Liu, D. Zhang, A. Weller, and B. Schölkopf. Controlling text-to-image diffusion by orthogonal finetuning. In *NeurIPS*, 2023\.

\[16\] Z. Qiu, W. Liu, A. Weller, and B. Schölkopf. Orthogonal finetuning made scalable. In *EMNLP*, 2025\.

\[17\] A. Razdaibiedina, Y. Mao, R. Hou, M. Khabsa, M. Lewis, and A. Almahairi. Progressive prompts: Continual learning for language models. In *ICLR*, 2023\.

\[18\] R. Shuttleworth, J. Andreas, A. Torralba, and P. Sharma. LoRA vs full fine-tuning: An illusion of equivalence. In *NeurIPS*, 2025\.

\[19\] Vision transformers that never stop learning. *arXiv:2603.07787*, 2026\.

\[20\] X. Wang, T. Chen, Q. Ge, H. Xia, R. Bao, R. Zheng, Q. Zhang, T. Gui, and X. Huang. Orthogonal subspace learning for language model continual learning. In *Findings of EMNLP*, 2023\.