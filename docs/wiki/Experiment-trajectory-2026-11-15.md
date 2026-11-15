# Experiment trajectory: 15 November 2026

Back to the [index](Experiment-trajectory).

## Audit of all saved revised-study JSON logs

This is a retrospective analysis on 15 November, not a new training run.
The run dates are not reconstructed from object timestamps. The audit read all
20 JSON/JSONL objects under the `plasticity_revised_v1` paths in R2: fifteen
indicator streams, two stream summaries, the rescaling control, the raw probe
log and its summary. The streams contain 165 boundary readings. The probe log
contains exactly 990 unique grid cells: fifteen streams, two endpoints, eleven
repetitions and three learning rates. All fifteen rescaling pairs are present.
The 60 binary checkpoint/baseline objects were inventoried but not downloaded.
No new GPU experiment was run, and no claim is made about unsaved console logs
or interrupted attempts absent from these records.

Reproduction: `python scripts/audit_plasticity_revised_logs.py outputs/revised_log_audit`.
The local cache preserves R2 keys beneath that directory. The script validates
coverage and emits per-stream comparisons, rescaling contrasts and SHA256
hashes of the source records. It does not contact R2 or train models.

Sources in bucket `uncle-experiments`:

- `uncle/indicators/plasticity_revised_v1/order_{A,B,C}/seed_{0,1,2,3,4}.jsonl`
- `uncle/indicators/plasticity_revised_v1/job2_summary.json`
- `uncle/indicators/plasticity_revised_v1/job3_summary.json`
- `uncle/indicators/plasticity_revised_v1/rescaling_control.json`
- `uncle/matched_probe/plasticity_revised_v1/revised_20class_train4800.jsonl`
- `uncle/matched_probe/plasticity_revised_v1/revised_20class_train4800_summary.json`

## Correction: the probe trains the generator and the code

Every probe record says `arm="full"`, `code_only=False`, `train_code=True`.
Notebook 47 configures precisely that arm. In `uncle/probe.py`, the optimizer
receives generator parameters and the fresh task code. The original state is
restored afterwards. This is full adaptation on a discarded branch, not
code-only learning on a frozen generator.

This was checked against both recorded probe revisions,
`18c805457dbe8d774912526c18a1887431131f2a` (330 records) and
`34bec79cab3f1f2bdc4b3377cf8c45960ba66810` (660 records). Their `uncle/probe.py`
implementations have no diff. The earlier frozen-generator descriptions in
the index, 14 November record and paper skeleton are superseded by this audit.
The accuracy values survive; their interpretation changes.

## Finding 1: the rank drop is concentrated at the first task

| Mean over fifteen streams | Boundary 0 | Boundary 1 | Boundary 9 |
| --- | ---: | ---: | ---: |
| Effective rank, in-stream images | 5.563 | 1.723 | 2.094 |
| srank, in-stream images | 215.60 | 66.13 | 81.13 |
| Dormant fraction, threshold 0.01 | 15.99% | 36.62% | 34.21% |

Effective rank falls at the first task in all fifteen runs. From boundary 1 to
9 it rises in ten and falls in five. The dormant fraction decreases from
boundary 1 to 9 in all fifteen. Rank has the same first-step and rebound counts
on held-out Tiny ImageNet and EuroSAT images, although those readings are not
independent replications.

The endpoint drop therefore does not describe a steady accumulation of
damage. The largest aggregate shift is the transition out of initialization.
Calling that shift specialization is a hypothesis. A trained intermediate
plasticity reference is still needed to distinguish it from loss and recovery.

## Finding 2: faster useful adaptation is already visible in the saved curves

The following comparisons fix the learning rate instead of selecting it from
validation scores. Each entry averages the same eleven repetitions per stream.

| Probe learning rate | Fresh model, 250 steps | Trained model, 62 steps | Trained model, 250 steps |
| --- | ---: | ---: | ---: |
| 0.0001 | 24.57% | 38.32% | 43.95% |
| 0.0003 | 24.92% | 38.54% | 42.33% |
| 0.001 | 14.34% | 21.68% | 32.42% |

At each rate, all fifteen stream-level averages at 62 steps exceed their
matched fresh-model averages at 250 steps. At rate 0.0001 the smallest margin
is 11.26 points. This establishes a useful budget comparison using about a
quarter of the updates; it is not an estimate of an exact fourfold speedup.
Only four intermediate readings were saved, and no exact time-to-threshold is
identified. The fifteen streams share five seed labels; the 45 comparisons
are not 45 independent experiments.

The final gain also remains positive in all fifteen streams at each fixed
rate. Thus validation-based rate selection is not necessary for the endpoint
gain. Ordinary forward transfer within Tiny ImageNet remains a possible cause.
The revised random-label and external-task training probes remain unmeasured.

## Finding 3: the weight-scale contribution depends on the intervention path

Use the four arms to separate two possible paths between the same checkpoints.
First change the weight direction, then the norm; or first change the norm,
then the direction. Here direction means the entire weight vector divided by
its norm, not a task code or an independently measured representation.

For effective rank on in-stream images:

| Contrast | Mean change |
| --- | ---: |
| Actual before-to-after unlearning | +1.5801 |
| Change norm at the original weight direction | +1.3141 |
| Change norm at the post-unlearning weight direction | +0.8257 |
| Difference between those scale effects | -0.4884 |

The existing 52.25% figure divides 0.8257 by 1.5801. The other path gives
83.16%, dividing 1.3141 by 1.5801. Both use the same four measured arms.
The two scale effects differ by about 31% of the raw change. The corresponding
path shares are 53.69% versus 82.85% on held-out Tiny ImageNet and 43.90% versus
72.87% on EuroSAT.

Scale and weight direction therefore do not act additively in these readings.
The 52% figure is one particular intervention contrast, not a unique causal
partition into scale and representation. Even the sign depends on the run:
shrinking the original weights raises in-stream rank in eleven pairs and
lowers it in four. These are descriptive intervention results; this audit
does not establish a population-level interaction with a significance test.

Uniformly rescaling weights can change the network's function through its
fixed BatchNorm buffers and other operations. No accuracy or plasticity probe
was run on the rescaled arms, so these results do not establish a way to
change rank while preserving behavior.

## Interpretation and next decision

The strongest candidate contribution is a protocol-dependent diagnostic:
an early rank drop coexists with faster full adaptation, and a two-way scale
control shows that a single scale correction has no unique attribution.
The present logs support these observations. They do not yet identify the
mechanism or prove that plasticity is preserved at every point in the stream.

Two earlier arguments need narrower wording. A near-zero correlation across
fifteen streams is not proof of no predictive relationship. Also, erasing
performance on an old task is not itself a demonstrated loss of ability to
learn a new one. No boundary-10 plasticity probe is in these logs, so the
unlearning arm cannot yet serve as a known plasticity-loss control.

The planned accuracy-spread gate also did not pass: the job 2 summary records
`gate_passed=false`. The completed runs remain evidence, but they should not
be described as satisfying the planned three-point seed-spread requirement.

Broad novelty should not rest on rank failing to predict plasticity:
[Wang et al., 2026](https://arxiv.org/abs/2605.09044) already construct
counterexamples for representation and neural-tangent-kernel rank.
[Prakash et al., 2026](https://proceedings.mlr.press/v306/prakash26b.html)
emphasize curvature at new-task initialization. Curvature measured only at a
trained task code would not directly test that proposed mechanism.

Priority: use the revised random-label probe to test whether the advantage
extends beyond useful transfer; measure curvature in the parameter space the
full probe actually trains; then use the saved middle checkpoint to test for
an intermediate dip. A same-checkpoint comparison across reference codes and
BatchNorm policies would help test whether the indicator describes the trained
learner or the particular route used to read it. Such comparisons need matched
images and sample counts. The post-probe rank uses 2,000 images and a trained
code, so it must not be divided by the 256-image reference-code rank as though
they were measurements under one protocol.

The user's pending random-direction and cross-seed rank-convergence studies
remain separate. This audit did not run or claim to complete them. Learning
speed can now be reported at the four saved steps without additional training.
