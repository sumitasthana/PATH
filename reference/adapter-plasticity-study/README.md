# Adapter plasticity study: read-only snapshot

Snapshot taken 17 October 2026 from `C:\LangChain\Algoverse\adapter-plasticity-study`,
a teammate's working folder. That folder is not a Git repository, so there is no commit
to pin this against. Treat every file here as a copy that will go stale.

**Nothing here runs in this repository.** The code is a plugin for
[Mammoth](https://github.com/aimagelab/mammoth) at a pinned commit, built for Colab with
an A100. It needs an installer, a pinned timm and peft, and a downloaded CIFAR-100. It is
here to be read, not executed. Do not import from it and do not wire it into `uncle/`.

## Why it is in this repository

The adapter study and the unlearning work in this repository meet at one question:
does a model keep its ability to learn new tasks after continued training? That ability
is called plasticity, and it is a separate question from forgetting.

`HANDOFF-unlearning-and-plasticity.md` is the document that connects them. It was written
on 14 October 2026 to pass this repository's unlearning results to the adapter study.
Its section 9 proposes the shared experiment: UnCLe claims in writing that unlearning
restores plasticity, no plasticity probe has ever tested that claim, and the mechanism
the paper gives for it is measurably wrong. Section 9 also records an open decision:
whether hypernetworks are in scope for the adapter study at all.

## What each file is

| File | What it holds |
| --- | --- |
| `Adapter Plasticity WiP.md` | The study's methodology and provisional findings. The most useful file here. Read this first. |
| `HANDOFF-unlearning-and-plasticity.md` | The bridge between the two projects. Section 9 holds the proposed shared experiment. |
| `vit_lae.py` | The model: a frozen ViT-B/16, one adapter (LoRA, OFT, or the LAE paper's KVLoRA), one linear head. |
| `lae_lora.py` | The training step, plus the startup checks that abort a run when something is not as claimed. |
| `seq_cifar100_lae.py` | CIFAR-100 split into 10 tasks of 10 classes, with the LAE paper's exact transforms and class orders. |
| `ap_run.py` | The driver. Reproduces a published number first, then changes one setting at a time and checks each change. |
| `presets.yaml` | Every run configuration and every gate rule. The driver accepts no free-form training flags. |

## What was left out

`ap_install.py`, a 146 KB installer that carries all the files above as embedded strings.
It also carries files the five above do not: `vit_peft.py`, `peft_seq.py`, `peft_er.py`,
and `ap_utils/`, which include the plasticity probe and the adapter diagnostics
(rotation angles, condition numbers). Those are the parts most relevant to the shared
experiment. They were not copied because reading them as quoted strings is painful. If
the shared experiment goes ahead, extract just the probe code then.

## One term worth knowing before reading the code

`rel_dW` appears throughout. It is one number saying how far the adapter moved the model
weights away from their pretrained values, as a fraction:

```
rel_dW = ||W_eff - W0|| / ||W0||
```

`W0` is the frozen pretrained weight matrix, `W_eff` is what the layer computes once the
adapter is applied, and `||X||` is the Frobenius norm, the square root of the sum of all
squared entries. If `rel_dW` is 0.05, the weights have moved by 5 percent of their own
size. LoRA and OFT change weights in different shapes, so their raw parameter values
cannot be compared; `rel_dW` is the common measure that can. It gives size only, not
direction.
