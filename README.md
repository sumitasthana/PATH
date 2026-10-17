# Does Unlearning Restore Plasticity? A Probe-Based Test in a Continual Hypernetwork

Working title. An independent reproduction of **UnCLe**, the method introduced
in *An Unlearning Framework for Continual Learning* (Adhikari et al., 2025,
[arXiv:2509.17530](https://arxiv.org/abs/2509.17530)), and a direct test of one
claim that paper makes in writing: that unlearning a task restores the model's
ability to learn new ones.

Plasticity means the ability to learn a new task. It is a separate question
from forgetting, and it has to be measured separately. The paper's evidence for
its claim is final retained accuracy, which mixes the two. This repository
measures plasticity directly, by training a copy of the model on a task it has
never seen and then discarding that copy.

The repository was earlier called CARK, for Component-level Attribution of
Residual Knowledge after Task Unlearning. That name was chosen before the
results were in, and the work went elsewhere: nothing here attributes residual
knowledge to a component. The GitHub URL still says `CARK` and the old links
still work.

A note on names, since the two get conflated. *An Unlearning Framework for
Continual Learning* is the paper. **UnCLe** is the method it introduces, and
what this repository implements. There is no paper called UnCLe.

## How the method works

One hypernetwork produces the weights of a target network from a short code,
one code per task. Learning a task trains the hypernetwork to classify it.
Forgetting a task trains the hypernetwork to turn that task code into noise,
which needs no data at all.

## The four numbers

Accuracy alone hides what goes wrong when unlearning happens inside continual
learning, so the paper reports four.

| Number | What it measures | Wanted |
| --- | --- | --- |
| **RA**, retain accuracy | Average over tasks still held at the end | as high as possible |
| **FA**, forget accuracy | Average over forgotten tasks | chance, which is 10% |
| **Spill** | How much a forget request disturbed every *other* task | as low as possible |
| **Relapse** | How much a forgotten task crept back as later tasks were learned | as low as possible |

Forget accuracy is the one that reads backwards. We want it at chance, not at
zero, because a task that has been removed should leave the model guessing
rather than confidently wrong.

Spill and relapse are the two failure modes the paper identifies in existing
unlearning methods. Neither shows up in accuracy at the end of a run.

## Where to look

- **HTML research write-up:** [What our unlearning experiments show so far](site/index.html),
  with six figures, plain-English explanations, and links to the saved evidence.
- **Results so far:** the [experiment trajectory](https://github.com/sumitasthana/CARK/wiki/Experiment-trajectory),
  a short findings index with dated records and figures.
- **Background:** the [concepts FAQ](https://github.com/sumitasthana/CARK/wiki/Concepts-and-processes)
  explains the method and how to read its results.
- **Inspecting a trained model:** the [model diagnostics guide](docs/MODEL_DIAGNOSTICS.md)
  covers weights, gradients, optimizer updates, and diagonal Fisher scores.
- **Screening a saved forgetting run:** the [research diagnostic](docs/RESEARCH_DIAGNOSTIC.md).
- **Notebooks:** the [notebook guide](notebooks/README.md) separates current
  experiments, CPU reports, and historical notebooks.

The proposed recovery measurements still need an experiment runner.

## Publish the HTML report

The static report is in `site/`. It opens locally without a server. GitHub Pages
currently publishes the root of `main`, so pushing report updates to that branch
updates [the live report](https://sumitasthana.github.io/CARK/site/index.html)
after the Pages build finishes.

The optional **Publish experiment report** workflow publishes only `site/` and
requires the Pages source to be set to **GitHub Actions**. That deployment mode
would serve the report at `https://sumitasthana.github.io/CARK/`.

To rebuild the report from the saved measurements, run
`python scripts/build_experiment_site.py` with Matplotlib and NumPy installed.
Use `--check` to verify that the generated files match the source evidence.
The wiki's Wednesday and Saturday publication schedule is separate.

## Citation

Adhikari, Sayanta, et al. "An Unlearning Framework for Continual Learning."
*arXiv*, 2025.
[doi:10.48550/arXiv.2509.17530](https://doi.org/10.48550/arXiv.2509.17530).

Ha, David, Andrew Dai, and Quoc V. Le. "HyperNetworks." *arXiv*, 2016.
[doi:10.48550/arXiv.1609.09106](https://doi.org/10.48550/arXiv.1609.09106).
