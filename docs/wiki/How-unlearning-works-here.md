# How unlearning works here

Unlearning a task is a short training run. It teaches the hypernetwork to turn
that task's code into junk, while a second term in the loss holds every other
task's weights where they already were. No images from the forgotten task are
used. The code for the task is all the procedure needs.

For the general ideas behind this (what machine unlearning is, what a
preservation loss is, how forgetting is measured), see the
[concepts FAQ](https://github.com/sumitasthana/PATH/wiki/Concepts-and-processes).
This page is only about what the code in this repository actually does.

## The pieces involved

The model is one hypernetwork. It takes a short vector, called a task code, and
produces the weights of a classifier. One code per task. The classifier itself
stores nothing between tasks: it is rebuilt from the code each time.

Two sets of numbers matter here:

- The **generator weights**, called phi in the paper. These are the trunk and
  the heads of the hypernetwork. Learning and forgetting both change these.
- The **task codes**, one per task. A forget request does not change them.

Before a request starts, the code takes a **snapshot**: a frozen copy of the
hypernetwork as it is at that moment. The snapshot is the reference that the
preservation term compares against.

## What one forget request does

In `uncle/trainer.py`, the method `forget(task, protected, burn_in)` runs this
loop. One request is a fixed number of small optimizer steps, 100 by default.
Each step does the following.

1. Feed the forgotten task's code through the hypernetwork and read the **raw**
   output, before the per-layer scaling is applied.
2. Draw fresh Gaussian noise of the same shape, 10 draws by default, and
   average the squared distance between the raw output and each draw. This is
   the forget term.
3. For every protected task, generate its weights now and compare them with the
   same task's weights from the snapshot, squared distance again. This is the
   preservation term.
4. Add them: `loss = gamma * forget_term + preserve_term`, with gamma at 0.01
   by default. One backward pass, one Adam step on the generator weights only.

The task codes and the snapshot are frozen for the whole request, so each
protected task's reference weights are generated once and reused.

## What the forget term actually minimizes

It does not make the output random. The noise draws have mean zero, so the
average squared distance works out to the squared length of the raw output plus
a constant. That is smallest when the raw output is zero. So the generated
weights for the forgotten task get pulled toward zero, not toward a random
vector.

Either way the task stops working, which is why the objective still does its
job. But it is worth stating plainly, because the name suggests something else.

## Settings that control it

| Setting | Default | What it does |
| --- | --- | --- |
| `gamma` | 0.01 | How hard the forget term pushes, relative to preservation |
| `beta` | 0.1 | The same weight, but used during learning instead |
| `burn_in` | 100 | Steps in one forget request |
| `burn_in_decay` | 0.9 | Each later forget request is 10% shorter |
| `burn_in_min` | 20 | Floor on that shrinking |
| `noise_samples` | 10 | Fresh noise draws averaged per step |
| `forgetting_learning_rate` | unset | Falls back to the learning rate |

The shrinking burn-in follows Appendix C of the paper: the first forget needs
the most work, later ones need less.

## What a forget request leaves behind

Three things survive on purpose or by omission, and all three matter when
reading results.

- **The task code stays.** Only the generator weights move. The code is still
  there and can still be fed in.
- **The BatchNorm statistics stay.** The paper's equation 3 covers generated
  parameters only, so the running means and variances recorded from the
  forgotten task's data are untouched.
- **The task stays on the protected list.** In `uncle/experiment.py`, a
  forgotten task is still passed to `protected` on every later request. That is
  what holds it near its forgotten state as new tasks are learned, instead of
  letting it drift back.

## How we check it worked

Four numbers, measured after every request:

- **FA**, forget accuracy. Should sit at chance, 10% here, not at zero. A model
  that is confidently wrong has not forgotten.
- **RA**, retain accuracy, averaged over tasks still held.
- **Spill**: how much this forget request disturbed the other tasks.
- **Relapse**: how much a forgotten task crept back while later tasks were
  learned.

Low accuracy on the forgotten task is the weakest of these. It is the first
thing to check and the last thing to trust.
