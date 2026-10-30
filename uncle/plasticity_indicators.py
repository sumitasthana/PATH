"""Weight change and weight norm at a task boundary.

The adapter plasticity study in `reference/adapter-plasticity-study` reports a
list of cheap plasticity indicators and finds that the plainest of them, how
far the weights moved, predicts probe accuracy better than effective rank,
dormant units or curvature. That study moves a small adapter and leaves a large
backbone frozen. This framework has no frozen part: a generator writes every
weight of the target network from a 32-number task code. So "how far the
weights moved" has to be defined before it can be measured.

It is defined here twice, because two different things can move.

    delta_w_target     how far the generated target weights moved
    delta_w_generator  how far the generator's own parameters moved

Both are relative: the distance from the starting vector divided by the
starting vector's norm, which is the adapter study's
dW = ||W - W0||_F / ||W0||_F. Relative distances can be compared between the
two, and against that study's numbers.

The target weights are read through one fixed reference code that belongs to no
task. Holding the input constant is the whole point. A task's own code moves
while the task is being learned, so weights generated from it mix two causes:
the generator changed, and the code changed. The reference code never changes,
so `delta_w_target` isolates the first.

Nothing here trains, forks or evaluates. One call generates one set of weights
and takes two norms, so a boundary measurement costs milliseconds and needs no
data. The generator trunk is Linear and ReLU only, with no dropout and no batch
normalisation, so train and eval mode make no difference to what it produces
and this module does not touch module modes.

Typical use, once per task boundary:

    code = reference_code(hypernet)
    start = flatten(hypernet, code)             # at t = 0, keep this
    ...                                          # learn a task
    record = weight_change(flatten(hypernet, code), start)
"""

import torch

# Fixed so that every run, every seed and every order reads the same reference
# code. Changing it makes new numbers incomparable with the old ones.
REFERENCE_SEED = 20261030


def reference_code(hypernet, seed=REFERENCE_SEED):
    """One fixed task code, belonging to no task and never trained.

    Drawn on the CPU from its own generator so the value does not depend on the
    device, on the run's seed, or on how much randomness the run has already
    used. Calling this twice gives the same code.
    """
    stream = torch.Generator().manual_seed(seed)
    code = torch.randn(hypernet.config.code_dim, generator=stream)
    return code.to(hypernet.config.torch_device)


@torch.no_grad()
def flatten(hypernet, code):
    """The two weight vectors that describe the model right now, on the CPU.

    `target` is every weight the generator writes for this code, concatenated
    in a fixed name order so two calls line up element for element. `generator`
    is the trunk and head parameters, which are what learning actually updates.

    Both are moved to the CPU. A ResNet50's weights are about 100 MB in float32,
    and a boundary measurement should not hold graphics memory while the next
    task trains.
    """
    weights = hypernet.weights_from_code(code)
    target = torch.cat([weights[name].reshape(-1) for name in sorted(weights)])
    generator = torch.cat([p.reshape(-1) for p in hypernet.generator_parameters()])
    return {"target": target.detach().float().cpu(),
            "generator": generator.detach().float().cpu()}


def weight_change(now, start):
    """Relative distance travelled, and current size, for both vectors.

    `start` is a `flatten` result from the boundary this run counts from,
    normally t = 0. Comparing a vector with itself gives exactly zero, which is
    what the t = 0 row should read.
    """
    record = {}
    for part in ("target", "generator"):
        here, began = now[part], start[part]
        record[f"norm_{part}"] = float(here.norm())
        record[f"delta_w_{part}"] = float((here - began).norm() / began.norm())
    return record
