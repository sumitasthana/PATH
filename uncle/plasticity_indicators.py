"""Cheap plasticity indicators, read at a task boundary.

Five measurements live here: how far the weights moved, how large they are,
how many directions the features use, how many units have gone quiet, and how
much one update moves the loss. None of them trains or forks, so the whole set
rides along with a stream run for the cost of one weight generation and a few
forward passes.

The first four are read once at a task boundary. The fifth rides on the
training loop itself, through the trainer's own step callback, and so is the
one measurement here that needs the task it is watching.

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

The rank and dormancy measurements read the network the reference code
generates, for the same reason: they ask what kind of network this generator
now writes, with nothing else allowed to vary.

Typical use, once per task boundary:

    code = reference_code(hypernet)
    start = flatten(hypernet, code)             # at t = 0, keep this
    ...                                          # learn a task
    record = weight_change(flatten(hypernet, code), start)
    record.update(ranks(hypernet, target, trainer, image_sets, code))
    record.update(dormancy(hypernet, target, trainer, image_sets, code))

And once per task, around the learning itself:

    rows, watch = loss_progress_recorder(hypernet, target, trainer, task)
    trainer.learn(task, protected, on_step=watch)
"""

import torch
from torch.func import functional_call
from torch.nn.functional import cross_entropy

from .probe import _features_and_logits, effective_rank

# Fixed so that every run, every seed and every order reads the same reference
# code. Changing it makes new numbers incomparable with the old ones.
REFERENCE_SEED = 20261030

# Kumar et al. 2021 define srank with delta = 0.01, and the adapter study uses
# that value, so the two sets of numbers line up.
SRANK_DELTA = 0.01


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


# -- how many directions the features use ------------------------------------

def srank(features, delta=SRANK_DELTA):
    """The smallest number of directions holding all but `delta` of the signal.

    Kumar et al. 2021. Sort the singular values largest first, add them up as a
    running share of the total, and count how many are needed to reach
    1 - delta. Unlike effective rank this is a whole number, and it ignores the
    long tail of tiny directions rather than letting them contribute.

    Both are reported because each misses something. Effective rank is swayed
    by a tail of near-zero directions; srank throws that tail away entirely.
    """
    values = torch.linalg.svdvals(features.double())
    shares = torch.cumsum(values, 0) / values.sum()
    return int((shares < 1 - delta).sum().item()) + 1


@torch.no_grad()
def features(hypernet, target, trainer, images, code, *, batch_size=64):
    """The classifier's input for one set of images, under one fixed code.

    The network is the one the reference code generates, so this is a property
    of the generator rather than of any learned task.

    Batch-normalisation statistics come from `trainer.buffer_template`, the
    clean set every new task starts from. They are the same at every boundary,
    so a change in these features can only come from the weights. Using a
    learned task's statistics instead would let two things move at once.

    Evaluation mode is forced for the same reason it is forced in the probe: a
    BatchNorm layer left in training mode rescales using whatever batch is in
    front of it, which revives a nearly dead signal and hides what is being
    measured. The previous modes are put back afterwards.
    """
    weights = hypernet.weights_from_code(code)
    modes = hypernet.training, target.training
    hypernet.eval()
    target.eval()
    collected = []
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(trainer.device)
            # task is unused once weights and buffers are both supplied.
            captured, _ = _features_and_logits(hypernet, target, trainer, None, batch,
                                               weights=weights,
                                               buffers=trainer.buffer_template)
            collected.append(captured.float().cpu())
    finally:
        hypernet.train(modes[0])
        target.train(modes[1])
    return torch.cat(collected)


def ranks(hypernet, target, trainer, image_sets, code, *, batch_size=64):
    """Effective rank and srank on each fixed image set.

    `image_sets` maps a name to a tensor of images. Three sets are worth
    keeping apart: images from the learned tasks, images from a held-out task
    in the same dataset, and images from a different dataset altogether. The
    adapter plasticity study measured rank on the current task's images first
    and had to redo the whole analysis on fixed sets, because that mixed "the
    representation of this data changed" with "the representation changed".

    Returns one entry per set, keyed by `{measure}_{set name}` so the result
    merges straight into the boundary record.
    """
    record = {}
    for name, images in image_sets.items():
        matrix = features(hypernet, target, trainer, images, code, batch_size=batch_size)
        record[f"effective_rank_{name}"] = effective_rank(matrix)
        record[f"srank_{name}"] = srank(matrix)
        record[f"images_{name}"] = int(matrix.shape[0])
    return record


# -- how many units have stopped responding ----------------------------------

# Sokar et al. 2023 report several cut-offs rather than one, and the adapter
# study keeps all three, so the same three are kept here.
THRESHOLDS = (0.01, 0.025, 0.1)


def activation_sites(module, run):
    """Total absolute activation per unit at every ReLU call `run` makes.

    A site is one call, not one module. Torchvision's ResNet blocks build a
    single `nn.ReLU` and call it three times, so hooking modules and keying by
    name would fold three different places in the network into one. Calls
    happen in a fixed order for a fixed architecture, so the position of a call
    identifies it, and the same position means the same place on every batch.

    `run` may push several batches through. The sums accumulate across all of
    them, so the result covers the whole image set and not just the last batch.

    Totals are returned rather than means. Every unit in a site is summed over
    the same number of values, and `dormant_fraction` divides each site by its
    own mean, so that number cancels and never needs carrying around.
    """
    sums, position = [], [0]

    def hook(_module, _inputs, output):
        index = position[0]
        values = output.detach().float().abs()
        # Unit means channel for a convolution and neuron for a linear layer,
        # which is dimension 1 either way. Everything else is summed over.
        total = values.sum(dim=[0] + list(range(2, values.dim())))
        if index == len(sums):
            sums.append(total)
        else:
            sums[index] += total
        position[0] = index + 1

    handles = [child.register_forward_hook(hook)
               for child in module.modules() if isinstance(child, torch.nn.ReLU)]
    try:
        run(lambda: position.__setitem__(0, 0))
    finally:
        for handle in handles:
            handle.remove()
    return sums


def dormant_fraction(sums, thresholds=THRESHOLDS):
    """The share of units that have gone quiet, at each cut-off.

    Sokar et al. 2023. A unit's score is its mean absolute activation divided
    by the mean over all units in the same site. The per-unit division by the
    number of values cancels between the two, so `sums` holds totals and the
    scores come out the same. A score of 1 means an
    ordinary unit and a score near 0 means one that has stopped contributing.
    Scoring within a site matters: sites differ in scale by orders of
    magnitude, and a single network-wide mean would mark whole quiet layers as
    dormant and whole loud ones as healthy.

    A site whose units are all exactly zero has no mean to divide by. Every
    unit there has stopped responding, so the whole site counts as dormant.
    """
    flagged = {threshold: 0 for threshold in thresholds}
    units = 0
    for total in sums:
        units += total.numel()
        if total.mean() <= 0:
            for threshold in thresholds:
                flagged[threshold] += total.numel()
            continue
        score = total / total.mean()
        for threshold in thresholds:
            flagged[threshold] += int((score <= threshold).sum())
    record = {f"dormant_{threshold}": flagged[threshold] / units
              for threshold in thresholds}
    record["units"] = units
    return record


@torch.no_grad()
def dormancy(hypernet, target, trainer, image_sets, code, *, batch_size=64):
    """Dormant-unit fractions on each fixed image set.

    Same inputs and the same reasoning as `ranks`: the network the reference
    code generates, clean batch-normalisation statistics, evaluation mode,
    fixed image sets. Nothing is changed and nothing is left behind.
    """
    weights = hypernet.weights_from_code(code)
    modes = hypernet.training, target.training
    hypernet.eval()
    target.eval()
    record = {}
    try:
        for name, images in image_sets.items():
            def run(reset, images=images):
                for start in range(0, len(images), batch_size):
                    reset()
                    _features_and_logits(hypernet, target, trainer, None,
                                         images[start:start + batch_size].to(trainer.device),
                                         weights=weights,
                                         buffers=trainer.buffer_template)
            for key, value in dormant_fraction(activation_sites(target, run)).items():
                record[f"{key}_{name}"] = value
    finally:
        hypernet.train(modes[0])
        target.train(modes[1])
    return record


# -- how much one update actually moved the loss ------------------------------

# The adapter study samples this every tenth update rather than every update.
# One extra forward pass in ten is about 5 percent added to stream time; every
# update would be closer to 50 percent, for a curve that looks the same.
LOSS_PROGRESS_EVERY = 10


def loss_progress(hypernet, target, trainer, task, record,
                  *, every=LOSS_PROGRESS_EVERY):
    """How far one update moved the loss on its own batch, as a fraction.

    Elsayed and Mahmood 2024. Take the batch the update was computed from, run
    it through the network again now that the update has landed, and compare.
    A fraction of 0.02 means the update cut this batch's loss by 2 percent.
    Falling progress at a fixed learning rate is the plainest sign of a model
    that has become hard to move.

    `record` is one `on_step` record from `UnCLe.learn`. Its `loss` field is
    the combined loss, task plus the weighted protection term, and the
    protection term cannot be recomputed here without the pre-task snapshot.
    So `task_loss` is the before value and cross entropy is the after value:
    the same quantity on both sides.

    Buffers are cloned before the forward pass. The network is still in
    training mode here, where batch normalisation normalises by this batch's
    own statistics and writes its running statistics back. Normalising by batch
    statistics is what we want, since that is how the before value was
    computed, but the write-back would corrupt the task's real buffers. The
    clone takes the write and is thrown away.

    Returns None on the steps that are not sampled, and on step 0, which has no
    update behind it. A step whose before value is exactly zero also returns
    None: there is no fraction of zero.
    """
    step = record.get("step", 0)
    if step == 0 or step % every or "images" not in record:
        return None
    before = record["task_loss"]
    if before == 0:
        return None
    buffers = {name: value.clone()
               for name, value in trainer.task_buffers[task].items()}
    scores = functional_call(target,
                             (hypernet.weights_for(task), buffers),
                             (record["images"].to(trainer.device),),
                             strict=True)
    after = float(cross_entropy(scores, record["labels"].to(trainer.device)))
    return {"step": step, "epoch": record["epoch"], "loss_before": before,
            "loss_after": after, "loss_progress": (before - after) / before}


def loss_progress_recorder(hypernet, target, trainer, task,
                           *, every=LOSS_PROGRESS_EVERY):
    """A list to read afterwards, and the `on_step` callback that fills it.

    Pass the callback as `on_step` to `UnCLe.learn`. The trainer already runs
    callbacks under `no_grad`, inside a forked random number generator, and
    puts module training modes back afterwards, so nothing here can reach the
    run being measured.

        rows, watch = loss_progress_recorder(hypernet, target, trainer, task)
        trainer.learn(task, protected, on_step=watch)
    """
    rows = []

    def watch(record):
        row = loss_progress(hypernet, target, trainer, task, record, every=every)
        if row is not None:
            rows.append(row)

    return rows, watch
