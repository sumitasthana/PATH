"""The adapter plasticity study's probe, ported to this framework.

Plasticity means the ability to learn a new task. This module measures it the
way `reference/adapter-plasticity-study` does: take a copy of the model, train
the copy on a task it has never seen, record how well it learns, then throw the
copy away. Measuring never changes the run being measured.

Two endpoints, both from that study:

  E1, capacity to change   final training loss on randomly labelled images.
                           Random labels carry no pattern, so the only way to
                           fit them is to memorise. Plasticity loss shows as a
                           rise in this loss.
  E2, useful learning      test accuracy on a task the model has never seen.
                           Plasticity loss shows as a drop.

How the parts map. That study trains a small adapter on a frozen pretrained
backbone, plus a fresh linear head per probe task. Here the generator plays the
adapter's part, because it is the one network every task updates, and a fresh
32-number task code plays the head's part, because it is small, new, and
belongs to one task. So:

  full probe        generator and a fresh task code both train
  code-only probe   the generator is frozen, only the fresh code trains

The code-only probe is that study's head-only control. Read the warning on
`relative_contribution` before using it: it does not port cleanly, because that
study's control sits on pretrained features and this framework has none.

Three differences from that study, forced by this framework rather than chosen:

  * Probe tasks must have exactly as many classes as the stream's tasks. The
    generator's output layout is fixed when it is built, from a classifier with
    `classes_per_task` outputs. Of that study's three natural probe tasks only
    EuroSAT, at ten classes, could ever fit; DTD at 47 and Flowers102 at 102
    cannot be generated at all.
  * Probe tasks here come from the same dataset as the stream. That study uses
    unrelated tasks on purpose, so that transfer from similar earlier classes
    cannot be mistaken for plasticity. Probes drawn from the same dataset do
    not separate the two, and any result must say so.
  * The probe trains with no protection term. It is a discarded fork, so there
    is nothing to protect.
"""

import copy
import math

import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader, RandomSampler


def snapshot(hypernet, trainer):
    """Copy everything a probe could change, so it can be put back exactly.

    The generator's weights, the chunk codes, every task code, and every task's
    running statistics. Tensors are cloned onto the CPU, so a snapshot does not
    hold graphics memory while a probe runs.
    """
    return {
        "generator": copy.deepcopy(hypernet.state_dict()),
        "task_buffers": {task: {name: value.detach().cpu().clone()
                                for name, value in buffers.items()}
                         for task, buffers in trainer.task_buffers.items()},
        "tasks": list(hypernet.task_codes),
    }


def restore(hypernet, trainer, state):
    """Put the model back exactly as `snapshot` found it.

    Any task code the probe added is removed, so a probe leaves no trace. The
    generator's own weights are loaded back rather than merely detached, which
    is what makes repeated probes from one checkpoint comparable.
    """
    for task in list(hypernet.task_codes):
        if task not in state["tasks"]:
            del hypernet.task_codes[task]
    hypernet.load_state_dict(state["generator"])
    trainer.task_buffers.clear()
    for task, buffers in state["task_buffers"].items():
        trainer.task_buffers[task] = {name: value.clone().to(trainer.device)
                                      for name, value in buffers.items()}


def unchanged(hypernet, state):
    """True when the generator is bit for bit what the snapshot holds.

    A probe is only a measurement if the model it measured is still there
    afterwards. Call this after `restore` rather than trusting it.
    """
    current = hypernet.state_dict()
    if set(current) != set(state["generator"]):
        return False
    return all(torch.equal(current[name].detach().cpu(), state["generator"][name].cpu())
               for name in current)


def effective_rank(features):
    """How many directions the features really use. Roy and Vetterli, 2007.

    A feature matrix can have 2048 columns and still carry almost all of its
    variation in a handful of directions. Effective rank says how many
    directions are doing real work, as a number that need not be whole.

    It is the exponential of the entropy of the normalised singular values:

        p_i = s_i / sum(s)        effective rank = exp(-sum p_i log p_i)

    where s_i are the singular values of the feature matrix. A matrix whose
    singular values are all equal has an effective rank equal to its number of
    columns. One with a single non-zero singular value has effective rank 1.

    Falling effective rank during continual training is a known plasticity
    warning sign, which is why that study records it.
    """
    values = torch.linalg.svdvals(features.double())
    total = values.sum()
    if total <= 0:
        return float("nan")
    weights = values / total
    weights = weights[weights > 0]
    return float(torch.exp(-(weights * weights.log()).sum()).item())


def _features_and_logits(hypernet, target, trainer, task, images):
    """Run one batch, returning the classifier's input as well as its output.

    The features are what the network has built by the time it reaches its last
    layer. They are captured with a hook on that layer rather than by editing
    the network, so the generated weights still load with strict=True.
    """
    weights = hypernet.weights_for(task)
    buffers = {name: value.clone() for name, value in trainer.task_buffers[task].items()}
    captured = {}

    def hook(_module, inputs, _output):
        captured["features"] = inputs[0].detach()

    from .hypernet import classifier_name
    name = classifier_name(target)
    handle = dict(target.named_modules())[name].register_forward_hook(hook)
    try:
        logits = functional_call(target, (weights, buffers), (images,), strict=True)
    finally:
        handle.remove()
    return captured["features"], logits


@torch.no_grad()
def _evaluate(hypernet, target, trainer, task, dataset, batch_size, feature_images=0):
    """Accuracy and mean loss on a dataset, plus effective rank if asked.

    Always runs in evaluation mode. A BatchNorm layer in training mode rescales
    using whatever batch is in front of it, which revives a nearly dead signal
    and hides the effect being measured.
    """
    modes = hypernet.training, target.training
    hypernet.eval()
    target.eval()
    correct = total = 0
    loss_sum = 0.0
    collected, collected_rows = [], 0
    try:
        for images, labels in DataLoader(dataset, batch_size=batch_size):
            images, labels = images.to(trainer.device), labels.to(trainer.device)
            features, logits = _features_and_logits(hypernet, target, trainer, task, images)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
            if collected_rows < feature_images:
                collected.append(features[:feature_images - collected_rows].cpu())
                collected_rows += collected[-1].shape[0]
    finally:
        hypernet.train(modes[0])
        target.train(modes[1])
    result = {"accuracy": 100.0 * correct / total, "loss": loss_sum / total, "images": total}
    if feature_images:
        result["effective_rank"] = effective_rank(torch.cat(collected))
        result["feature_images"] = collected_rows
    return result


def probe_once(hypernet, target, trainer, *, task, train_set, test_set=None,
               steps, learning_rate, batch_size=64, code_only=False,
               seed=12345, feature_images=0, checkpoints=4):
    """Train a discarded copy on one task and report how well it learned.

    task            a task name the model has never learned. A fresh code is
                    created for it and removed again at the end.
    train_set       images the probe trains on
    test_set        images it is scored on. Omit for the random-label probe,
                    where a test score has no meaning.
    steps           how many updates. Fixed, and never varied between probe
                    points, or the comparison measures the budget instead.
    code_only       freeze the generator, so only the 32-number code trains.
                    This is the control that separates the generator's own
                    contribution from the code's.
    seed            fixes both the batch order and the fresh code's starting
                    values. It must depend on the probe task and the
                    repetition, and never on the probe point, so that two
                    points differ only by the model's state. Repetitions have
                    to differ in the starting code as well as the batch order,
                    or the spread they measure is too small and the noise floor
                    built from it lets anything through.
    feature_images  how many images to use for the effective rank. 0 skips it.
    checkpoints     how many evenly spaced readings to take during training,
                    including the last. The study records 4.

    Returns the final training loss, the test accuracy if a test set was given,
    the readings along the way, and the effective rank.

    The model is restored before returning, including on failure.
    """
    if task in hypernet.task_codes:
        raise ValueError(f"Task {task} is already learned. A probe needs an unseen task.")
    if steps < checkpoints:
        raise ValueError("steps must be at least the number of checkpoints")

    state = snapshot(hypernet, trainer)
    try:
        code = hypernet.add_task(task)
        # add_task draws the code from a stream keyed by the run's seed and the
        # task name, so every repetition would otherwise start from the same
        # code and differ only in batch order. Redraw it from the probe seed,
        # which carries the repetition. This is the study's requirement that
        # head initialisation depend on the probe task and the repetition.
        with torch.no_grad():
            stream = torch.Generator(device=code.device)
            stream.manual_seed(seed)
            code.copy_(torch.randn(code.shape, generator=stream,
                                   device=code.device, dtype=code.dtype))
        trainer.task_buffers[task] = {name: value.clone()
                                      for name, value in trainer.buffer_template.items()}
        trainable = [code] if code_only else [*hypernet.generator_parameters(), code]
        hypernet.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(trainable, lr=learning_rate)

        # Sampling with replacement for a fixed number of steps, as the study
        # does, so the budget is the same whatever the probe set's size.
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            train_set, batch_size=batch_size,
            sampler=RandomSampler(train_set, replacement=True,
                                  num_samples=steps * batch_size, generator=generator))

        hypernet.train()
        target.train()
        marks = {round(steps * (i + 1) / checkpoints) for i in range(checkpoints)}
        readings, losses = [], []
        step = 0
        for images, labels in loader:
            step += 1
            images, labels = images.to(trainer.device), labels.to(trainer.device)
            weights = hypernet.weights_for(task)
            buffers = trainer.task_buffers[task]
            logits = functional_call(target, (weights, buffers), (images,), strict=True)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite probe loss at step {step}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step in marks:
                reading = {"step": step,
                           "train_loss": sum(losses[-50:]) / len(losses[-50:])}
                if test_set is not None:
                    scored = _evaluate(hypernet, target, trainer, task, test_set, 256)
                    reading["test_accuracy"] = scored["accuracy"]
                    reading["test_loss"] = scored["loss"]
                readings.append(reading)
            if step >= steps:
                break

        # The final training loss is measured over the whole probe set in
        # evaluation mode, not read off the last mini-batch, which is noisy.
        final_train = _evaluate(hypernet, target, trainer, task, train_set, 256,
                                feature_images=feature_images)
        result = {
            "task": task, "steps": steps, "learning_rate": learning_rate,
            "code_only": code_only, "seed": seed,
            "final_train_loss": final_train["loss"],
            "final_train_accuracy": final_train["accuracy"],
            "readings": readings,
        }
        if feature_images:
            result["effective_rank"] = final_train["effective_rank"]
            result["effective_rank_images"] = final_train["feature_images"]
        if test_set is not None:
            scored = _evaluate(hypernet, target, trainer, task, test_set, 256)
            result["test_accuracy"] = scored["accuracy"]
            result["test_loss"] = scored["loss"]
        return result
    finally:
        restore(hypernet, trainer, state)


def best_of_grid(runs, *, endpoint):
    """Pick one run from a learning-rate grid, the way the study picks.

    endpoint "E2"  highest test accuracy, ties broken by lower test loss
    endpoint "E1"  lowest final training loss

    A model late in a stream may need a different rate than a fresh one. Using
    a single fixed rate would confuse plasticity loss with a rate mismatch,
    which is why the probe runs a grid and keeps the best.
    """
    if not runs:
        raise ValueError("Need at least one run.")
    if endpoint == "E2":
        return min(runs, key=lambda r: (-r["test_accuracy"], r["test_loss"]))
    if endpoint == "E1":
        return min(runs, key=lambda r: r["final_train_loss"])
    raise ValueError("endpoint must be E1 or E2")


def noise_floor(repetitions_at_zero, repetitions_at_t, count_at_zero, count_at_t):
    """The smallest change worth believing, as the study defines it.

        d = 1.96 * s * sqrt(1/n_t + 1/n_0)

    s is the standard deviation of single repetitions at the first probe point,
    n_0 and n_t are how many repetitions were taken at each point. A measured
    change smaller than this is inside the measurement's own wobble.

    repetitions_at_zero and repetitions_at_t are the spreads; pass the same
    value twice when only the first point was repeated.
    """
    del repetitions_at_t
    if count_at_zero < 2:
        raise ValueError("The noise floor needs at least two repetitions at the first point.")
    return 1.96 * repetitions_at_zero * math.sqrt(1 / count_at_t + 1 / count_at_zero)


def relative_contribution(full, control, full_at_zero, control_at_zero, *, endpoint):
    """R(t): how much of the generator's own contribution is left.

        G_t = full_t - control_t   (E2, accuracy)
        G_t = control_t - full_t   (E1, loss)
        R_t = G_t / G_0

    1 means the generator helps as much as it did at the start, 0 means it adds
    nothing beyond training a fresh code.

    WARNING. This does not port cleanly from the adapter study and should be
    read with care, or left out. There the control is a linear probe on frozen
    pretrained features, which scores well, so G_0 is a solid number. Here the
    control at t = 0 is a frozen random generator with a 32-number code, which
    sits near chance. G_0 is then close to zero, the ratio is dividing by almost
    nothing, and R(t) comes out small whatever the generator's real state. Use
    a later point as the baseline, or report G_t on its own and say why.
    """
    if endpoint == "E2":
        now, start = full - control, full_at_zero - control_at_zero
    elif endpoint == "E1":
        now, start = control - full, control_at_zero - full_at_zero
    else:
        raise ValueError("endpoint must be E1 or E2")
    if abs(start) < 1e-9:
        raise ValueError("The baseline contribution is zero, so R(t) has no meaning here.")
    return now / start


class RandomLabels(torch.utils.data.Dataset):
    """Wrap a dataset and give every image a fixed random label.

    Random labels carry no pattern, so fitting them measures raw capacity to
    change with nothing to transfer. This is the standard trainability test the
    study cites.

    The labels are drawn once from a seeded generator and stored, so every
    probe point sees exactly the same wrong labels. A fresh draw per point
    would make the points incomparable.
    """

    def __init__(self, base, classes, seed):
        self.base = base
        self.classes = classes
        self.labels = torch.randint(0, classes, (len(base),),
                                    generator=torch.Generator().manual_seed(seed))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, _ = self.base[index]
        return image, int(self.labels[index])
