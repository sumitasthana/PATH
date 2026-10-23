"""Two measurements that accuracy alone cannot give, in one place.

Accuracy cannot separate a model that guesses from a model that answers the
same class for every image. Both score about 10% on ten classes. The study
needs that separation, so it reports two further numbers.

  input sensitivity   how far any single output score travels across the test
                      set. A working model moves a lot, because the picture
                      decides the scores. A model that has stopped reading its
                      input barely moves at all.

  relative weight     how far the generated weights moved away from a saved
  change              reference, as a fraction of the reference's own size.

The two answer different questions and must both be read. The unlearning
result is exactly the case where they disagree: over 100 unlearning steps the
generated weights lose about 43% of their size, which is unremarkable, while
input sensitivity falls by a factor of several million.

`relative_change` keeps the definition used by the adapter plasticity study in
`reference/adapter-plasticity-study`, so the two projects report the same
quantity. That study writes it as

    d_W = || W_eff - W_0 ||_F / || W_0 ||_F

pooled over every matrix, with the sums taken before the division and the
arithmetic done in float64.

Before this module the probe below was copied into notebooks 21, 22, 23, 24
and 26, in five versions that had drifted apart. Import it instead.
"""

from torch.func import functional_call
from torch.utils.data import DataLoader
import torch


def probe(hypernet, target, trainer, dataset, *, code_task, data_task=None,
          classes=10, batch_size=256, weights=None, buffers=None):
    """Score one task's generated weights and say whether the model is alive.

    hypernet, target, trainer   the model being measured, as the study builds it
    dataset                     the evaluation split: a Dataset of (image, label)
    code_task                   whose task code generates the weights
    data_task                   whose images are used. Defaults to code_task.
                                They differ in the recovery probe, where a brand
                                new code is trained on an old task's images and
                                must still be judged on those images.
    classes                     how many classes the task has
    batch_size                  evaluation batch size. Affects speed only.
    weights                     use these generated weights instead of the ones the
                                task code produces now. For scoring a deliberately
                                changed model, such as weights rescaled after unlearning.
    buffers                     use these running statistics instead of the task's own.
                                `trainer.buffer_template` gives the neutral set a fresh
                                task would start from.

    Returns accuracy as a percentage, the per-class prediction counts, how many
    classes the model ever chooses, which class it chooses most, and the input
    sensitivity.

    Evaluation mode matters more than usual. A BatchNorm layer in training mode
    normalises using whatever batch is in front of it, which divides a nearly
    dead signal by its own very small spread and returns it to normal size.
    That hides exactly the effect being measured. This function switches to
    evaluation mode and puts the previous mode back afterwards, so a
    measurement taken in the middle of training does not disturb the training.
    """
    data_task = code_task if data_task is None else data_task
    was_training = hypernet.training, target.training
    hypernet.eval()
    target.eval()

    # Both default to what the model would use on its own. A caller passes them in to
    # score a model it has changed on purpose, such as weights multiplied back up after
    # unlearning, or the task's running statistics replaced with neutral ones. Scoring
    # those through this same function is what keeps the arms comparable.
    if weights is None:
        weights = hypernet.weights_for(code_task)
    # The task's own running statistics. Copied so evaluation cannot change them.
    source = trainer.task_buffers[code_task] if buffers is None else buffers
    buffers = {name: value.clone() for name, value in source.items()}

    picked, correct, total = [], 0, 0
    lowest = highest = None
    try:
        with torch.no_grad():
            for images, labels in DataLoader(dataset, batch_size=batch_size):
                scores = functional_call(target, (weights, buffers),
                                         (images.to(trainer.device),), strict=True)
                choice = scores.argmax(1)
                picked.append(choice.cpu())
                correct += (choice == labels.to(trainer.device)).sum().item()
                total += labels.numel()
                # Reducing over dim 0 reduces over images, not over classes.
                # Each of the ten score positions keeps its own running lowest
                # and highest value across the whole test set.
                low, high = scores.min(dim=0).values, scores.max(dim=0).values
                lowest = low if lowest is None else torch.minimum(lowest, low)
                highest = high if highest is None else torch.maximum(highest, high)
    finally:
        hypernet.train(was_training[0])
        target.train(was_training[1])

    if not total:
        raise ValueError("The evaluation dataset is empty.")

    counts = torch.bincount(torch.cat(picked), minlength=classes).tolist()
    return {
        "accuracy": 100.0 * correct / total,
        "images": total,
        "class_counts": counts,
        "distinct_classes": sum(1 for count in counts if count > 0),
        "chosen_class": max(range(classes), key=lambda index: counts[index]),
        # Taking the largest of the ten ranges is deliberately generous to the
        # model: one output that still moves is enough to count it as alive.
        "input_sensitivity": (highest - lowest).max().item(),
    }


def relative_change(current, reference, *, per_tensor=False):
    """How far the weights moved from a reference, relative to its own size.

    current, reference   two weight dictionaries with the same keys, as
                         `hypernet.weights_for(task)` returns. `reference` is
                         the earlier state, written W_0 in the study.
    per_tensor           also return the same ratio computed separately for
                         each tensor, which says where the change sits

    The sums are taken over every tensor before the division, so this is one
    ratio for the whole weight set and not an average of per-tensor ratios.
    Large tensors therefore count for more, which is intended: it is the total
    movement that is being measured.

    The arithmetic runs in float64. The squared sums run over millions of
    entries, and float32 loses accuracy on a sum that long.

    Returns 0.0 when the two are identical, which is how a notebook checks that
    a reference was captured correctly before anything changed it.
    """
    if set(current) != set(reference):
        raise ValueError("Weight dictionaries have different keys.")

    changed = total = 0.0
    parts = {}
    for name, before in reference.items():
        after = current[name]
        if after.shape != before.shape:
            raise ValueError(f"Shape changed for {name}: {before.shape} to {after.shape}")
        before = before.detach().double()
        delta = after.detach().double() - before
        moved = delta.square().sum().item()
        size = before.square().sum().item()
        changed += moved
        total += size
        if per_tensor:
            parts[name] = (moved / size) ** 0.5 if size > 0 else float("nan")

    if total <= 0:
        raise ValueError("The reference weights are all zero, so a relative change has no meaning.")

    pooled = (changed / total) ** 0.5
    return (pooled, parts) if per_tensor else pooled


def direction_change(current, reference, *, per_tensor=False):
    """How far the weights turned, ignoring how much they grew or shrank.

    current, reference   two weight dictionaries with the same keys, as
                         `hypernet.weights_for(task)` returns. `reference` is
                         the earlier state.
    per_tensor           also return the same number separately for each
                         tensor, which says where the turning happened

    This is the companion to `relative_change`. That one says how far the
    weights moved. This one says whether they still point the same way:

        1.0   the same direction. Only the size changed.
        0.0   unrelated directions.
       -1.0   the opposite direction.

    The two answer different questions, and unlearning is the case where they
    come apart. An unlearning step that only shrinks the weights gives a large
    relative change and a direction of 1.0. One that scrambles them gives a
    large relative change and a direction near 0. Size alone cannot tell those
    apart, and whether a rescaling can undo the unlearning depends on which
    one happened.

    Pooling works the way `relative_change` pools: every tensor is treated as
    part of one long vector, so the result is the cosine of the two full
    weight sets and not an average of per-tensor cosines. Large tensors
    therefore count for more.

    A tensor of all zeros points nowhere, so its per-tensor entry is NaN
    rather than an invented number. The pooled value raises instead, because a
    reference of all zeros means the comparison itself is meaningless.

    The arithmetic runs in float64. The sums run over millions of entries and
    float32 loses accuracy on a sum that long.
    """
    if set(current) != set(reference):
        raise ValueError("Weight dictionaries have different keys.")

    dot = reference_square = current_square = 0.0
    parts = {}
    for name, before in reference.items():
        after = current[name]
        if after.shape != before.shape:
            raise ValueError(f"Shape changed for {name}: {before.shape} to {after.shape}")
        before = before.detach().double().reshape(-1)
        after = after.detach().double().reshape(-1)
        pair = (before * after).sum().item()
        before_size = before.square().sum().item()
        after_size = after.square().sum().item()
        dot += pair
        reference_square += before_size
        current_square += after_size
        if per_tensor:
            scale = (before_size * after_size) ** 0.5
            parts[name] = pair / scale if scale > 0 else float("nan")

    if reference_square <= 0 or current_square <= 0:
        raise ValueError("One side is all zeros, so a direction comparison has no meaning.")

    pooled = dot / (reference_square * current_square) ** 0.5
    return (pooled, parts) if per_tensor else pooled


def weight_size(hypernet, task):
    """The size of one task's generated weights, before the per-layer scaling.

    This is the number the conclusion report tracks during unlearning, where it
    falls to about 57% of its starting value. It uses `raw_for`, the generator's
    output before scaling, because that is what the unlearning objective acts
    on.
    """
    with torch.no_grad():
        return hypernet.raw_for(task).norm().item()
