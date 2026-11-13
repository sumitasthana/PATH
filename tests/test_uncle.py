"""Checks that run in seconds and never touch MNIST.

    python tests/test_uncle.py

Uses a tiny synthetic dataset: each class is a bright square in its own spot.
Trivially learnable, which is the point. These check the machinery, not accuracy.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uncle import Config, HyperNetwork, UnCLe, build_target  # noqa: E402
from uncle.metrics import relapse, spill, summary  # noqa: E402


class Blocks(Dataset):
    """Class k is a bright 5x5 square at grid position k."""

    def __init__(self, count: int, shift: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 10, (count,), generator=generator)
        self.images = torch.zeros(count, 1, 28, 28)

        for index, label in enumerate(self.labels):
            row = (int(label) // 5) * 10 + shift
            column = (int(label) % 5) * 5
            self.images[index, 0, row:row + 5, column:column + 5] = 1.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], int(self.labels[index])


def build(**overrides):
    config = Config(
        tasks=("A", "B"),
        requests=(("learn", "A"), ("learn", "B"), ("forget", "A")),
        backbone="cnn", chunks=32, classes_per_task=10,
        epochs=25, batch_size=64, burn_in=150, device="cpu",
        **overrides,
    )
    torch.manual_seed(config.seed)

    tasks = {
        "A": {"train": Blocks(256, 0, 1), "test": Blocks(128, 0, 2)},
        "B": {"train": Blocks(256, 3, 3), "test": Blocks(128, 3, 4)},
    }
    target = build_target(config)
    hypernet = HyperNetwork(target, config)

    return UnCLe(hypernet, config, target, tasks), hypernet, target


def test_config_rejects_bad_requests():
    for bad in (
        (("forget", "A"),),                    # forgotten before learned
        (("learn", "A"), ("learn", "A")),      # learned twice
        (("learn", "Z"),),                     # unknown task
        (("rename", "A"),),                    # unknown action
    ):
        try:
            Config(tasks=("A", "B"), requests=bad)
        except ValueError:
            continue
        raise AssertionError(f"Config accepted a bad request list: {bad}")


def test_generated_weights_fit_the_target():
    _, hypernet, target = build()
    hypernet.add_task("A")
    weights = hypernet.weights_for("A")

    expected = dict(target.named_parameters())
    assert set(weights) == set(expected), "generated names do not match the target"

    for name, value in weights.items():
        assert value.shape == expected[name].shape, f"{name} has the wrong shape"
        assert torch.isfinite(value).all(), f"{name} is not finite"


def test_heads_split_by_parameter_type():
    """Appendix B: one head for weights, one for BatchNorm, one for residuals."""
    config = Config(
        tasks=("A",), requests=(("learn", "A"),),
        backbone="cnn", chunks=16, device="cpu",
    )

    resnet_shaped = nn.Sequential()
    resnet_shaped.add_module("conv", nn.Conv2d(1, 4, 3, padding=1, bias=False))
    resnet_shaped.add_module("bn", nn.BatchNorm2d(4))
    resnet_shaped.add_module("downsample", nn.Conv2d(4, 8, 1, bias=False))
    resnet_shaped.add_module("head", nn.Linear(8 * 28 * 28, 10))

    hypernet = HyperNetwork(resnet_shaped, config)

    assert set(hypernet.heads) == {"weights", "batchnorm", "residual"}
    assert hypernet.layout["bn.weight"][0] == "batchnorm"
    assert hypernet.layout["downsample.weight"][0] == "residual"
    assert hypernet.layout["conv.weight"][0] == "weights"

    # Every head gets at least one chunk and the budget is spent exactly.
    assert sum(hypernet.chunk_counts.values()) == config.chunks
    assert all(count >= 1 for count in hypernet.chunk_counts.values())

    # Generation still reproduces every tensor of the target, exactly once.
    hypernet.add_task("A")
    weights = hypernet.weights_for("A")
    expected = dict(resnet_shaped.named_parameters())

    assert set(weights) == set(expected)
    for name, value in weights.items():
        assert value.shape == expected[name].shape, name

    assert hypernet.raw_for("A").numel() == sum(
        p.numel() for p in expected.values()
    )


def test_one_head_when_the_target_has_one_parameter_type():
    """Our plain CNN has no BatchNorm and no residuals, so one head is correct."""
    _, hypernet, _ = build()
    assert set(hypernet.heads) == {"weights"}


def test_snapshot_does_not_follow_the_live_network():
    uncle, hypernet, _ = build()
    hypernet.add_task("A")
    snapshot = hypernet.snapshot()

    code = hypernet.task_codes["A"].detach()
    before = snapshot.weights_from_code(code)["0.weight"].clone()

    with torch.no_grad():
        for parameter in hypernet.trunk.parameters():
            parameter.add_(0.1)

    after = snapshot.weights_from_code(code)["0.weight"]
    assert torch.equal(before, after), "the snapshot moved when the live network did"
    assert not any(p.requires_grad for p in snapshot.parameters())


def test_preserve_is_zero_with_nothing_to_protect():
    uncle, hypernet, _ = build()
    hypernet.add_task("A")
    assert uncle.preserve([], hypernet.snapshot()).item() == 0.0


def test_learn_then_forget():
    uncle, _, _ = build()

    uncle.learn("A", protected=[])
    learned_a = uncle.accuracy("A")
    assert learned_a > 70, f"task A did not learn, scored {learned_a:.1f}%"

    uncle.learn("B", protected=["A"])
    kept_a = uncle.accuracy("A")
    assert kept_a > 50, f"task A was wrecked by learning B, scored {kept_a:.1f}%"

    before_b = uncle.accuracy("B")
    uncle.forget("A", protected=["B"])

    forgotten_a = uncle.accuracy("A")
    assert forgotten_a < 30, f"task A was not forgotten, scored {forgotten_a:.1f}%"

    moved_b = abs(uncle.accuracy("B") - before_b)
    assert moved_b < 15, f"forgetting A spilled onto B by {moved_b:.1f} points"


def test_forget_needs_no_data():
    """The forget path must not read the task's dataset."""
    uncle, _, _ = build()
    uncle.learn("A", protected=[])

    uncle.tasks["A"]["train"] = None  # any read would raise
    uncle.forget("A", protected=[])


def test_metrics_match_the_paper_definitions():
    history = [
        {"index": 0, "action": "learn", "task": "A", "before": {},
         "after": {"A": 90.0}, "seen": ["A"], "forgotten": []},
        {"index": 1, "action": "forget", "task": "A",
         "before": {"A": 90.0, "B": 80.0}, "after": {"A": 10.0, "B": 78.0},
         "seen": ["A", "B"], "forgotten": ["A"]},
        {"index": 2, "action": "learn", "task": "C",
         "before": {"A": 10.0, "B": 78.0},
         "after": {"A": 14.0, "B": 77.0, "C": 85.0},
         "seen": ["A", "B", "C"], "forgotten": ["A"]},
    ]

    assert spill(history[1]) == 2.0            # only B moved, by 2 points
    assert spill(history[0]) is None           # learning has no spill
    assert relapse(history) == {"A": 4.0}      # 10 right after, 14 at the end

    numbers = summary(history)
    assert numbers["forget_accuracy"] == 14.0
    assert numbers["retain_accuracy"] == 81.0  # mean of B and C
    assert numbers["mean_spill"] == 2.0


def test_burn_in_anneals():
    """Appendix C: start at 100, drop 10% per unlearn, never below 20."""
    config = Config(
        tasks=("A",), requests=(("learn", "A"),), backbone="cnn", device="cpu",
    )
    assert config.burn_in_for(0) == 100
    assert config.burn_in_for(1) == 90
    assert config.burn_in_for(2) == 81
    assert config.burn_in_for(50) == config.burn_in_min


def test_sequences_parse_to_the_papers_requests():
    from uncle.config import PERMUTED_MNIST_SEQUENCES, parse_sequence

    requests = parse_sequence(PERMUTED_MNIST_SEQUENCES[1])
    assert len(requests) == 15
    assert requests[0] == ("learn", "1")
    assert requests[2] == ("forget", "1")
    assert sum(1 for action, _ in requests if action == "forget") == 5


def test_tiny_imagenet_sequences_match_the_paper():
    """Table 4's Tiny-ImageNet rows: 30 requests over 20 tasks, all three valid.

    Building a Config runs the request-list checks: nothing learned twice,
    nothing forgotten before it is learned or forgotten twice. A typo in the
    transcription almost certainly trips one of them.
    """
    from uncle.config import TINY_IMAGENET_SEQUENCES, dataset_defaults

    for number in (1, 2, 3):
        settings = dataset_defaults("tiny_imagenet", number)
        config = Config(**settings, device="cpu")

        assert len(config.requests) == 30
        assert len(config.tasks) == 20
        assert config.beta == 0.01       # Appendix C: 1e-2 for Tiny-ImageNet

        tokens = TINY_IMAGENET_SEQUENCES[number].split()
        assert max(int(token[1:]) for token in tokens) == 19

    # Spot-check the first few of each row against the printed table.
    assert dataset_defaults("tiny_imagenet", 1)["requests"][:3] == (
        ("learn", "3"), ("learn", "0"), ("forget", "3"),
    )
    assert dataset_defaults("tiny_imagenet", 2)["requests"][:3] == (
        ("learn", "12"), ("learn", "13"), ("learn", "5"),
    )
    assert dataset_defaults("tiny_imagenet", 3)["requests"][:3] == (
        ("learn", "2"), ("learn", "7"), ("forget", "2"),
    )


def test_study_orderings_rotate_three_tasks_and_hold_six_still():
    """Methodology 3.4, checked rather than trusted.

    Nine tasks are learned and task 9 is held out. Six tasks must sit in the
    same position in all three orderings, and the other three must rotate
    through positions 1, 5 and 9, each one first once, middle once and last
    once. Getting this wrong would not crash anything: it would quietly turn
    the order comparison into a comparison of three different task mixes.
    """
    from uncle.config import PROBE_TASK, TINY_IMAGENET_ORDERS, study_defaults

    orders = {}
    for name in TINY_IMAGENET_ORDERS:
        settings = study_defaults(name)
        config = Config(**settings, device="cpu")   # runs the request checks
        learned = [task for action, task in config.requests if action == "learn"]

        assert len(config.requests) == len(learned) == 9   # nothing unlearned
        assert sorted(learned, key=int) == [str(i) for i in range(9)]
        assert PROBE_TASK not in learned                   # never trained on
        assert PROBE_TASK in config.tasks                  # but still nameable
        orders[name] = learned

    positions = list(zip(*orders.values()))            # one tuple per position
    fixed = [index for index, column in enumerate(positions) if len(set(column)) == 1]
    moving = [index for index in range(9) if index not in fixed]

    assert len(fixed) == 6
    assert moving == [0, 4, 8]                         # positions 1, 5 and 9

    # Each rotating task takes each of the three positions exactly once.
    for index in moving:
        assert len(set(positions[index])) == 3
    rotating = {positions[index][0] for index in moving}
    assert len(rotating) == 3
    for task in rotating:
        assert sorted(order.index(task) for order in orders.values()) == moving


def test_resnet_target_keeps_buffers_per_task():
    """BatchNorm statistics are not generated, so each task holds its own."""
    config = Config(
        tasks=("A", "B"), requests=(("learn", "A"), ("learn", "B")),
        backbone="resnet18", chunks=200, epochs=1, batch_size=8, device="cpu",
    )
    target = build_target(config)

    assert any("running_mean" in name for name, _ in target.named_buffers())

    hypernet = HyperNetwork(config=config, target=target)
    assert set(hypernet.heads) == {"weights", "batchnorm", "residual"}

    uncle = UnCLe(hypernet, config, target, tasks=None)
    assert uncle.buffer_template, "the template should hold ResNet's buffers"

    # Generation has to cover every ResNet parameter, exactly once.
    hypernet.add_task("A")
    weights = hypernet.weights_for("A")
    expected = dict(target.named_parameters())
    assert set(weights) == set(expected)
    for name, value in weights.items():
        assert value.shape == expected[name].shape, name

    # One real forward pass, with buffers supplied the way learn() does it.
    uncle.task_buffers["A"] = {
        name: value.clone() for name, value in uncle.buffer_template.items()
    }
    before = uncle.task_buffers["A"]["bn1.running_mean"].clone()

    target.train()
    scores = functional_call(
        target, (weights, uncle.task_buffers["A"]),
        (torch.randn(8, 1, 28, 28),), strict=True,
    )

    assert scores.shape == (8, config.classes_per_task)
    assert torch.isfinite(scores).all()
    # Train mode must have moved this task's statistics.
    assert not torch.equal(before, uncle.task_buffers["A"]["bn1.running_mean"])


def test_dataset_defaults_are_the_papers_stated_values():
    """Every value the paper states for a dataset, checked in one place.

    The backbone used to be missing here, so Tiny ImageNet silently inherited
    the dataclass default of ResNet18 where the paper specifies ResNet50.
    """
    from uncle.config import dataset_defaults

    expected = {
        "permuted_mnist": {"beta": 0.1, "backbone": "resnet18",
                           "tasks": 10, "requests": 15},
        "tiny_imagenet": {"beta": 0.01, "backbone": "resnet50",
                          "tasks": 20, "requests": 30},
    }
    for dataset, want in expected.items():
        got = dataset_defaults(dataset, 1)
        assert got["beta"] == want["beta"], (dataset, got["beta"])
        assert got["backbone"] == want["backbone"], (dataset, got["backbone"])
        assert len(got["tasks"]) == want["tasks"], (dataset, len(got["tasks"]))
        assert len(got["requests"]) == want["requests"], dataset

    # And the values the paper states once, for every dataset.
    config = Config(**dataset_defaults("tiny_imagenet", 1))
    assert config.hidden == (128, 256, 512)
    assert config.code_dim == 32 and config.chunks == 200
    # The paper rows name 20 tasks, so they are the old 20 x 10 split and a
    # task has ten classes. Twenty tasks of twenty classes would be 400, which
    # Tiny ImageNet does not have. The ten groups of twenty are study_defaults.
    assert config.classes_per_task == 10
    assert config.learning_rate == 0.001
    assert config.gamma == 0.01 and config.noise_samples == 10
    assert (config.burn_in, config.burn_in_decay, config.burn_in_min) == (100, 0.9, 20)


def test_tiny_imagenet_tasks_come_from_the_saved_partition():
    """One loader, one grouping. The partition itself is tested in test_tasks.py."""
    from uncle.data import build_tasks
    from uncle.streams import build_tasks as build_from_partition

    # data.build_tasks is a router now, not a second implementation.
    assert build_tasks.__module__ == "uncle.data"
    assert build_from_partition.__module__ == "uncle.streams"
    assert not hasattr(sys.modules["uncle.data"], "_class_groups")
    assert not hasattr(sys.modules["uncle.data"], "_fetch_tiny_imagenet")


def test_the_class_count_picks_the_saved_split_it_belongs_to():
    """The partition file travels with `classes_per_task`, never apart from it.

    These were two halves of one fact for a while: a global default path said
    ten groups of twenty while the paper rows still asked for twenty tasks, so
    `main.py --dataset tiny_imagenet` died asking a ten-group file for task 19.
    Reading the path off the class count is what keeps both rows buildable.
    """
    from uncle.config import dataset_defaults, study_defaults
    from uncle.tasks import PARTITIONS, load_partition

    paper = Config(**dataset_defaults("tiny_imagenet", 1))
    study = Config(**study_defaults("order_A"))

    assert (paper.classes_per_task, len(paper.tasks)) == (10, 20)
    assert (study.classes_per_task, len(study.tasks)) == (20, 10)

    for config in (paper, study):
        # Every class used, none twice, and the file holds every task the
        # config names. This is the check that the failing run tripped over.
        partition = load_partition(path=PARTITIONS[config.classes_per_task])
        wnids = [w for group in partition["tasks"] for w in group["wnids"]]
        assert len(wnids) == len(set(wnids)) == 200
        assert len(partition["tasks"]) == len(config.tasks)
        assert len(partition["tasks"][0]["wnids"]) == config.classes_per_task


def test_a_task_cannot_have_more_classes_than_the_dataset():
    """The bare default used to give Permuted MNIST twenty classes out of ten.

    Nothing failed when it did: the head simply grew ten outputs that no label
    could ever reach, and chance was reported at 5 percent instead of 10.
    """
    assert Config().classes_per_task == 10        # matches Permuted MNIST

    with pytest.raises(ValueError):
        Config(classes_per_task=20)               # Permuted MNIST has ten

    # Fewer than the dataset holds is a deliberate speed knob, not a mistake.
    assert Config(classes_per_task=4).classes_per_task == 4


def test_task_codes_do_not_depend_on_request_order():
    """Task 3's starting code is the same whether it is learned first or third.

    This is what makes a paired reference honest. If the code came from the
    shared global generator, removing another task's requests would change
    task 3's starting point too, and the recovery gap would measure that
    instead of residual knowledge.
    """
    config = Config(backbone="cnn", chunks=4, hidden=(8,), code_dim=4,
                    tasks=("0", "1", "3"), requests=(("learn", "3"),))

    torch.manual_seed(0)
    first = HyperNetwork(build_target(config), config)
    first.add_task("3")

    torch.manual_seed(0)
    later = HyperNetwork(build_target(config), config)
    for name in ("0", "1", "3"):
        later.add_task(name)

    assert torch.equal(first.task_codes["3"], later.task_codes["3"])
    # Different tasks still get different codes.
    assert not torch.equal(later.task_codes["0"], later.task_codes["3"])


def test_the_preserve_term_has_no_gradient_at_the_first_forget_step():
    """Measured, not assumed: only the noise term pulls at step 1.

    `preserve` compares the model against a snapshot taken when the request
    started, so at the first step it is comparing the model with itself: zero
    loss and zero gradient. Whatever damage that first step does to the
    retained tasks, nothing in the objective is resisting it.
    """
    config = Config(backbone="cnn", chunks=4, hidden=(8,), code_dim=4,
                    tasks=("0", "1"), requests=(("learn", "0"),))
    torch.manual_seed(0)
    hypernet = HyperNetwork(build_target(config), config)
    uncle = UnCLe(hypernet, config, build_target(config))
    hypernet.add_task("0")
    hypernet.add_task("1")

    steps = []
    uncle.forget("0", ["1"], burn_in=2, on_step=steps.append, measure=True)

    first, second = steps[1], steps[2]
    assert _norm(first["preserve_gradient"]) == 0.0, first["preserve_gradient"]
    assert _norm(first["noise_gradient"]) > 0.0, first["noise_gradient"]
    # Once the first update has moved the model, the penalty starts pulling back.
    assert _norm(second["preserve_gradient"]) > 0.0, second["preserve_gradient"]
    # Both terms report every parameter group, so neither can hide in a total.
    assert set(first["noise_gradient"]) == {"trunk", "heads.weights"}


def _norm(norms):
    return sum(value ** 2 for value in norms.values()) ** 0.5


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")

    print(f"\n{len(tests)} checks passed.")


if __name__ == "__main__":
    main()
