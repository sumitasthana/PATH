"""Every knob in one place, with the paper's values as the defaults."""

from dataclasses import dataclass, field
import hashlib
import math

import torch


def random_stream(seed: int, purpose: str, task: str, device) -> torch.Generator:
    """A generator whose draws depend only on the seed, the purpose and the task.

    The global generator is shared by everything, so a draw made for task 9
    depends on how many draws every earlier request happened to make. That is
    fine for one run and fatal for the paired references the study needs: a
    reference with task 3's requests removed would differ from the unlearned
    model everywhere, not just at task 3, and the recovery gap would include
    that difference.

    Each purpose and task therefore gets its own stream, addressed by name
    rather than by position in the run. Task 3's code is the same whether it
    was learned first or last, and its forget noise is the same whether two
    requests preceded it or twenty.
    """
    digest = hashlib.blake2b(
        f"{seed}:{purpose}:{task}".encode(), digest_size=8
    ).digest()
    generator = torch.Generator(device=device)
    generator.manual_seed(int.from_bytes(digest, "big") >> 1)
    return generator

# Appendix C, Table 4. The author runs three random request sequences per
# dataset. `L#n` means learn task n, `U#n` means unlearn task n.
PERMUTED_MNIST_SEQUENCES = {
    1: "L1 L0 U1 L5 L8 L9 L7 U0 L2 L3 L4 U8 U3 U5 L6",
    2: "L6 L7 L2 L1 L0 U1 L9 U7 U2 U0 L4 U4 L8 U6 L5",
    3: "L7 L1 L2 L8 L0 U1 L3 L6 U3 U2 L4 L5 U8 L9 U7",
}

FIVE_TASKS_SEQUENCES = {
    1: "L0 L1 U0 L2 L3 L4 U1",
    2: "L3 L4 L2 L0 L1 U3 U0",
    3: "L0 L2 U0 L4 L3 U2 U4",
}

# Tiny-ImageNet runs 30 requests over 20 tasks, twice as long as the rest.
TINY_IMAGENET_SEQUENCES = {
    1: "L3 L0 U3 L9 L5 L17 L1 L7 L14 L15 L19 U17 U7 L6 U15 U9 L12 L4 U5 U4 "
       "U6 U0 U1 U14 U12 L13 L18 L2 L11 L8",
    2: "L12 L13 L5 L8 L2 U8 L14 U13 U5 U2 L3 U3 L16 U12 L11 U16 L7 L15 L10 L19 "
       "L9 U14 U7 L18 L6 L1 L0 L4 U6 L17",
    3: "L2 L7 U2 L18 L12 U7 U18 L16 L0 U16 U0 L13 L4 U12 U13 L9 L19 U19 U4 L10 "
       "L14 L5 U5 U10 L11 L1 U1 L17 L6 L3",
}

# This study's three task orderings, which are NOT from the UnCLe paper. The
# old orderings were the learn-only prefix of TINY_IMAGENET_SEQUENCES[1] and
# addressed task IDs up to 17, which the ten-group split no longer has.
#
# Nine tasks are learned, IDs 0 through 8. Task 9 is never trained on: it is
# the held-out in-distribution probe (methodology 3.1).
#
# Tasks 1, 2, 3, 5, 6 and 7 hold the same position in all three orderings.
# Tasks 0, 4 and 8 rotate through positions 1, 5 and 9, so each of the three
# is learned first once, in the middle once, and last once. That rotation is
# what separates an effect of the order from an effect of one task being
# harder than another (methodology 3.4).
TINY_IMAGENET_ORDERS = {
    "order_A": "L0 L1 L2 L3 L4 L5 L6 L7 L8",
    "order_B": "L4 L1 L2 L3 L8 L5 L6 L7 L0",
    "order_C": "L8 L1 L2 L3 L0 L5 L6 L7 L4",
}

# The group the stream never sees. Its test images are the `probe` image set.
PROBE_TASK = "9"

# Per-dataset defaults from the paper: the request sequences, how many tasks
# the sequences address, beta (Appendix C: 1e-1 for Permuted MNIST and
# CIFAR-100, 1e-2 for Tiny-ImageNet, 1e-3 for 5-Tasks), and the backbone
# ("ResNet18 in the case of Permuted MNIST experiments and ResNet50 elsewhere
# to demonstrate scalability"). The backbone belongs here rather than in the
# dataclass default: leaving it to the global default silently gave Tiny
# ImageNet a ResNet18, which is not the paper's setting and says nothing when
# it happens.


DATASETS = {
    "permuted_mnist": {
        "channels": 1, "size": 28, "classes": 10, "task_count": 10, "beta": 0.1,
        "classes_per_task": 10,
        "backbone": "resnet18", "sequences": PERMUTED_MNIST_SEQUENCES,
    },
    "tiny_imagenet": {
        # The paper rows: 20 tasks of 10 classes, the old saved split. This
        # study's 10 tasks of 20 classes come from `study_defaults`, so the
        # two counts here stay the pair that multiplies to 200.
        "channels": 3, "size": 64, "classes": 200, "task_count": 20, "beta": 0.01,
        "classes_per_task": 10,
        "backbone": "resnet50", "sequences": TINY_IMAGENET_SEQUENCES,
    },
}


def parse_sequence(text: str) -> tuple[tuple[str, str], ...]:
    """Turn "L1 L0 U1" into (("learn", "1"), ("learn", "0"), ("forget", "1"))."""
    actions = {"L": "learn", "U": "forget"}
    return tuple(
        (actions[token[0]], token[1:]) for token in text.split()
    )


def dataset_defaults(dataset: str, sequence: int) -> dict:
    """The paper's task list, request sequence, beta and backbone for a dataset.

    Sequence numbers are the rows of Table 4. The task list is sized to the
    sequence: Permuted MNIST names tasks 0-9, Tiny-ImageNet names 0-19. Those
    20 Tiny ImageNet tasks are the old 20 x 10 split in
    uncle/task_partition.json, which the paper rows still need, so this returns
    `classes_per_task` of 10 and a run built from it reads that file. This
    study's nine-task orderings come from `study_defaults` instead.
    Everything here is a value the paper states, so a caller that overrides one
    is departing from the paper on purpose.
    """
    settings = DATASETS[dataset]
    return {
        "dataset": dataset,
        "tasks": tuple(str(index) for index in range(settings["task_count"])),
        "requests": parse_sequence(settings["sequences"][sequence]),
        "beta": settings["beta"],
        "backbone": settings["backbone"],
        "classes_per_task": settings["classes_per_task"],
    }


def study_defaults(order: str) -> dict:
    """This study's settings for one of the three orderings.

    The same shape `dataset_defaults` returns, so a run builds its Config the
    same way either side of the change. `tasks` holds all ten groups, not just
    the nine learned ones, so the held-out probe group stays addressable when a
    run builds its image sets. Nothing requests it, so nothing trains on it.
    """
    return {
        "dataset": "tiny_imagenet",
        "tasks": tuple(str(index) for index in range(10)),
        "requests": parse_sequence(TINY_IMAGENET_ORDERS[order]),
        "beta": DATASETS["tiny_imagenet"]["beta"],
        "backbone": DATASETS["tiny_imagenet"]["backbone"],
        # Not the DATASETS value, which is the paper rows' ten. Twenty is what
        # picks the ten-group split this study is built on.
        "classes_per_task": 20,
    }


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Config:
    """Settings for one continual learn and unlearn experiment.

    The defaults are the paper's Permuted-MNIST setting: ResNet18 generated in
    200 chunks, 10 tasks, and request sequence 1 from Table 4. That needs a
    GPU. For a quick CPU run pass backbone="cnn" with fewer chunks and tasks.
    """

    dataset: str = "permuted_mnist"   # see DATASETS
    tasks: tuple[str, ...] = tuple(str(index) for index in range(10))
    requests: tuple[tuple[str, str], ...] = parse_sequence(
        PERMUTED_MNIST_SEQUENCES[1]
    )

    # How many classes one task has. It is also what picks the Tiny ImageNet
    # class grouping to read: twenty gives uncle/task_partition_10x20.json, the
    # 200 WordNet IDs shuffled with seed 42 into ten disjoint groups of twenty,
    # which is this study's split; ten gives the old uncle/task_partition.json
    # of 20 groups of ten, which the paper rows were built on. Either way every
    # class is used and nothing is dropped. The value also sets chance accuracy,
    # 5 percent for twenty classes and 10 percent for ten.
    #
    # The default matches the default dataset, Permuted MNIST, which always has
    # ten classes. `dataset_defaults` and `study_defaults` override it.
    classes_per_task: int = 10

    backbone: str = "resnet18"        # "resnet18", "resnet50", or "cnn"
    seed: int = 0
    initialization: str = "legacy"   # old checkpoints keep their output scales
    epochs: int = 5
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 1e-3       # Adam, the paper's value
    forgetting_learning_rate: float | None = None  # None follows learning_rate

    beta: float = 0.1     # hold-other-tasks-still term, eq. 2. Paper: 0.1 for PMNIST
    gamma: float = 0.01   # push-toward-noise term, eq. 3. Paper: 0.01 for PMNIST

    code_dim: int = 32               # Appendix B: task and chunk codes are both 32
    chunks: int = 200                # Appendix B: 200 chunks per task network
    hidden: tuple[int, ...] = (128, 256, 512)
    noise_samples: int = 10          # n in eq. 3

    # Appendix C: burn-in starts at 100 and drops 10% after each unlearn,
    # never below 20.
    burn_in: int = 100
    burn_in_decay: float = 0.9
    burn_in_min: int = 20

    data_root: str = "./data"
    device: str = field(default_factory=_default_device)

    def __post_init__(self) -> None:
        if self.initialization not in ("legacy", "hyperfan_in"):
            raise ValueError("initialization must be legacy or hyperfan_in")
        if self.forgetting_learning_rate is not None and (
            not math.isfinite(self.forgetting_learning_rate)
            or self.forgetting_learning_rate <= 0
        ):
            raise ValueError("forgetting_learning_rate must be finite and positive.")
        # The request list is the part a user edits most, so check it properly.
        learned: set[str] = set()
        forgotten: set[str] = set()

        for action, task in self.requests:
            if action not in ("learn", "forget"):
                raise ValueError(f"Unknown action: {action}")
            if task not in self.tasks:
                raise ValueError(f"Request names an unknown task: {task}")

            if action == "learn":
                if task in learned:
                    raise ValueError(f"Task {task} is learned twice.")
                learned.add(task)
            else:
                if task not in learned:
                    raise ValueError(f"Task {task} is forgotten before it is learned.")
                if task in forgotten:
                    raise ValueError(f"Task {task} is forgotten twice.")
                forgotten.add(task)

        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        # A task head has one output per class, so asking for more classes than
        # the dataset has trains a head on labels that can never appear and
        # reports chance at the wrong value, with nothing failing. A bare
        # Config() inheriting Tiny ImageNet's twenty on ten-class MNIST is the
        # case this catches. Fewer than the dataset has stays allowed: a short
        # CPU check uses a handful of classes on purpose.
        available = DATASETS[self.dataset]["classes"]
        if self.classes_per_task < 2:
            raise ValueError("classes_per_task must be at least 2.")
        if self.classes_per_task > available:
            raise ValueError(
                f"{self.dataset} has {available} classes, so a task cannot "
                f"have {self.classes_per_task}."
            )
        if self.backbone not in ("resnet18", "resnet50", "cnn"):
            raise ValueError(f"Unknown backbone: {self.backbone}")
        if self.chunks < 1:
            raise ValueError("chunks must be at least 1.")
        if not self.hidden:
            raise ValueError("hidden must name at least one layer width.")
        if not 0 < self.burn_in_decay <= 1:
            raise ValueError("burn_in_decay must be in (0, 1].")

    @property
    def input_channels(self) -> int:
        return DATASETS[self.dataset]["channels"]

    @property
    def input_size(self) -> int:
        return DATASETS[self.dataset]["size"]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def burn_in_for(self, completed_forgets: int) -> int:
        """Appendix C's annealed burn-in for the next unlearn request."""
        annealed = int(self.burn_in * self.burn_in_decay ** completed_forgets)
        return max(self.burn_in_min, annealed)
