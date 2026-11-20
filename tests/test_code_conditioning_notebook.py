"""Checks for notebook 54, which asks whether the generator needs the task code.

Two things in that notebook can be wrong without anything raising an error, and
a silent wrong answer here would go straight into the paper.

The first is `starting_code`. The `initial` arm scores a group through the code
it began with, rebuilt from the run seed rather than read from a file, because
nothing ever stored it. If that rebuild does not match what `add_task` drew at
the time, the arm measures a random code that the group never had, and the
notebook still prints a tidy table.

The second is `fresh_buffers`. Every arm re-estimates its own BatchNorm
statistics, and it does that by relying on BatchNorm writing its running mean
and variance back into the tensors passed to `functional_call`. If that write
did not land, every arm would be scored through the neutral template instead,
which would look like a result rather than a bug.

Both functions are taken out of the notebook as source, so a change there that
is not reflected here fails the test.
"""

from pathlib import Path
import json
import sys
import unittest

import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.config import Config, random_stream
from uncle.hypernet import HyperNetwork, build_target

NOTEBOOK = (Path(__file__).resolve().parents[1]
            / "notebooks/54_code_conditioning.ipynb")


def function_from_notebook(name, namespace):
    """Run the notebook cell that defines `name`, in the namespace given.

    The cell is executed whole, so the other functions it defines come along.
    They refer to names this test does not supply, which is fine: those are
    looked up when a function is called, and nothing calls them here.
    """
    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in book["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and f"def {name}(" in source:
            exec(compile(source, f"nb54:{name}", "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"notebook 54 has no cell defining {name}")


def small_model():
    """A tiny CPU model with the same structure as the study's, built quickly."""
    config = Config(dataset="permuted_mnist", backbone="cnn", classes_per_task=4,
                    chunks=4, hidden=(8,), code_dim=4, device="cpu", seed=3,
                    batch_size=8, epochs=1, requests=(("learn", "0"),),
                    tasks=tuple(str(i) for i in range(10)))
    target = build_target(config)
    return config, HyperNetwork(target, config)


class StartingCodeTests(unittest.TestCase):
    """The `initial` arm has to use the code the group really started from."""

    def setUp(self):
        self.starting_code = function_from_notebook(
            "starting_code", {"torch": torch, "random_stream": random_stream})

    def test_it_matches_what_add_task_drew(self):
        config, hypernet = small_model()
        drawn = hypernet.add_task("2").detach().clone()
        self.assertTrue(torch.equal(self.starting_code(config, "2"), drawn))

    def test_a_different_group_gets_a_different_code(self):
        # The rebuild is keyed by the group's name as well as the seed. If it
        # were not, every group's `initial` arm would use the same code and the
        # arm would be comparing groups rather than codes.
        config, _ = small_model()
        self.assertFalse(torch.equal(self.starting_code(config, "2"),
                                     self.starting_code(config, "3")))

    def test_a_different_seed_gets_a_different_code(self):
        config, _ = small_model()
        other = Config(**{**config.__dict__, "seed": config.seed + 1})
        self.assertFalse(torch.equal(self.starting_code(config, "2"),
                                     self.starting_code(other, "2")))


class StubTrainer:
    """The two fields `fresh_buffers` reads, and nothing else."""

    def __init__(self, buffer_template, device):
        self.buffer_template = buffer_template
        self.device = device


class FreshBuffersTests(unittest.TestCase):
    """Re-estimated statistics have to be the arm's own, not the template's.

    The study's backbone is a ResNet, which is too large to build here, and the
    small CPU backbone has no BatchNorm at all, so it could not show the write
    happening. This uses a two-layer network with one BatchNorm instead. What
    is being checked is the mechanism, which is the same either way: whether a
    forward pass in training mode writes its running statistics into the
    tensors handed to `functional_call`.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.target = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1),
                                    nn.BatchNorm2d(4), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(4 * 8 * 8, 4))
        self.weights = {name: value.detach().clone()
                        for name, value in self.target.named_parameters()}
        template = {name: value.detach().clone()
                    for name, value in self.target.named_buffers()}
        self.template = template
        self.trainer = StubTrainer(template, torch.device("cpu"))
        # Pictures with an obvious offset, so a running mean that was updated
        # cannot be confused with one that was left at zero.
        images = torch.randn(32, 1, 8, 8) + 5.0
        self.tasks = {"0": {"train": TensorDataset(images,
                                                   torch.randint(0, 4, (32,)))}}
        self.fresh_buffers = function_from_notebook("fresh_buffers", {
            "torch": torch, "functional_call": functional_call,
            "DataLoader": DataLoader, "tasks": self.tasks,
            "BATCH_SIZE": 8, "BUFFER_BATCHES": 4,
        })

    def test_the_statistics_are_updated(self):
        buffers = self.fresh_buffers(self.target, self.trainer, "0", self.weights)
        moved = [name for name, value in buffers.items()
                 if "num_batches" not in name
                 and not torch.allclose(value, self.template[name])]
        self.assertTrue(moved, "no running statistic moved, so every arm would "
                               "be scored through the neutral template")
        self.assertTrue(all(torch.isfinite(value).all() for value in buffers.values()))

    def test_the_template_is_left_alone(self):
        # The template is shared by every arm and every group. If an arm wrote
        # into it, the next arm would start from the previous one's statistics
        # and the comparison would depend on the order the arms ran in.
        before = {name: value.clone() for name, value in self.template.items()}
        self.fresh_buffers(self.target, self.trainer, "0", self.weights)
        for name, value in self.template.items():
            self.assertTrue(torch.equal(value, before[name]),
                            f"fresh_buffers wrote into the shared template at {name}")

    def test_the_network_is_left_in_evaluation_mode_if_it_began_there(self):
        # The notebook scores with `measure.probe`, which expects to set the
        # mode itself. A measurement that leaves training mode on would make
        # BatchNorm normalise by the batch in front of it at scoring time.
        self.target.eval()
        self.fresh_buffers(self.target, self.trainer, "0", self.weights)
        self.assertFalse(self.target.training)


if __name__ == "__main__":
    unittest.main()
