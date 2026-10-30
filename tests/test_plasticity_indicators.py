"""Checks for the weight-change indicator.

Two things have to hold. The number has to read the model, so it must be zero
when nothing moved and positive when something did. And taking the measurement
must not change the model, because every boundary measurement happens in the
middle of a stream that has to continue unaffected.
"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.config import Config
from uncle.hypernet import HyperNetwork, build_target
from uncle.plasticity_indicators import flatten, reference_code, weight_change
from uncle.probe import snapshot, unchanged
from uncle.trainer import UnCLe


def small_model():
    """A tiny CPU model with the same structure as the study's, built quickly."""
    config = Config(dataset="permuted_mnist", backbone="cnn", classes_per_task=4,
                    chunks=4, hidden=(8,), code_dim=4, device="cpu", seed=0,
                    batch_size=8, epochs=1, requests=(("learn", "0"),),
                    tasks=tuple(str(i) for i in range(10)))
    target = build_target(config)
    return config, target, HyperNetwork(target, config)


def nudge(hypernet, size=0.01):
    """Move the generator by one plain optimizer step on a made-up loss."""
    parameters = hypernet.generator_parameters()
    optimizer = torch.optim.SGD(parameters, lr=size)
    loss = sum(parameter.square().sum() for parameter in parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


class ReferenceCode(unittest.TestCase):

    def test_same_every_call(self):
        """The reference code is the anchor, so it must never drift."""
        _, _, hypernet = small_model()
        first = reference_code(hypernet)
        torch.randn(1000)                      # burn some global randomness
        self.assertTrue(torch.equal(first, reference_code(hypernet)))

    def test_independent_of_the_run(self):
        """Two models that have done different things still share the code."""
        _, _, one = small_model()
        _, _, other = small_model()
        other.add_task("0")
        other.add_task("1")
        self.assertTrue(torch.equal(reference_code(one), reference_code(other)))

    def test_is_not_a_task_code(self):
        config, _, hypernet = small_model()
        code = reference_code(hypernet)
        self.assertEqual(code.shape, (config.code_dim,))
        hypernet.add_task("0")
        self.assertFalse(torch.equal(code, hypernet.task_codes["0"].detach()))


class WeightChange(unittest.TestCase):

    def test_zero_against_itself(self):
        """The t = 0 row has to read exactly zero, not nearly zero."""
        _, _, hypernet = small_model()
        start = flatten(hypernet, reference_code(hypernet))
        record = weight_change(start, start)
        self.assertEqual(record["delta_w_target"], 0.0)
        self.assertEqual(record["delta_w_generator"], 0.0)

    def test_zero_when_the_generator_did_not_move(self):
        """Measuring twice without training must not invent movement."""
        _, _, hypernet = small_model()
        code = reference_code(hypernet)
        start = flatten(hypernet, code)
        record = weight_change(flatten(hypernet, code), start)
        self.assertEqual(record["delta_w_target"], 0.0)
        self.assertEqual(record["delta_w_generator"], 0.0)

    def test_positive_after_one_step(self):
        """One optimizer step has to show up in both numbers."""
        _, _, hypernet = small_model()
        code = reference_code(hypernet)
        start = flatten(hypernet, code)
        nudge(hypernet)
        record = weight_change(flatten(hypernet, code), start)
        self.assertGreater(record["delta_w_generator"], 0.0)
        self.assertGreater(record["delta_w_target"], 0.0)

    def test_grows_with_the_step(self):
        """A bigger step must read as a bigger distance, or the number is noise."""
        _, _, small = small_model()
        _, _, large = small_model()
        code = reference_code(small)
        start_small, start_large = flatten(small, code), flatten(large, code)
        nudge(small, size=0.001)
        nudge(large, size=0.01)
        moved_small = weight_change(flatten(small, code), start_small)
        moved_large = weight_change(flatten(large, code), start_large)
        self.assertGreater(moved_large["delta_w_generator"],
                           moved_small["delta_w_generator"])

    def test_norms_are_positive(self):
        _, _, hypernet = small_model()
        record = weight_change(*(flatten(hypernet, reference_code(hypernet)),) * 2)
        self.assertGreater(record["norm_target"], 0.0)
        self.assertGreater(record["norm_generator"], 0.0)

    def test_task_code_movement_is_excluded(self):
        """Training a task code alone must not register as generator movement.

        This is why the reference code exists. Weights read through a task's own
        code would move here; weights read through the reference code must not.
        """
        _, _, hypernet = small_model()
        code = reference_code(hypernet)
        hypernet.add_task("0")
        start = flatten(hypernet, code)
        with torch.no_grad():
            hypernet.task_codes["0"].add_(1.0)
        record = weight_change(flatten(hypernet, code), start)
        self.assertEqual(record["delta_w_target"], 0.0)


class MeasuringChangesNothing(unittest.TestCase):

    def test_model_is_bit_identical_afterwards(self):
        config, target, hypernet = small_model()
        trainer = UnCLe(hypernet, config, target, {})
        hypernet.add_task("0")
        state = snapshot(hypernet, trainer)
        flatten(hypernet, reference_code(hypernet))
        self.assertTrue(unchanged(hypernet, state))

    def test_no_gradients_are_left_behind(self):
        """A measurement that leaves gradients would corrupt the next update."""
        _, _, hypernet = small_model()
        flatten(hypernet, reference_code(hypernet))
        for parameter in hypernet.generator_parameters():
            self.assertIsNone(parameter.grad)

    def test_result_holds_no_graph(self):
        """Keeping a baseline must not keep the graph that built it alive."""
        _, _, hypernet = small_model()
        start = flatten(hypernet, reference_code(hypernet))
        self.assertFalse(start["target"].requires_grad)
        self.assertFalse(start["generator"].requires_grad)


if __name__ == "__main__":
    unittest.main()
