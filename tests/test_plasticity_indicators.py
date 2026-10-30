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
from uncle.plasticity_indicators import (features, flatten, ranks, reference_code,
                                         srank, weight_change)
from uncle.probe import effective_rank, snapshot, unchanged
from uncle.trainer import UnCLe


def small_model():
    """A tiny CPU model with the same structure as the study's, built quickly."""
    config = Config(dataset="permuted_mnist", backbone="cnn", classes_per_task=4,
                    chunks=4, hidden=(8,), code_dim=4, device="cpu", seed=0,
                    batch_size=8, epochs=1, requests=(("learn", "0"),),
                    tasks=tuple(str(i) for i in range(10)))
    target = build_target(config)
    return config, target, HyperNetwork(target, config)


def ready(size=6):
    """A model, a trainer that can supply buffers, and some images to push through."""
    config, target, hypernet = small_model()
    trainer = UnCLe(hypernet, config, target, {})
    torch.manual_seed(0)
    images = torch.randn(size, config.input_channels,
                         config.input_size, config.input_size)
    return config, target, hypernet, trainer, images


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


class Srank(unittest.TestCase):

    def test_one_direction_reads_one(self):
        """A matrix built from a single direction uses a single direction."""
        column = torch.randn(40, 1)
        self.assertEqual(srank(column @ torch.randn(1, 12)), 1)

    def test_equal_directions_read_all_of_them(self):
        """When every direction matters equally, none can be dropped."""
        self.assertEqual(srank(torch.eye(10)), 10)

    def test_never_exceeds_the_columns(self):
        matrix = torch.randn(30, 7)
        self.assertLessEqual(srank(matrix), 7)

    def test_a_long_tail_is_ignored(self):
        """srank drops the tail that effective rank keeps. That is the point.

        Two strong directions plus eight faint ones. srank counts the strong
        ones; effective rank is pulled upward by the faint ones.
        """
        scale = torch.tensor([5.0, 5.0] + [0.01] * 8)
        matrix = torch.randn(60, 10) * scale
        self.assertEqual(srank(matrix), 2)
        self.assertGreater(effective_rank(matrix), srank(matrix))


class Features(unittest.TestCase):

    def test_shape_is_one_row_per_image(self):
        _, target, hypernet, trainer, images = ready(size=6)
        matrix = features(hypernet, target, trainer, images, reference_code(hypernet))
        self.assertEqual(matrix.shape[0], 6)
        self.assertGreater(matrix.shape[1], 0)

    def test_batching_does_not_change_the_answer(self):
        """A boundary set is larger than one batch, so the split must not matter."""
        _, target, hypernet, trainer, images = ready(size=10)
        code = reference_code(hypernet)
        whole = features(hypernet, target, trainer, images, code, batch_size=64)
        split = features(hypernet, target, trainer, images, code, batch_size=3)
        self.assertTrue(torch.allclose(whole, split, atol=1e-6))

    def test_same_code_gives_the_same_features(self):
        _, target, hypernet, trainer, images = ready()
        code = reference_code(hypernet)
        first = features(hypernet, target, trainer, images, code)
        second = features(hypernet, target, trainer, images, code)
        self.assertTrue(torch.equal(first, second))

    def test_moving_the_generator_moves_the_features(self):
        _, target, hypernet, trainer, images = ready()
        code = reference_code(hypernet)
        before = features(hypernet, target, trainer, images, code)
        nudge(hypernet, size=0.1)
        after = features(hypernet, target, trainer, images, code)
        self.assertFalse(torch.equal(before, after))

    def test_training_mode_is_restored(self):
        """The stream continues after the measurement and must find its modes."""
        _, target, hypernet, trainer, images = ready()
        hypernet.train()
        target.train()
        features(hypernet, target, trainer, images, reference_code(hypernet))
        self.assertTrue(hypernet.training)
        self.assertTrue(target.training)


class Ranks(unittest.TestCase):

    def test_one_entry_per_set_per_measure(self):
        _, target, hypernet, trainer, images = ready()
        record = ranks(hypernet, target, trainer,
                       {"in_stream": images, "held_out": images.flip(0)},
                       reference_code(hypernet))
        self.assertEqual(set(record), {
            "effective_rank_in_stream", "srank_in_stream", "images_in_stream",
            "effective_rank_held_out", "srank_held_out", "images_held_out"})
        self.assertEqual(record["images_in_stream"], len(images))

    def test_rank_cannot_exceed_the_images(self):
        """Six images cannot span more than six directions, whatever the width."""
        _, target, hypernet, trainer, images = ready(size=6)
        record = ranks(hypernet, target, trainer, {"set": images},
                       reference_code(hypernet))
        self.assertLessEqual(record["srank_set"], 6)
        self.assertLessEqual(record["effective_rank_set"], 6.0 + 1e-6)


class MeasuringChangesNothing(unittest.TestCase):

    def test_model_is_bit_identical_afterwards(self):
        config, target, hypernet = small_model()
        trainer = UnCLe(hypernet, config, target, {})
        hypernet.add_task("0")
        state = snapshot(hypernet, trainer)
        flatten(hypernet, reference_code(hypernet))
        self.assertTrue(unchanged(hypernet, state))

    def test_reading_features_changes_nothing(self):
        """Forward passes touch BatchNorm buffers unless they are held still."""
        config, target, hypernet, trainer, images = ready()
        hypernet.add_task("0")
        state = snapshot(hypernet, trainer)
        template = {name: value.clone() for name, value in trainer.buffer_template.items()}
        ranks(hypernet, target, trainer, {"set": images}, reference_code(hypernet))
        self.assertTrue(unchanged(hypernet, state))
        for name, value in template.items():
            self.assertTrue(torch.equal(value, trainer.buffer_template[name]), name)

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
