"""Checks for the fork-and-discard plasticity probe.

The probe's whole claim is that measuring does not change the model being
measured, so most of these check exactly that.
"""

from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.config import Config
from uncle.hypernet import HyperNetwork, build_target
from uncle.probe import (RandomLabels, best_of_grid, effective_rank, noise_floor,
                         probe_once, relative_contribution, restore, snapshot, unchanged)
from uncle.trainer import UnCLe


def small_model():
    """A tiny CPU model with the same structure as the study's, built quickly."""
    config = Config(dataset="permuted_mnist", backbone="cnn", classes_per_task=4,
                    chunks=4, hidden=(8,), code_dim=4, device="cpu", seed=0,
                    batch_size=8, epochs=1, requests=(("learn", "0"),),
                    tasks=tuple(str(i) for i in range(10)))
    target = build_target(config)
    hypernet = HyperNetwork(target, config)
    return config, target, hypernet


def fake_tasks(config, names, size=16):
    """Image tensors shaped like the real dataset, with labels in range."""
    torch.manual_seed(0)
    tasks = {}
    for name in names:
        images = torch.randn(size, config.input_channels,
                             config.input_size, config.input_size)
        labels = torch.randint(0, config.classes_per_task, (size,))
        tasks[name] = {"train": TensorDataset(images, labels),
                       "test": TensorDataset(images, labels)}
    return tasks


class EffectiveRankTests(unittest.TestCase):
    def test_equal_directions_give_the_column_count(self):
        # An identity matrix has equal singular values, so every direction
        # carries the same share and the effective rank is the column count.
        self.assertAlmostEqual(effective_rank(torch.eye(5)), 5.0, places=6)

    def test_one_direction_gives_one(self):
        # Every row is a multiple of the same vector, so there is one direction.
        features = torch.arange(1.0, 7.0).reshape(6, 1) @ torch.tensor([[1.0, 2.0, 3.0]])
        self.assertAlmostEqual(effective_rank(features), 1.0, places=6)

    def test_collapsed_features_rank_below_spread_features(self):
        torch.manual_seed(0)
        spread = torch.randn(40, 8)
        collapsed = spread.clone()
        collapsed[:, 1:] *= 1e-6
        self.assertLess(effective_rank(collapsed), effective_rank(spread))


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.config, self.target, self.hypernet = small_model()
        self.tasks = fake_tasks(self.config, ["0", "1"])
        self.trainer = UnCLe(self.hypernet, self.config, self.target, self.tasks,
                             progress=False)
        self.trainer.learn("0", [])

    def test_restore_undoes_a_change(self):
        state = snapshot(self.hypernet, self.trainer)
        with torch.no_grad():
            for parameter in self.hypernet.generator_parameters():
                parameter.add_(1.0)
        self.assertFalse(unchanged(self.hypernet, state))
        restore(self.hypernet, self.trainer, state)
        self.assertTrue(unchanged(self.hypernet, state))

    def test_restore_removes_a_task_added_after_the_snapshot(self):
        state = snapshot(self.hypernet, self.trainer)
        self.hypernet.add_task("1")
        self.trainer.task_buffers["1"] = dict(self.trainer.buffer_template)
        self.assertIn("1", self.hypernet.task_codes)
        restore(self.hypernet, self.trainer, state)
        self.assertNotIn("1", self.hypernet.task_codes)
        self.assertNotIn("1", self.trainer.task_buffers)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.config, self.target, self.hypernet = small_model()
        self.tasks = fake_tasks(self.config, ["0", "1"])
        self.trainer = UnCLe(self.hypernet, self.config, self.target, self.tasks,
                             progress=False)
        self.trainer.learn("0", [])

    def probe(self, **kwargs):
        settings = dict(task="1", train_set=self.tasks["1"]["train"],
                        test_set=self.tasks["1"]["test"], steps=4,
                        learning_rate=1e-3, batch_size=4, checkpoints=2)
        settings.update(kwargs)
        return probe_once(self.hypernet, self.target, self.trainer, **settings)

    def test_probe_leaves_the_model_untouched(self):
        # This is the fork-and-discard claim, and the reason the probe exists.
        state = snapshot(self.hypernet, self.trainer)
        self.probe()
        self.assertTrue(unchanged(self.hypernet, state))
        self.assertNotIn("1", self.hypernet.task_codes)

    def test_probe_restores_even_when_training_fails(self):
        state = snapshot(self.hypernet, self.trainer)
        # An impossible learning rate drives the loss to a non-finite value,
        # which the probe must notice and still clean up after.
        with self.assertRaises(FloatingPointError):
            self.probe(learning_rate=1e30, steps=40, checkpoints=1)
        self.assertTrue(unchanged(self.hypernet, state))
        self.assertNotIn("1", self.hypernet.task_codes)

    def test_two_probes_from_one_model_agree(self):
        # Same seed, same model, no leftover state: identical results.
        first, second = self.probe(), self.probe()
        self.assertAlmostEqual(first["final_train_loss"], second["final_train_loss"], places=10)
        self.assertAlmostEqual(first["test_accuracy"], second["test_accuracy"], places=10)

    def test_repetitions_differ_in_the_starting_code(self):
        # Repetitions exist to measure the probe's own wobble. If the seed
        # changed only the batch order they would come out nearly identical,
        # the noise floor built from their spread would be far too small, and
        # any later comparison would read as significant.
        first = self.probe(seed=1, steps=12)
        second = self.probe(seed=2, steps=12)
        self.assertNotAlmostEqual(first["final_train_loss"],
                                  second["final_train_loss"], places=6)

    def test_repetitions_spread_enough_to_measure(self):
        # The noise floor is built from this spread, so it must be a real
        # number rather than zero. Five repetitions on one unchanged model.
        losses = [self.probe(seed=s, steps=12)["final_train_loss"] for s in range(5)]
        self.assertGreater(max(losses) - min(losses), 0.0)
        self.assertEqual(len(set(round(v, 9) for v in losses)), 5)

    def test_code_only_leaves_the_generator_alone(self):
        before = [p.detach().clone() for p in self.hypernet.generator_parameters()]
        self.probe(code_only=True, steps=8)
        after = self.hypernet.generator_parameters()
        for old, new in zip(before, after):
            self.assertTrue(torch.equal(old, new))

    def test_reports_readings_and_effective_rank(self):
        result = self.probe(checkpoints=2, feature_images=8)
        self.assertEqual([r["step"] for r in result["readings"]], [2, 4])
        self.assertIn("test_accuracy", result["readings"][0])
        self.assertGreater(result["effective_rank"], 0)
        self.assertEqual(result["effective_rank_images"], 8)

    def test_refuses_a_task_the_model_already_knows(self):
        with self.assertRaises(ValueError):
            self.probe(task="0", train_set=self.tasks["0"]["train"])

    def test_random_labels_probe_needs_no_test_set(self):
        labelled = RandomLabels(self.tasks["1"]["train"], self.config.classes_per_task, 99)
        result = self.probe(train_set=labelled, test_set=None)
        self.assertNotIn("test_accuracy", result)
        self.assertGreater(result["final_train_loss"], 0)


class RandomLabelsTests(unittest.TestCase):
    def test_labels_are_fixed_and_reproducible(self):
        base = TensorDataset(torch.randn(20, 3), torch.zeros(20, dtype=torch.long))
        first = RandomLabels(base, 10, seed=7)
        second = RandomLabels(base, 10, seed=7)
        self.assertTrue(torch.equal(first.labels, second.labels))
        self.assertEqual(first[3][1], second[3][1])

    def test_a_different_seed_gives_different_labels(self):
        base = TensorDataset(torch.randn(50, 3), torch.zeros(50, dtype=torch.long))
        self.assertFalse(torch.equal(RandomLabels(base, 10, 1).labels,
                                     RandomLabels(base, 10, 2).labels))


class SelectionTests(unittest.TestCase):
    def test_e2_takes_the_highest_accuracy(self):
        runs = [{"test_accuracy": 40.0, "test_loss": 1.0, "learning_rate": 1e-4},
                {"test_accuracy": 55.0, "test_loss": 2.0, "learning_rate": 1e-3}]
        self.assertEqual(best_of_grid(runs, endpoint="E2")["learning_rate"], 1e-3)

    def test_e2_breaks_a_tie_on_loss(self):
        runs = [{"test_accuracy": 50.0, "test_loss": 2.0, "learning_rate": 1e-4},
                {"test_accuracy": 50.0, "test_loss": 1.0, "learning_rate": 1e-3}]
        self.assertEqual(best_of_grid(runs, endpoint="E2")["learning_rate"], 1e-3)

    def test_e1_takes_the_lowest_training_loss(self):
        runs = [{"final_train_loss": 2.0, "learning_rate": 1e-4},
                {"final_train_loss": 0.5, "learning_rate": 1e-3}]
        self.assertEqual(best_of_grid(runs, endpoint="E1")["learning_rate"], 1e-3)


class DecisionRuleTests(unittest.TestCase):
    def test_noise_floor_grows_with_the_spread(self):
        small = noise_floor(1.0, 1.0, 5, 2)
        large = noise_floor(2.0, 2.0, 5, 2)
        self.assertAlmostEqual(large, 2 * small)

    def test_noise_floor_shrinks_with_more_repetitions(self):
        self.assertLess(noise_floor(1.0, 1.0, 20, 20), noise_floor(1.0, 1.0, 5, 2))

    def test_noise_floor_needs_repetitions(self):
        with self.assertRaises(ValueError):
            noise_floor(1.0, 1.0, 1, 2)

    def test_relative_contribution_is_one_when_nothing_changed(self):
        self.assertAlmostEqual(
            relative_contribution(50.0, 20.0, 50.0, 20.0, endpoint="E2"), 1.0)

    def test_relative_contribution_is_zero_when_the_generator_adds_nothing(self):
        self.assertAlmostEqual(
            relative_contribution(20.0, 20.0, 50.0, 20.0, endpoint="E2"), 0.0)

    def test_relative_contribution_refuses_a_zero_baseline(self):
        # This is the case that makes R(t) untrustworthy in this framework:
        # with no pretrained backbone the control at t=0 sits near the full
        # probe, so the baseline contribution is near zero.
        with self.assertRaises(ValueError):
            relative_contribution(50.0, 20.0, 10.0, 10.0, endpoint="E2")


if __name__ == "__main__":
    unittest.main()
