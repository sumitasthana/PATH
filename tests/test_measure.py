"""Checks for the shared measurement helpers."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.measure import probe, relative_change


class RelativeChangeTests(unittest.TestCase):
    def test_identical_weights_give_zero(self):
        weights = {"a": torch.tensor([3.0, 4.0]), "b": torch.ones(2, 2)}
        self.assertEqual(relative_change(weights, dict(weights)), 0.0)

    def test_known_value(self):
        # Reference norm is 5. The change is one unit. 1 / 5 = 0.2.
        reference = {"a": torch.tensor([3.0, 4.0])}
        current = {"a": torch.tensor([3.0, 5.0])}
        self.assertAlmostEqual(relative_change(current, reference), 0.2, places=12)

    def test_pools_before_dividing(self):
        # Two tensors: one moves, one does not. Pooling must use the total
        # reference size of both, not average the two separate ratios.
        reference = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0, 5.0])}
        current = {"a": torch.tensor([3.0, 5.0]), "b": torch.tensor([0.0, 5.0])}
        # sqrt(1 / (25 + 25)) = 0.1414..., not the mean of 0.2 and 0.0.
        self.assertAlmostEqual(relative_change(current, reference), (1 / 50) ** 0.5, places=12)

    def test_per_tensor_locates_the_change(self):
        reference = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0, 5.0])}
        current = {"a": torch.tensor([3.0, 5.0]), "b": torch.tensor([0.0, 5.0])}
        pooled, parts = relative_change(current, reference, per_tensor=True)
        self.assertAlmostEqual(parts["a"], 0.2, places=12)
        self.assertEqual(parts["b"], 0.0)
        self.assertLess(pooled, parts["a"])

    def test_rejects_mismatched_keys(self):
        with self.assertRaises(ValueError):
            relative_change({"a": torch.ones(2)}, {"b": torch.ones(2)})

    def test_rejects_changed_shape(self):
        with self.assertRaises(ValueError):
            relative_change({"a": torch.ones(3)}, {"a": torch.ones(2)})


class _Target(nn.Module):
    """A one-layer stand-in for the generated classifier."""

    def __init__(self, features, classes):
        super().__init__()
        self.fc = nn.Linear(features, classes)

    def forward(self, images):
        return self.fc(images)


class _Trainer:
    """Just the two attributes probe reads from the real trainer."""

    def __init__(self, task):
        self.device = torch.device("cpu")
        self.task_buffers = {task: {}}


class _Hypernet(nn.Module):
    """Returns a fixed weight set, so probe's arithmetic is the only variable."""

    def __init__(self, weights):
        super().__init__()
        self._weights = weights

    def weights_for(self, task):
        return self._weights


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.classes = 3
        self.images = torch.randn(20, 4)
        self.labels = torch.randint(0, self.classes, (20,))
        self.dataset = TensorDataset(self.images, self.labels)
        self.target = _Target(4, self.classes)
        self.trainer = _Trainer("t")
        self.weights = {"fc.weight": self.target.fc.weight.detach().clone(),
                        "fc.bias": self.target.fc.bias.detach().clone()}
        self.hypernet = _Hypernet(self.weights)

    def scores(self):
        return self.images @ self.weights["fc.weight"].T + self.weights["fc.bias"]

    def test_streaming_range_matches_one_shot(self):
        # The batches are smaller than the dataset, so the running minimum and
        # maximum must survive across batches. This is the part that breaks.
        expected = (self.scores().max(0).values - self.scores().min(0).values).max().item()
        result = probe(self.hypernet, self.target, self.trainer, self.dataset,
                       code_task="t", classes=self.classes, batch_size=3)
        self.assertAlmostEqual(result["input_sensitivity"], expected, places=5)
        self.assertEqual(result["images"], 20)

    def test_accuracy_and_counts_agree(self):
        chosen = self.scores().argmax(1)
        result = probe(self.hypernet, self.target, self.trainer, self.dataset,
                       code_task="t", classes=self.classes, batch_size=7)
        self.assertAlmostEqual(result["accuracy"],
                               100.0 * (chosen == self.labels).sum().item() / 20, places=9)
        self.assertEqual(sum(result["class_counts"]), 20)
        self.assertEqual(result["distinct_classes"],
                         sum(1 for c in result["class_counts"] if c > 0))

    def test_a_dead_model_reads_near_zero(self):
        # Weights of zero make every image give the same scores, which is the
        # state the study found after unlearning. Sensitivity must detect it.
        self.weights["fc.weight"].zero_()
        self.weights["fc.bias"].zero_()
        result = probe(self.hypernet, self.target, self.trainer, self.dataset,
                       code_task="t", classes=self.classes, batch_size=7)
        self.assertEqual(result["input_sensitivity"], 0.0)
        self.assertEqual(result["distinct_classes"], 1)

    def test_training_mode_is_restored(self):
        self.hypernet.train()
        self.target.train()
        probe(self.hypernet, self.target, self.trainer, self.dataset,
              code_task="t", classes=self.classes)
        self.assertTrue(self.hypernet.training)
        self.assertTrue(self.target.training)


if __name__ == "__main__":
    unittest.main()
