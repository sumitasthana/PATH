"""Checks for the weight-change indicator.

Two things have to hold. The number has to read the model, so it must be zero
when nothing moved and positive when something did. And taking the measurement
must not change the model, because every boundary measurement happens in the
middle of a stream that has to continue unaffected.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.config import Config
from uncle.hypernet import HyperNetwork, build_target
from uncle import plasticity_indicators
from uncle.plasticity_indicators import (LOSS_PROGRESS_EVERY, BoundaryLog, activation_sites,
                                         dormancy, dormant_fraction, features, flatten,
                                         indicator_key, loss_progress, loss_progress_recorder,
                                         progress_summary, ranks, reference_code, srank,
                                         weight_change)
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
        """A boundary set is larger than one batch, so the split must not matter.

        Not bit-for-bit. A convolution adds its terms in a different order at a
        different batch size, so float32 results drift by about 1e-5 on values
        that reach 25. The tolerance is set to catch a wrong slice, which would
        move whole rows, and to ignore that rounding.
        """
        _, target, hypernet, trainer, images = ready(size=10)
        code = reference_code(hypernet)
        whole = features(hypernet, target, trainer, images, code, batch_size=64)
        split = features(hypernet, target, trainer, images, code, batch_size=3)
        self.assertTrue(torch.allclose(whole, split, atol=1e-4))

    def test_a_wrong_slice_would_be_caught(self):
        """The tolerance above is loose, so check it still sees shuffled rows."""
        _, target, hypernet, trainer, images = ready(size=10)
        code = reference_code(hypernet)
        matrix = features(hypernet, target, trainer, images, code)
        self.assertFalse(torch.allclose(matrix, matrix.flip(0), atol=1e-4))

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


class SharedRelu(torch.nn.Module):
    """One ReLU module used at three places, as a torchvision ResNet block does."""

    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.first = torch.nn.Linear(4, 6)
        self.second = torch.nn.Linear(6, 8)
        self.third = torch.nn.Linear(8, 5)

    def forward(self, x):
        x = self.relu(self.first(x))
        x = self.relu(self.second(x))
        return self.relu(self.third(x))


class ActivationSites(unittest.TestCase):

    def test_one_site_per_call_not_per_module(self):
        """Three calls on one module are three places in the network, not one."""
        model = SharedRelu()
        sums = activation_sites(model, lambda reset: (reset(), model(torch.randn(7, 4))))
        self.assertEqual(len(sums), 3)
        self.assertEqual([vector.numel() for vector in sums], [6, 8, 5])

    def test_batches_accumulate_into_one_total(self):
        """Two batches of five must give the same totals as one batch of ten."""
        model = SharedRelu()
        torch.manual_seed(0)
        images = torch.randn(10, 4)

        def whole(reset):
            reset()
            model(images)

        def split(reset):
            for start in (0, 5):
                reset()
                model(images[start:start + 5])

        one, two = activation_sites(model, whole), activation_sites(model, split)
        for first, second in zip(one, two):
            self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_without_the_reset_a_batch_opens_new_sites(self):
        """The reset is what makes batch two land on batch one's sites."""
        model = SharedRelu()
        images = torch.randn(4, 4)
        sums = activation_sites(model, lambda reset: (reset(), model(images), model(images)))
        self.assertEqual(len(sums), 6)

    def test_hooks_are_removed(self):
        model = SharedRelu()
        activation_sites(model, lambda reset: (reset(), model(torch.randn(3, 4))))
        self.assertEqual(len(model.relu._forward_hooks), 0)


class DormantFraction(unittest.TestCase):

    def test_equal_units_are_never_dormant(self):
        record = dormant_fraction([torch.ones(4)])
        self.assertEqual(record["dormant_0.1"], 0.0)
        self.assertEqual(record["units"], 4)

    def test_silent_units_are_counted(self):
        """Totals of 0, 0, 1, 3 give a site mean of 1, so two units score zero."""
        record = dormant_fraction([torch.tensor([0.0, 0.0, 1.0, 3.0])])
        self.assertEqual(record["dormant_0.01"], 0.5)

    def test_a_fully_silent_site_is_fully_dormant(self):
        """There is no site mean to divide by, and every unit has stopped."""
        record = dormant_fraction([torch.zeros(5)])
        self.assertEqual(record["dormant_0.01"], 1.0)

    def test_a_higher_cut_off_never_catches_less(self):
        torch.manual_seed(0)
        record = dormant_fraction([torch.rand(50), torch.rand(30)])
        self.assertLessEqual(record["dormant_0.01"], record["dormant_0.025"])
        self.assertLessEqual(record["dormant_0.025"], record["dormant_0.1"])

    def test_a_quiet_site_is_not_a_dormant_site(self):
        """Scoring against one network-wide mean would wipe out the quiet site.

        Both sites are perfectly healthy: every unit matches its neighbours.
        The quiet one sits far below every cut-off in raw terms, so only
        dividing by its own site mean saves it.
        """
        loud, quiet = torch.tensor([1000.0, 1000.0]), torch.tensor([0.001, 0.001])
        self.assertEqual(dormant_fraction([loud, quiet])["dormant_0.1"], 0.0)

    def test_scale_does_not_matter_within_a_site(self):
        """Twice as many images means twice the totals and the same answer."""
        once = dormant_fraction([torch.tensor([0.0, 2.0, 4.0])])
        twice = dormant_fraction([torch.tensor([0.0, 4.0, 8.0])])
        self.assertEqual(once, twice)


class Dormancy(unittest.TestCase):

    def test_one_entry_per_cut_off_per_set(self):
        _, target, hypernet, trainer, images = ready()
        record = dormancy(hypernet, target, trainer, {"held_out": images},
                          reference_code(hypernet))
        self.assertEqual(set(record), {
            "dormant_0.01_held_out", "dormant_0.025_held_out",
            "dormant_0.1_held_out", "units_held_out"})
        self.assertGreater(record["units_held_out"], 0)

    def test_fractions_stay_between_zero_and_one(self):
        _, target, hypernet, trainer, images = ready()
        record = dormancy(hypernet, target, trainer, {"set": images},
                          reference_code(hypernet))
        for threshold in (0.01, 0.025, 0.1):
            value = record[f"dormant_{threshold}_set"]
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_same_code_gives_the_same_answer(self):
        _, target, hypernet, trainer, images = ready()
        code = reference_code(hypernet)
        first = dormancy(hypernet, target, trainer, {"set": images}, code)
        second = dormancy(hypernet, target, trainer, {"set": images}, code)
        self.assertEqual(first, second)

    def test_batching_does_not_change_the_answer(self):
        _, target, hypernet, trainer, images = ready(size=10)
        code = reference_code(hypernet)
        whole = dormancy(hypernet, target, trainer, {"set": images}, code, batch_size=64)
        split = dormancy(hypernet, target, trainer, {"set": images}, code, batch_size=3)
        self.assertEqual(whole, split)


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
        dormancy(hypernet, target, trainer, {"set": images}, reference_code(hypernet))
        self.assertTrue(unchanged(hypernet, state))
        for name, value in template.items():
            self.assertTrue(torch.equal(value, trainer.buffer_template[name]), name)

    def test_dormancy_leaves_no_hooks_on_the_target(self):
        """A hook left behind would fire during the next task's training."""
        _, target, hypernet, trainer, images = ready()
        dormancy(hypernet, target, trainer, {"set": images}, reference_code(hypernet))
        for child in target.modules():
            self.assertEqual(len(child._forward_hooks), 0)

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


def fake_tasks(config, names, size=32):
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


def learner():
    """A trainer with one task's worth of real data, ready to learn it."""
    config, target, hypernet = small_model()
    tasks = fake_tasks(config, ["0"])
    return config, target, hypernet, UnCLe(hypernet, config, target, tasks)


def with_fake_forward(hypernet, target, trainer, record):
    """Run `loss_progress` with the forward pass replaced by fixed logits.

    Returns the row and the buffers the forward pass was handed. The fixed
    logits make the after value the same every time, so a test can state the
    arithmetic exactly instead of asserting on a trained network.
    """
    seen = []

    def spy(module, state, args, **kwargs):
        seen.append(state[1])
        return torch.zeros(len(args[0]), 4)

    real = plasticity_indicators.functional_call
    plasticity_indicators.functional_call = spy
    try:
        row = loss_progress(hypernet, target, trainer, "0", record, every=1)
    finally:
        plasticity_indicators.functional_call = real
    return row, seen


def ready_to_spy():
    """A learner with task 0 registered and one stand-in buffer to watch."""
    _, target, hypernet, trainer = learner()
    hypernet.add_task("0")
    trainer.task_buffers["0"] = {"fake": torch.ones(3)}
    return target, hypernet, trainer


def sampled_record(**changes):
    """One `on_step` record that passes every skip rule."""
    record = {"step": 1, "epoch": 1, "task_loss": 1.0, "loss": 5.0,
              "images": torch.zeros(2, 1, 1, 1),
              "labels": torch.zeros(2, dtype=torch.long)}
    record.update(changes)
    return record


class LossProgress(unittest.TestCase):
    """The arithmetic and the sampling rule, with no training involved."""

    def test_step_zero_is_skipped(self):
        """Step zero reports the state before any update, so nothing moved yet."""
        self.assertIsNone(loss_progress(None, None, None, "0", {"step": 0}, every=1))

    def test_unsampled_steps_are_skipped(self):
        """Sampling is the whole reason this is cheap."""
        record = {"step": 7, "epoch": 1, "task_loss": 1.0,
                  "images": torch.zeros(1), "labels": torch.zeros(1)}
        self.assertIsNone(loss_progress(None, None, None, "0", record, every=10))

    def test_a_record_without_a_batch_is_skipped(self):
        """Epoch and step-zero records carry no batch to run again."""
        record = {"step": 10, "epoch": 1, "task_loss": 1.0}
        self.assertIsNone(loss_progress(None, None, None, "0", record, every=10))

    def test_a_zero_before_value_is_skipped(self):
        """There is no fraction of zero, so the row would be meaningless."""
        record = {"step": 10, "epoch": 1, "task_loss": 0.0,
                  "images": torch.zeros(1), "labels": torch.zeros(1)}
        self.assertIsNone(loss_progress(None, None, None, "0", record, every=10))

    def test_the_fraction_matches_the_two_losses(self):
        """The reported fraction must be the drop divided by the starting loss."""
        _, target, hypernet, trainer = learner()
        rows, watch = loss_progress_recorder(hypernet, target, trainer, "0", every=1)
        trainer.learn("0", [], on_step=watch)
        self.assertTrue(rows)
        for row in rows:
            self.assertAlmostEqual(
                row["loss_progress"],
                (row["loss_before"] - row["loss_after"]) / row["loss_before"], places=6)

    def test_the_before_value_leaves_the_protection_term_out(self):
        """The after value is a bare cross entropy, so the before value must be too.

        The record's `loss` field adds the weighted protection term, which this
        function cannot recompute. Using it would report a drop that is mostly
        the two terms disagreeing rather than the update doing work.
        """
        target, hypernet, trainer = ready_to_spy()
        row, _ = with_fake_forward(hypernet, target, trainer,
                                   sampled_record(task_loss=1.0, loss=5.0))
        self.assertEqual(row["loss_before"], 1.0)

    def test_the_measurement_consumes_no_randomness(self):
        """The study's own loop does not guard its callbacks, so this one must.

        The real network draws no random numbers in a forward pass, so no
        ordinary run would show a missing guard. A stand-in that does draw one
        makes the guard the only thing standing between the measurement and
        every batch that follows it.
        """
        target, hypernet, trainer = ready_to_spy()
        seen = []

        def greedy(module, state, args, **kwargs):
            torch.rand(1)
            return torch.zeros(len(args[0]), 4)

        real = plasticity_indicators.functional_call
        plasticity_indicators.functional_call = greedy
        try:
            before = torch.get_rng_state()
            loss_progress(hypernet, target, trainer, "0", sampled_record(), every=1)
            seen.append(torch.get_rng_state())
        finally:
            plasticity_indicators.functional_call = real
        self.assertTrue(torch.equal(before, seen[0]))

    def test_the_measurement_builds_no_graph(self):
        """A graph held through a ResNet50 forward pass is real memory."""
        target, hypernet, trainer = ready_to_spy()
        enabled = []

        def watcher(module, state, args, **kwargs):
            enabled.append(torch.is_grad_enabled())
            return torch.zeros(len(args[0]), 4)

        real = plasticity_indicators.functional_call
        plasticity_indicators.functional_call = watcher
        try:
            loss_progress(hypernet, target, trainer, "0", sampled_record(), every=1)
        finally:
            plasticity_indicators.functional_call = real
        self.assertEqual(enabled, [False])

    def test_the_default_matches_the_adapter_study(self):
        """Every tenth update is the adapter study's own sampling rate."""
        self.assertEqual(LOSS_PROGRESS_EVERY, 10)


class LossProgressRecorder(unittest.TestCase):

    def test_rows_land_on_the_sampled_steps_only(self):
        """One row every `every` updates, numbered by the update it followed."""
        _, target, hypernet, trainer = learner()
        rows, watch = loss_progress_recorder(hypernet, target, trainer, "0", every=2)
        trainer.learn("0", [], on_step=watch)
        self.assertEqual([row["step"] for row in rows], [2, 4])

    def test_an_update_usually_lowers_its_own_batch_loss(self):
        """A reading that never moves would mean the pass is measuring nothing."""
        _, target, hypernet, trainer = learner()
        rows, watch = loss_progress_recorder(hypernet, target, trainer, "0", every=1)
        trainer.learn("0", [], on_step=watch)
        self.assertTrue(any(row["loss_progress"] > 0 for row in rows))

    def test_watching_does_not_change_the_run(self):
        """The measurement rides along, so the learning must come out identical."""
        _, _, plain, trainer = learner()
        quiet = trainer.learn("0", [])
        quiet_weights = flatten(plain, reference_code(plain))

        _, target, hypernet, trainer = learner()
        _, watch = loss_progress_recorder(hypernet, target, trainer, "0", every=1)
        watched = trainer.learn("0", [], on_step=watch)
        watched_weights = flatten(hypernet, reference_code(hypernet))

        self.assertEqual(quiet, watched)
        torch.testing.assert_close(quiet_weights["generator"],
                                   watched_weights["generator"], rtol=0, atol=0)

    def test_the_live_buffers_are_not_handed_to_the_forward_pass(self):
        """Training mode writes batch statistics back, so a clone must take the write.

        The small CPU backbone has no batch normalisation, so nothing would go
        wrong here even if the live buffers were passed. The check is on the
        argument itself, which is the only place the mistake would show on a
        model that does have it.
        """
        target, hypernet, trainer = ready_to_spy()
        _, seen = with_fake_forward(hypernet, target, trainer, sampled_record())

        self.assertEqual(len(seen), 1)
        self.assertIsNot(seen[0], trainer.task_buffers["0"])
        self.assertIsNot(seen[0]["fake"], trainer.task_buffers["0"]["fake"])


class ProgressSummary(unittest.TestCase):

    def test_the_mean_is_the_mean_of_the_rows(self):
        rows = [{"step": 10, "loss_progress": 0.2}, {"step": 20, "loss_progress": 0.4}]
        self.assertAlmostEqual(progress_summary(rows)["loss_progress_mean"], 0.3)

    def test_the_curve_keeps_the_step_numbers(self):
        """A mean hides whether a task started slow or ended slow."""
        rows = [{"step": 10, "loss_progress": 0.2}, {"step": 20, "loss_progress": 0.4}]
        self.assertEqual(progress_summary(rows)["loss_progress_curve"],
                         [[10, 0.2], [20, 0.4]])

    def test_no_rows_gives_no_mean(self):
        """Boundary zero has no task behind it, so there is no number."""
        self.assertIsNone(progress_summary([])["loss_progress_mean"])


class IndicatorKey(unittest.TestCase):

    def test_one_file_per_run(self):
        """Order and seed both go in the path, so two runs never share a file."""
        self.assertEqual(indicator_key("paired_v2", "order_01", 2),
                         "uncle/indicators/paired_v2/order_01/seed_2.jsonl")


class Boundaries(unittest.TestCase):
    """The writer, checked on a tiny model and a temporary folder."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "seed_0.jsonl"
        self.config, self.target, self.hypernet, self.trainer, images = ready()
        self.image_sets = {"in_stream": images}

    def log(self, hypernet=None):
        return BoundaryLog(self.path, hypernet or self.hypernet, self.target,
                           self.trainer, self.image_sets,
                           study="test", order="order_01", seed=0)

    def rows(self):
        return [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines()]

    def test_boundary_zero_has_not_moved(self):
        """The first boundary is its own baseline, so the distance is exactly zero."""
        row = self.log().record(0)
        self.assertEqual(row["t"], 0)
        self.assertIsNone(row["task"])
        self.assertEqual(row["delta_w_generator"], 0.0)
        self.assertEqual(row["delta_w_target"], 0.0)

    def test_one_line_per_boundary(self):
        """The file is read back line by line, so a row must never span two lines."""
        log = self.log()
        log.record(0)
        nudge(self.hypernet)
        log.record(1, task="3")
        self.assertEqual([row["t"] for row in self.rows()], [0, 1])
        self.assertEqual(self.rows()[1]["task"], "3")

    def test_a_later_boundary_is_measured_against_boundary_zero(self):
        log = self.log()
        log.record(0)
        nudge(self.hypernet)
        self.assertGreater(log.record(1, task="3")["delta_w_generator"], 0)

    def test_every_measure_lands_in_one_flat_row(self):
        """One line holds all of it, so the reporting step needs no nesting."""
        row = self.log().record(0)
        for key in ("norm_target", "delta_w_target", "effective_rank_in_stream",
                    "srank_in_stream", "dormant_0.1_in_stream", "units_in_stream",
                    "loss_progress_mean", "loss_progress_curve"):
            self.assertIn(key, row)

    def test_loss_progress_rows_are_folded_in(self):
        rows = [{"step": 10, "loss_progress": 0.5}]
        row = self.log().record(1, task="3", progress_rows=rows)
        self.assertAlmostEqual(row["loss_progress_mean"], 0.5)

    def test_a_boundary_already_written_is_not_written_again(self):
        """A resumed run must not append a second copy of a boundary it finished."""
        self.log().record(0)
        self.assertIsNone(self.log().record(0))
        self.assertEqual(len(self.rows()), 1)

    def test_a_resumed_run_keeps_the_original_baseline(self):
        """A fresh baseline would reset every later distance to zero.

        This is the whole reason the baseline is saved to its own file. The
        resumed log has never seen boundary zero's weights, so without that
        file it would measure boundary one against itself.
        """
        self.log().record(0)
        nudge(self.hypernet)
        self.assertGreater(self.log().record(1, task="3")["delta_w_generator"], 0)

    def test_rows_without_a_baseline_are_refused(self):
        """Carrying on would silently restart the distances from the wrong place."""
        self.log().record(0)
        self.path.with_suffix(".baseline.pt").unlink()
        with self.assertRaises(ValueError):
            self.log()

    def test_a_fresh_folder_needs_no_setup(self):
        """The writer is handed to a run, not set up beside one."""
        self.path = Path(self.folder.name) / "deeper" / "seed_0.jsonl"
        self.assertEqual(self.log().record(0)["t"], 0)
        self.assertTrue(self.path.exists())


class WrittenRecordsRead(unittest.TestCase):
    """The writer and the reporting have to agree about every key name.

    Both halves are hand-written, so a key renamed on one side and not the
    other would pass both sets of tests and produce an empty column. This is
    the only check that runs a real record through both.
    """

    def test_a_real_record_flattens_into_the_long_table(self):
        from uncle.indicator_report import long_table, read_rows

        config, target, hypernet, trainer, images = ready()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seed_0.jsonl"
            log = BoundaryLog(path, hypernet, target, trainer,
                              {"in_stream": images}, study="test",
                              order="order_01", seed=0)
            log.record(0)
            nudge(hypernet)
            log.record(1, task="3", progress_rows=[{"step": 10, "loss_progress": 0.5}])

            flat = long_table(read_rows(path))

        found = {(row["measure"], row["image_set"]) for row in flat}
        for pair in (("delta_w_target", ""), ("effective_rank", "in_stream"),
                     ("srank", "in_stream"), ("dormant_0.01", "in_stream"),
                     ("loss_progress_mean", "")):
            self.assertIn(pair, found)


if __name__ == "__main__":
    unittest.main()
