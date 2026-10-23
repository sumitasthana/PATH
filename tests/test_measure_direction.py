"""Checks for measure.direction_change.

The case this function exists for is unlearning, where the weights shrink a lot.
Shrinking must read as no change of direction, or the measurement cannot tell a
recoverable scaling apart from a destroyed pattern.
"""

import math

import torch

from uncle.measure import direction_change, relative_change


def test_pure_shrink_keeps_direction():
    """Halving every weight moves them a long way but turns them not at all."""
    reference = {"a": torch.randn(4, 3), "b": torch.randn(7)}
    current = {name: value * 0.5 for name, value in reference.items()}

    assert math.isclose(direction_change(current, reference), 1.0, abs_tol=1e-12)
    # The companion measure disagrees, which is the whole point of having both.
    assert math.isclose(relative_change(current, reference), 0.5, abs_tol=1e-12)


def test_sign_flip_is_minus_one():
    reference = {"a": torch.randn(5, 2)}
    current = {"a": -reference["a"]}
    assert math.isclose(direction_change(current, reference), -1.0, abs_tol=1e-12)


def test_pooling_is_one_long_vector_not_an_average():
    """One tensor keeps its direction, a much larger one reverses.

    An average of the two per-tensor cosines would be 0.0. Pooling by size gives
    a negative answer, because the larger tensor carries more of the weights.
    """
    reference = {"small": torch.ones(2), "large": torch.ones(200)}
    current = {"small": torch.ones(2), "large": -torch.ones(200)}

    pooled, parts = direction_change(current, reference, per_tensor=True)
    assert math.isclose(parts["small"], 1.0, abs_tol=1e-12)
    assert math.isclose(parts["large"], -1.0, abs_tol=1e-12)
    assert pooled < -0.9


def test_zero_tensor_is_nan_not_a_number_we_invented():
    reference = {"alive": torch.ones(3), "dead": torch.ones(3)}
    current = {"alive": torch.ones(3), "dead": torch.zeros(3)}

    _pooled, parts = direction_change(current, reference, per_tensor=True)
    assert math.isnan(parts["dead"])
    assert math.isclose(parts["alive"], 1.0, abs_tol=1e-12)


def test_mismatched_keys_raise():
    try:
        direction_change({"a": torch.ones(2)}, {"b": torch.ones(2)})
    except ValueError:
        return
    raise AssertionError("Different keys should raise.")


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print("ok", name)
