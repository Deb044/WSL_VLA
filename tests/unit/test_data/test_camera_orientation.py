"""The measured rollout-camera orientation verdict (scripts/setup/verify_camera_orientation.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wsl_vla.data.preprocessing import IMAGE_TRANSFORMS, apply_image_transform

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/setup/verify_camera_orientation.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("verify_camera_orientation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene(seed: int = 0) -> np.ndarray:
    """An image with no flip or rotation symmetry, plus mild renderer noise."""

    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0:32, 0:32]
    image = np.stack([4 * rows + columns, 2 * rows + 5 * columns, 7 * columns], axis=-1)
    return np.clip(image + rng.integers(0, 6, size=image.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("transform", IMAGE_TRANSFORMS)
def test_measured_alignment_recovers_the_transform_between_simulator_and_demonstration(
    script, transform
):
    stored = scene()
    # The simulator frame is whatever maps onto the stored frame under ``transform``;
    # every candidate transform is its own inverse.
    simulator = apply_image_transform(stored[None], transform)[0]
    noisy = np.clip(simulator.astype(np.int16) + 2, 0, 255).astype(np.uint8)

    result = script.measured_alignment(noisy, stored)

    assert result["best"] == transform
    assert set(result["mean_squared_error"]) == set(IMAGE_TRANSFORMS)
    assert result["margin"] > script.MARGIN
    assert result["mean_squared_error"][transform] == min(result["mean_squared_error"].values())


def test_an_ambiguous_scene_does_not_produce_a_confident_verdict(script):
    """A frame that looks the same under every transform cannot verify an orientation."""

    uniform = np.full((16, 16, 3), 90, dtype=np.uint8)
    result = script.measured_alignment(uniform, uniform)

    assert result["margin"] < script.MARGIN
