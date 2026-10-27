"""Image folders in the AnnoCell layout, written directly (as a pipeline would export them)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from annocell import folder as af

CHANNELS = ["DAPI", "NaK", "F480", "AF"]


def image() -> np.ndarray:
    return np.arange(4 * 20 * 30, dtype=np.uint16).reshape(4, 20, 30)


def start_masks() -> np.ndarray:
    """Three proposed cells: 1 and 2 side by side at the top, 3 at the bottom left."""
    masks = np.zeros((20, 30), dtype=np.uint32)
    masks[0:10, 0:10] = 1
    masks[0:10, 10:20] = 2
    masks[12:20, 0:10] = 3
    return masks


def write_folder(folder: Path, test: bool = False, colours: dict | None = None) -> Path:
    """A training image (proposal + one comparison layer), or a test image drawn from scratch (``test``)."""
    af.write_image(folder, image())
    record = {
        "image": folder.name,
        "format_version": af.FORMAT_VERSION,
        "mode": "test" if test else "train",
        "channels": CHANNELS,
        "visible_channels": ["NaK", "DAPI"],
        "colours": colours or {},
    }
    if not test:
        af.write_masks(folder, start_masks(), af.START)
        compare = np.zeros((20, 30), dtype=np.uint32)
        compare[0:10, 0:20] = 1
        af.write_masks(folder / af.COMPARE, compare, "model_a.tif")
        record["start_segmentation"] = {"labels": "cells"}
        record["compare"] = [{"labels": "model_a", "trained_on_image": True}]
    af.write_record(folder, record)
    return folder


@pytest.fixture
def train_folder(tmp_path) -> Path:
    return write_folder(tmp_path / "round_02" / "test_image")


@pytest.fixture
def test_folder(tmp_path) -> Path:
    return write_folder(tmp_path / "test" / "patch_y500_x500", test=True, colours={"DAPI": "blue"})
