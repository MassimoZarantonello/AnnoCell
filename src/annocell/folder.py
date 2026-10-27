"""The round folder: everything the annotation tool reads and writes, with numpy and tifffile only.

One folder per image (``<root>/round_<NN>/<image>/`` for training images, ``<root>/test/<patch>/`` for test
patches), self-contained so that it can be copied to the annotator's laptop:

- ``image.tif``: every channel of the crop, (channels, y, x); the names are in ``annotation.json`` (``channels``);
- ``start_masks.tif``: the proposed masks as given to the annotator; absent in test mode (start from empty);
- ``compare/<segmentation>.tif``: read-only comparison masks; ``annotation.json`` (``compare``) says whether each
  model was trained on this image;
- ``masks.tif``: the current masks (uint32 labels), written by the tool;
- ``cells.csv``: one row per cell of ``masks.tif``: ``cell_ID``, ``state`` (``unreviewed`` / ``ok``),
  ``origin`` (``proposed`` / ``edited`` / ``drawn``, computed against the start masks) and ``macrophage``
  (0 / 1, test mode only);
- ``annotation.json``: where it all comes from (written by the exporting pipeline, e.g. macroseg), plus the review
  summary; ``format_version`` is the version of this layout (``FORMAT_VERSION``).

Files are written to a temporary name and renamed, so a crash during a save never leaves a broken file.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

FORMAT_VERSION = 1  # of the folder layout; bump on any change an older reader would misread

# File names inside an image folder.
IMAGE = "image.tif"
START = "start_masks.tif"
MASKS = "masks.tif"
CELLS = "cells.csv"
RECORD = "annotation.json"
COMPARE = "compare"

# Review state of a cell (``state``), and where its outline comes from (``origin``).
UNREVIEWED, OK = "unreviewed", "ok"
PROPOSED, EDITED, DRAWN = "proposed", "edited", "drawn"


@dataclass
class Cell:
    """One row of ``cells.csv``: review state, origin and (test mode) macrophage mark of a cell."""

    state: str = UNREVIEWED
    origin: str = PROPOSED
    macrophage: bool = False


def read_record(folder: str | Path) -> dict:
    return json.loads((Path(folder) / RECORD).read_text())


def write_record(folder: str | Path, record: Mapping) -> None:
    # Bytes, because _write_atomic opens the temporary file in binary mode (tifffile needs it).
    _write_atomic(Path(folder) / RECORD, lambda f: f.write((json.dumps(record, indent=2) + "\n").encode()))


def is_test(record: Mapping) -> bool:
    """Test mode: drawn from scratch, no proposal and no comparison layers."""
    return record.get("mode") == "test"


def read_image(folder: str | Path) -> tuple[np.ndarray, list[str]]:
    """The image (channels, y, x) and its channel names."""
    image = tifffile.imread(Path(folder) / IMAGE)
    image = image[np.newaxis] if image.ndim == 2 else image  # a single channel is read as (y, x)
    names = read_record(folder).get("channels") or [f"channel_{i}" for i in range(len(image))]
    return image, list(names)


def write_image(folder: str | Path, image: np.ndarray) -> None:
    _write_tif(Path(folder) / IMAGE, image)


def read_start(folder: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """The proposed masks; empty (all background) in test mode, which needs ``shape``."""
    path = Path(folder) / START
    if path.exists():
        return tifffile.imread(path).astype(np.uint32)
    if shape is None:
        raise FileNotFoundError(f"{path} does not exist; pass the image shape to start from empty masks.")
    return np.zeros(shape, dtype=np.uint32)


def read_masks(folder: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """The current masks: ``masks.tif`` once the tool has saved, else the start masks."""
    path = Path(folder) / MASKS
    return tifffile.imread(path).astype(np.uint32) if path.exists() else read_start(folder, shape)


def write_masks(folder: str | Path, masks: np.ndarray, name: str = MASKS) -> None:
    _write_tif(Path(folder) / name, np.asarray(masks, dtype=np.uint32))


def read_cells(folder: str | Path) -> dict[int, Cell]:
    """The per-cell review table; empty if not written yet (every cell then unreviewed and proposed)."""
    path = Path(folder) / CELLS
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {
            int(row["cell_ID"]): Cell(row["state"], row["origin"], row.get("macrophage", "0") == "1")
            for row in csv.DictReader(f)
        }


def write_cells(folder: str | Path, cells: Mapping[int, Cell]) -> None:
    """Write the review table, one row per cell sorted by ID; ``macrophage`` as 0 / 1."""

    def write(f):
        lines = ["cell_ID,state,origin,macrophage"]
        lines += [f"{i},{c.state},{c.origin},{int(c.macrophage)}" for i, c in sorted(cells.items())]
        f.write(("\n".join(lines) + "\n").encode())

    _write_atomic(Path(folder) / CELLS, write)


def compare_layers(folder: str | Path) -> list[tuple[str, np.ndarray, bool]]:
    """The comparison masks: (name, masks, trained on this image), in the order of ``annotation.json``."""
    folder = Path(folder)
    info = read_record(folder).get("compare") or []
    return [
        (c["labels"], tifffile.imread(folder / COMPARE / f"{c['labels']}.tif"), bool(c.get("trained_on_image")))
        for c in info
    ]


def origins(masks: np.ndarray, start: np.ndarray) -> dict[int, str]:
    """``proposed`` (identical to the start cell with the same ID), ``edited`` (an ID of the start masks with
    different pixels) or ``drawn`` (an ID not in the start masks), for every cell of ``masks``."""
    # Computed from the pixels, not from the edits, so undo and redo can never leave an origin wrong. Drawn cells get
    # IDs above every ID of the start masks (``Annotator.next_id``), so a start ID still present is that same cell.
    present = np.unique(masks[masks > 0])
    in_start = set(np.unique(start[start > 0]).tolist())
    changed = masks != start
    # A cell is touched if any of its pixels changed, in the current masks or in the start masks (it lost pixels).
    touched = set(np.unique(masks[changed]).tolist()) | set(np.unique(start[changed]).tolist())
    return {int(i): DRAWN if i not in in_start else EDITED if i in touched else PROPOSED for i in present.tolist()}


def sync_cells(cells: Mapping[int, Cell], masks: np.ndarray, start: np.ndarray) -> dict[int, Cell]:
    """``cells`` matched to ``masks``: deleted cells dropped, new cells added, origins recomputed.

    A new cell (absent from ``cells``) is OK when drawn or edited, unreviewed when proposed.
    """
    synced = {}
    for i, origin in origins(masks, start).items():
        # A known cell keeps its state and macrophage mark; only its origin is updated.
        cell = cells.get(i) or Cell(state=UNREVIEWED if origin == PROPOSED else OK)
        synced[i] = Cell(cell.state, origin, cell.macrophage)
    return synced


def summary(cells: Mapping[int, Cell]) -> dict:
    """Counts of the review: cells, OK, unreviewed, by origin, macrophages."""
    values = list(cells.values())
    return {
        "cells": len(values),
        OK: sum(c.state == OK for c in values),
        UNREVIEWED: sum(c.state == UNREVIEWED for c in values),
        **{origin: sum(c.origin == origin for c in values) for origin in (PROPOSED, EDITED, DRAWN)},
        "macrophages": sum(c.macrophage for c in values),
    }


def save(folder: str | Path, masks: np.ndarray, cells: Mapping[int, Cell]) -> None:
    """Write the masks, the cell table and the review summary in ``annotation.json``."""
    write_masks(folder, masks)
    write_cells(folder, cells)
    record = read_record(folder)
    record["review"] = {**summary(cells), "saved_at": datetime.now().isoformat(timespec="seconds")}
    write_record(folder, record)


def image_folders(parent: str | Path) -> list[Path]:
    """The image folders directly under ``parent`` (a round or the test folder), sorted by name."""
    return sorted(p.parent for p in Path(parent).glob(f"*/{RECORD}"))


def ids_inside(masks: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> list[int]:
    """IDs of the cells entirely inside the window: cells cut by a window edge are left out, unless that edge
    is the edge of the image."""
    height, width = masks.shape
    y0, x0, y1, x1 = max(0, y0), max(0, x0), min(height, y1), min(width, x1)
    if y1 <= y0 or x1 <= x0:
        return []
    window = masks[y0:y1, x0:x1]
    # The cells on the window's border rows and columns are cut by the view; the image's own edges do not count.
    edges = []
    if y0 > 0:
        edges.append(window[0])
    if y1 < height:
        edges.append(window[-1])
    if x0 > 0:
        edges.append(window[:, 0])
    if x1 < width:
        edges.append(window[:, -1])
    cut = set(np.unique(np.concatenate(edges)).tolist()) if edges else set()
    return [int(i) for i in np.unique(window).tolist() if i and i not in cut]


def _write_tif(path: Path, array: np.ndarray) -> None:
    # minisblack: a 3-channel image is stored as 3 grey planes, not read back as an RGB picture.
    _write_atomic(path, lambda f: tifffile.imwrite(f, array, photometric="minisblack"))


def _write_atomic(path: Path, write) -> None:
    """Call ``write`` on a temporary file next to ``path``, then rename it to ``path``: the rename is atomic, so a
    crash while writing leaves the previous file intact, never half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as f:
        write(f)
    os.replace(tmp, path)
