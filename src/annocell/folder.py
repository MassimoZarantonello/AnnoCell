"""The round folder: everything the annotation tool reads and writes, with numpy and tifffile only.

The tree: one folder per annotator, the only one they open, holding their rounds and test regions:

- ``<root>/<annotator>/round_<NN>/<roi>/``: an ROI given to that annotator in round ``NN`` (``round_folder``);
- ``<root>/<annotator>/test/<roi>/``: a test region, drawn from scratch (``test_regions_folder``).

``rois`` lists them. Annotator names are lowercase letters, digits and ``-`` (``check_annotator``).

One folder per image (an ROI), self-contained so that it can be copied to the annotator's laptop:

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

``annotation.json`` fields the tool and overviews read (all optional except ``format_version``):

- written at export: ``annotator``, ``round`` (absent for a test region), ``image``, ``mode`` (``train`` /
  ``test``), ``source`` (the slide) and ``region`` (``[y0, y1, x0, x1]`` in slide pixels), ``exported_on``,
  ``channels``, ``visible_channels``, ``colours``, ``start_segmentation``, ``compare``;
- written when the round is over: ``closed_on`` (``close``): the tool opens the ROI read-only;
- written by the tool at every save: ``review`` (``summary``, ``saved_at``, ``started_at``: the first save, i.e. the
  first change; ``completed_at``: when the last cell was checked, None while some are not; ``start_cells`` and
  ``deleted``: the proposal's cells, and those deleted); ``is_complete`` reads it.

Next to the annotators' folders, ``<root>/scores/<model>.json``: each model's score on the test regions
(``write_score``, ``read_scores``), written by the pipeline.

Files are written to a temporary name and renamed, so a crash during a save never leaves a broken file.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
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
# Folder names in the tree, under an annotator's folder.
ROUND = "round_{:02d}"
TEST = "test"
SCORES = "scores"  # <root>/scores/<model>.json: each model's score on the test regions (``write_score``)
ANNOTATOR = re.compile(r"[a-z0-9-]+")  # a folder name on Mac and Windows alike

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


def is_complete(record: Mapping) -> bool:
    """Every cell checked (at least one cell, none unreviewed) at the last save: what training takes.

    For a test region (drawn from scratch, every drawn cell OK) it only says that cells were drawn, not that the
    drawing is finished.
    """
    review = record.get("review") or {}
    return review.get("cells", 0) > 0 and review.get(UNREVIEWED, 1) == 0


def close(folder: str | Path, on: date | None = None) -> None:
    """The round of this ROI is over: ``closed_on`` (default today); the tool then opens the ROI read-only."""
    record = read_record(folder)
    record["closed_on"] = (on or date.today()).isoformat()
    write_record(folder, record)


def is_closed(record: Mapping) -> bool:
    return bool(record.get("closed_on"))


# The tree


def check_annotator(name: str) -> str:
    """``name`` if it is a valid annotator name (lowercase letters, digits and ``-``), else a ValueError."""
    if not ANNOTATOR.fullmatch(name):
        raise ValueError(f"Annotator name {name!r}: use lowercase letters, digits and '-' only (e.g. 'alice').")
    return name


def round_folder(root: str | Path, annotator: str, round_number: int) -> Path:
    """``<root>/<annotator>/round_<NN>``: the ROIs of round ``round_number`` given to ``annotator``."""
    return Path(root) / check_annotator(annotator) / ROUND.format(round_number)


def test_regions_folder(root: str | Path, annotator: str) -> Path:
    """``<root>/<annotator>/test``: the test regions ``annotator`` draws from scratch."""
    return Path(root) / check_annotator(annotator) / TEST


@dataclass(frozen=True)
class Roi:
    """An ROI folder of the tree; ``round`` is None for a test region."""

    folder: Path
    annotator: str
    round: int | None

    @property
    def name(self) -> str:
        return self.folder.name


def rois(root: str | Path) -> list[Roi]:
    """Every ROI folder under ``root``, by annotator, then round (test regions last), then name.

    Annotator and round are read from the folder names; other folders (not an annotator name, not ``round_<NN>`` or
    ``test``) are left out.
    """
    found = []
    for path in Path(root).glob(f"*/*/*/{RECORD}"):
        folder = path.parent
        annotator, group = folder.parent.parent.name, folder.parent.name
        number = re.fullmatch(r"round_(\d+)", group)
        if ANNOTATOR.fullmatch(annotator) and (number or group == TEST):
            found.append(Roi(folder, annotator, int(number[1]) if number else None))
    return sorted(found, key=lambda r: (r.annotator, r.round is None, r.round or 0, r.name))


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


def save(folder: str | Path, masks: np.ndarray, cells: Mapping[int, Cell], start: np.ndarray | None = None) -> None:
    """Write the masks, the cell table and the review summary in ``annotation.json``.

    The review keeps when the annotator started (the first save) and when they finished (the save at which no cell
    was left unreviewed; cleared if a cell is unchecked again). Not how long they worked: only these two moments.
    With ``start`` (the proposal), it also counts the proposal's cells (``start_cells``) and those deleted
    (``deleted``: gone entirely; a cell that lost pixels is edited): with ``proposed`` (kept) and ``edited``, how
    much of the proposal had to be fixed.
    """
    write_masks(folder, masks)
    write_cells(folder, cells)
    record = read_record(folder)
    previous = record.get("review") or {}
    now = datetime.now().isoformat(timespec="seconds")
    review = {**summary(cells), "saved_at": now, "started_at": previous.get("started_at") or now}
    if start is not None:
        proposed = np.unique(start[start > 0])
        review["start_cells"] = int(len(proposed))
        review["deleted"] = int(len(np.setdiff1d(proposed, np.unique(masks[masks > 0]))))
    done = review["cells"] > 0 and review[UNREVIEWED] == 0
    review["completed_at"] = (previous.get("completed_at") or now) if done else None
    record["review"] = review
    write_record(folder, record)


def write_score(root: str | Path, score: Mapping) -> Path:
    """Write the score of one model on the test regions to ``<root>/scores/<model>.json`` and return the file.

    ``score``: ``model`` (its name), ``round`` (the round whose corrections it was trained on; 0 for the starting
    model), ``scored_on`` and ``regions``: one entry per test region with ``region``, ``annotator`` and the scores
    (``AP@0.5``, ``AP@0.75``, ...). Written by the pipeline (AnnoLoop), read by overviews (AnnOverview).
    """
    path = Path(root) / SCORES / f"{score['model']}.json"
    _write_atomic(path, lambda f: f.write((json.dumps(dict(score), indent=2) + "\n").encode()))
    return path


def read_scores(root: str | Path) -> list[dict]:
    """Every model's score on the test regions under ``root``, by round then model name."""
    scores = [json.loads(p.read_text()) for p in (Path(root) / SCORES).glob("*.json")]
    return sorted(scores, key=lambda s: (s.get("round", 0), s.get("model", "")))


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
