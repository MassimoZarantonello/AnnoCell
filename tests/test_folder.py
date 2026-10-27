"""The folder format: reading, writing, origins of the cells, review counts."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from annocell import folder as af
from conftest import write_folder


def test_reads_what_the_pipeline_wrote(train_folder):
    image, names = af.read_image(train_folder)
    assert image.shape == (4, 20, 30) and names == ["DAPI", "NaK", "F480", "AF"]
    start = af.read_start(train_folder)
    np.testing.assert_array_equal(af.read_masks(train_folder), start)  # nothing saved yet: the proposal
    assert af.read_cells(train_folder) == {}
    [(name, masks, trained)] = af.compare_layers(train_folder)
    assert name == "model_a" and trained and masks.max() == 1
    assert not af.is_test(af.read_record(train_folder))


def test_test_folder_starts_empty(test_folder):
    assert af.is_test(af.read_record(test_folder))
    assert af.read_masks(test_folder, (20, 30)).max() == 0 and af.compare_layers(test_folder) == []


def test_origins_and_sync():
    start = np.zeros((10, 30), dtype=np.uint32)
    start[:, 0:10], start[:, 10:20] = 1, 2
    masks = start.copy()
    masks[0:2, 10:20] = 0  # cell 2 trimmed
    masks[:, 20:30] = 5  # cell 5 drawn
    assert af.origins(masks, start) == {1: af.PROPOSED, 2: af.EDITED, 5: af.DRAWN}
    cells = af.sync_cells({1: af.Cell(af.OK), 9: af.Cell()}, masks, start)
    assert cells[1].state == af.OK and 9 not in cells  # deleted cells are dropped
    assert cells[2].state == cells[5].state == af.OK  # cells new to the table: OK unless proposed


def test_ids_inside_leaves_out_cells_cut_by_the_view():
    masks = np.zeros((20, 20), dtype=np.uint32)
    masks[2:5, 2:5], masks[8:12, 8:12], masks[0:3, 15:18] = 1, 2, 3
    assert af.ids_inside(masks, 0, 10, 0, 10) == [1]  # cell 2 is cut by the window edge
    assert af.ids_inside(masks, 0, 20, 0, 20) == [1, 2, 3]
    assert af.ids_inside(masks, -5, 13, 6, 30) == [2, 3]  # image edges do not cut


def test_save_writes_masks_cells_and_review(train_folder):
    masks = af.read_start(train_folder)
    masks[masks == 3] = 0  # deleted
    masks[12:20, 20:30] = 4  # drawn
    cells = af.sync_cells({}, masks, af.read_start(train_folder))
    cells[1].macrophage = True
    af.save(train_folder, masks, cells)
    np.testing.assert_array_equal(af.read_masks(train_folder), masks)
    assert af.read_cells(train_folder) == cells
    review = af.read_record(train_folder)["review"]
    assert (review["cells"], review[af.UNREVIEWED], review[af.DRAWN], review["macrophages"]) == (3, 2, 1, 1)
    assert not list(train_folder.glob(".*.tmp"))  # no temporary file left behind


def test_image_folders_lists_folders_with_a_record(train_folder):
    (train_folder.parent / "not_an_image").mkdir()
    assert af.image_folders(train_folder.parent) == [train_folder]


def test_rois_walks_the_tree(train_folder, test_folder, tmp_path):
    (tmp_path / "alice" / "notes").mkdir()  # not a round: left out
    write_folder(af.round_folder(tmp_path, "alice", 10) / "roi_b")
    write_folder(af.round_folder(tmp_path, "bob", 2) / "roi_a")
    found = [(r.annotator, r.round, r.name) for r in af.rois(tmp_path)]
    assert found == [
        ("alice", 2, "test_image"),
        ("alice", 10, "roi_b"),  # by round number, not by name
        ("bob", 2, "roi_a"),
        ("carla", None, "patch_y500_x500"),
    ]
    assert af.rois(tmp_path)[0].folder == train_folder
    assert af.read_record(train_folder)["annotator"] == "alice"


def test_annotator_names_are_checked(tmp_path):
    assert af.round_folder(tmp_path, "anne-2", 3) == tmp_path / "anne-2" / "round_03"
    for name in ("Alice", "alice smith", "", "../bob"):
        with pytest.raises(ValueError):
            af.check_annotator(name)


def test_complete_and_closed(train_folder):
    record = af.read_record(train_folder)
    assert not af.is_complete(record) and not af.is_closed(record)  # never opened: no review yet
    masks = af.read_start(train_folder)
    cells = af.sync_cells({}, masks, masks)
    af.save(train_folder, masks, cells)
    assert not af.is_complete(af.read_record(train_folder))  # every cell still unreviewed
    for cell in cells.values():
        cell.state = af.OK
    af.save(train_folder, masks, cells)
    assert af.is_complete(af.read_record(train_folder))
    af.close(train_folder, date(2026, 11, 2))
    record = af.read_record(train_folder)
    assert af.is_closed(record) and record["closed_on"] == "2026-11-02"
    assert af.is_complete(record)  # closing keeps the review
