"""The napari tool, headless: layers, modes, drawing, deleting, review states, saving."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from annocell import folder as af

napari = pytest.importorskip("napari")


def test_tool_corrects_marks_and_saves(train_folder, test_folder):
    from qtpy.QtWidgets import QApplication

    from annocell.app import DELETE, DRAW, NAVIGATE, TOGGLE, Annotator, folders_to_open

    folders, index = folders_to_open(train_folder.parent)
    assert folders == [train_folder] and index == 0
    viewer = napari.Viewer(show=False)
    try:
        tool = Annotator(viewer, folders, index)
        names = ["DAPI", "NaK", "F480", "AF", "masks (from cells)"]
        assert [layer.name for layer in viewer.layers] == names
        assert [layer.visible for layer in viewer.layers][:4] == [True, True, False, False]
        assert [layer.colormap.name for layer in viewer.layers][:4] == ["red", "green", "blue", "red"]  # in turn
        assert len(tool.cells) == 3 and all(c.state == af.UNREVIEWED for c in tool.cells.values())
        tool.set_compare(True)  # comparison masks only on request, below the masks
        assert [layer.name for layer in viewer.layers][-2:] == ["model_a (trained on this image)", "masks (from cells)"]
        tool.set_compare(False)
        assert len(viewer.layers) == 5 and viewer.layers.selection.active is tool.masks

        tool.toggle_mode(TOGGLE)  # O: OK mode
        assert tool.mode == TOGGLE and not tool.masks.mouse_pan and tool.masks.cursor == "pointing"
        _click(tool, viewer, 1, (5, 5))  # cell 1
        assert tool.cells[1].state == af.OK
        tool.toggle_mode(TOGGLE)  # O again: back to navigating
        assert tool.mode == NAVIGATE and tool.masks.mouse_pan

        tool.toggle_mode(DELETE)
        assert tool.mode == DELETE and not tool.masks.mouse_pan  # in delete mode, dragging deletes
        _click(tool, viewer, 1, (15, 5))  # cell 3
        QApplication.processEvents()  # the tool reads the new pixels after the paint event
        assert 3 not in tool.cells and not (tool.masks.data == 3).any()
        tool.masks.mode = "paint"  # napari's brush (key P) is switched off at once
        assert tool.masks.mode == "pan_zoom" and tool.mode == DELETE
        assert viewer.window._qt_viewer.controls.widgets[tool.masks].paint_button.isHidden()
        tool.set_mode(NAVIGATE)
        assert tool.masks.mouse_pan

        tool.masks.data_setitem((np.array([0, 1]), np.array([12, 12])), 0)  # two pixels of cell 2 to background
        QApplication.processEvents()
        assert tool.cells[2].state == af.OK and tool.cells[2].origin == af.EDITED

        tool.toggle_mode(DRAW)
        assert tool.mode == DRAW and tool.masks.mode == "pan_zoom"
        assert tool.draw_cell(np.array([[12.0, 20], [12, 21], [13, 21]])) == 0  # too small: a slip
        cell = tool.draw_cell(np.array([[12.0, 20], [12, 29], [19, 29], [19, 20]]))
        QApplication.processEvents()
        assert cell == 4 and tool.cells[4] == af.Cell(af.OK, af.DRAWN, False)
        assert tool.draw_cell(np.array([[12.0, 0], [12, 9], [19, 9], [19, 0]])) == 5  # each outline a new cell
        QApplication.processEvents()
        # Left click in draw mode, move (no button held), back to the start: the outline closes by itself.
        _trace(tool, viewer, 1, [(2, 22), (2, 28), (8, 28), (8, 22), (2, 22)])
        QApplication.processEvents()
        assert (tool.masks.data[4:7, 24:27] == 6).all() and tool.drawing is None and tool.outline.data == []
        tool.set_mode(NAVIGATE)
        _click(tool, viewer, 1, (15, 5))  # a left click while navigating does nothing
        assert tool.drawing is None
        # Right click starts an outline in any mode; Esc drops it, a second click closes it.
        _trace(tool, viewer, 2, [(12, 12), (12, 18), (18, 18)])
        assert tool.drawing is not None and tool.masks.mode == "pan_zoom" and not tool.masks.mouse_pan
        viewer.window._qt_viewer._key_map_handler.press_key("Escape")  # napari's key dispatch, masks layer active
        assert tool.drawing is None and tool.masks.mouse_pan and tool.mode == NAVIGATE
        _trace(tool, viewer, 2, [(5, 5), (-40, 5), (5, 90)])  # outside the image: the outline follows its edge
        assert np.asarray(tool.drawing).min() == 0 and np.asarray(tool.drawing)[:, 1].max() == 29
        tool.escape()
        _trace(tool, viewer, 2, [(12, 12), (12, 18), (18, 18), (18, 12)])
        _click(tool, viewer, 1, (18, 12))
        QApplication.processEvents()
        assert (tool.masks.data[14:17, 14:17] == 7).all()
        tool.save()
        review = af.read_record(train_folder)["review"]
        assert (review["cells"], review[af.OK], review[af.UNREVIEWED]) == (6, 6, 0)
        assert (
            review[af.DRAWN] == 4 and review[af.EDITED] == 1 and set(af.read_cells(train_folder)) == {1, 2, 4, 5, 6, 7}
        )

        tool.cells[1].state = af.UNREVIEWED
        viewer.reset_view()
        tool.all_in_view_ok()  # the whole image is in view
        assert tool.cells[1].state == af.OK

        tool.folders = [train_folder, test_folder]
        tool.go(1)
        assert tool.test_mode and tool.masks.name == "masks (from scratch)" and tool.masks.data.max() == 0
        assert viewer.layers["DAPI"].colormap.name == "blue"  # colours from annotation.json
        tool.set_compare(True)
        assert "model_a" not in [layer.name for layer in viewer.layers]  # no masks shown in test mode
        tool.draw_cell(np.array([[2.0, 2], [2, 8], [8, 8], [8, 2]]))
        QApplication.processEvents()
        viewer.cursor.position = (5, 5)
        tool.toggle_macrophage()
        tool.go(-1)  # saves the test patch
        assert af.read_cells(test_folder) == {1: af.Cell(af.OK, af.DRAWN, True)}
    finally:
        viewer.close()


def _click(tool, viewer, button, point):
    """Press and release: ``_on_mouse`` is a generator (napari resumes it on every move and on release), so step
    it on press and again on release (a delete stroke is committed, as one undo step, on release)."""
    viewer.cursor.position = point
    event = SimpleNamespace(button=button, type="mouse_press", position=point)
    gen = tool._on_mouse(viewer, event)
    next(gen, None)
    event.type = "mouse_release"
    next(gen, None)


def _drag(tool, viewer, points):
    """Left press at the first point, move (button held) through the others, release."""
    event = SimpleNamespace(button=1, type="mouse_press", position=points[0])
    gen = tool._on_mouse(viewer, event)
    next(gen, None)
    for point in points[1:]:
        event.type, event.position = "mouse_move", point
        next(gen, None)
    event.type = "mouse_release"
    next(gen, None)


def _trace(tool, viewer, button, points):
    """Click at the first point, then move the mouse (no button held) through the others."""
    _click(tool, viewer, button, points[0])
    for point in points[1:]:
        tool._on_move(viewer, SimpleNamespace(button=None, type="mouse_move", position=point))


def test_delete_mode_drag_deletes_every_cell_passed_over(train_folder):
    from qtpy.QtWidgets import QApplication

    from annocell.app import DELETE, Annotator

    viewer = napari.Viewer(show=False)
    try:
        tool = Annotator(viewer, [train_folder], 0)
        tool.toggle_mode(DELETE)
        _drag(tool, viewer, [(5, 5), (5, 12), (5, 15)])  # across cells 1 and 2
        QApplication.processEvents()
        assert not np.isin(tool.masks.data, [1, 2]).any() and (tool.masks.data == 3).any()
        assert set(tool.cells) == {3} and not tool.masks.mouse_pan  # dragging deletes, it does not pan
        tool.masks.undo()  # the whole stroke is one undo step
        QApplication.processEvents()
        assert np.isin(tool.masks.data, [1, 2]).any()
    finally:
        viewer.close()


def test_ok_mode_drag_marks_in_the_direction_of_the_first_cell(train_folder):
    from annocell.app import TOGGLE, Annotator

    viewer = napari.Viewer(show=False)
    try:
        tool = Annotator(viewer, [train_folder], 0)
        tool.toggle_mode(TOGGLE)
        # From cell 1 (not checked) across cell 2, with several move events over each: both OK, none flipped back.
        _drag(tool, viewer, [(5, 5), (5, 7), (5, 9), (5, 12), (5, 14), (5, 16)])
        assert tool.cells[1].state == tool.cells[2].state == af.OK and tool.cells[3].state == af.UNREVIEWED
        _drag(tool, viewer, [(5, 15), (5, 12), (5, 5)])  # starting on an OK cell: back to not checked
        assert tool.cells[1].state == tool.cells[2].state == af.UNREVIEWED
        _drag(tool, viewer, [(11, 25), (15, 5)])  # starting on background: marks OK
        assert tool.cells[3].state == af.OK
    finally:
        viewer.close()


def test_tool_refuses_folders_in_another_layout(tmp_path):
    from annocell.app import folders_to_open

    folder = tmp_path / "round_01" / "image"
    folder.mkdir(parents=True)
    (folder / af.RECORD).write_text("{}")  # e.g. a folder made for the Cellpose GUI: no image.tif
    for path in (folder, folder.parent):
        with pytest.raises(FileNotFoundError, match="AnnoCell layout"):
            folders_to_open(path)


def test_starts_empty_and_opens_a_folder(train_folder, test_folder, tmp_path):
    from annocell.app import TOGGLE, Annotator

    viewer = napari.Viewer(show=False)
    try:
        tool = Annotator(viewer)  # annocell without a folder
        assert not tool.is_open and len(viewer.layers) == 0
        viewer.window._qt_viewer._key_map_handler.press_key("o")  # keys, mouse and autosave do nothing yet
        _click(tool, viewer, 2, (5, 5))
        tool.autosave()
        tool.go(1)
        assert not tool.is_open and tool.drawing is None

        (tmp_path / "elsewhere").mkdir()
        tool.open_folder(tmp_path / "elsewhere")  # not an AnnoCell folder: refused with a message
        assert not tool.is_open

        tool.open_folder(train_folder.parent)  # a round folder
        assert tool.is_open and tool.folder == train_folder and tool.masks.name == "masks (from cells)"
        tool.toggle_mode(TOGGLE)
        _click(tool, viewer, 1, (5, 5))
        tool.open_folder(test_folder)  # one image folder: the image open so far is saved first
        assert af.read_cells(train_folder)[1].state == af.OK
        assert tool.folder == test_folder and tool.test_mode
    finally:
        viewer.close()


def test_annotator_folder_opens_the_latest_open_round(tmp_path):
    from annocell.app import folders_to_open
    from conftest import write_folder

    alice = tmp_path / "alice"
    for round_number, name in ((1, "a"), (2, "b"), (2, "c"), (10, "d")):
        write_folder(af.round_folder(tmp_path, "alice", round_number) / name)
    af.close(alice / "round_01" / "a")
    af.close(alice / "round_10" / "d")
    folders, index = folders_to_open(alice)  # round 10 is closed: round 2, its first image with cells to check
    assert [f.name for f in folders] == ["b", "c"] and index == 0
    af.close(alice / "round_02" / "b")
    af.close(alice / "round_02" / "c")
    assert folders_to_open(alice)[0] == [alice / "round_10" / "d"]  # all closed: the latest round, read-only
    carla = write_folder(af.test_regions_folder(tmp_path, "carla") / "patch", test=True)
    assert folders_to_open(tmp_path / "carla")[0] == [carla]  # only test regions
    with pytest.raises(FileNotFoundError, match="your own folder"):
        folders_to_open(tmp_path)  # the shared root: several annotators, nothing to pick


def test_closed_roi_is_read_only(train_folder):
    from annocell.app import DELETE, NAVIGATE, Annotator

    af.close(train_folder)
    viewer = napari.Viewer(show=False)
    try:
        tool = Annotator(viewer, [train_folder], 0)
        assert tool.read_only and not tool.masks.editable
        assert "alice, round 2" in _panel(tool) and "Round 2 closed: read-only" in _panel(tool)
        tool.toggle_mode(DELETE)
        assert tool.mode == NAVIGATE  # no editing mode
        _click(tool, viewer, 2, (5, 5))  # no outline
        assert tool.drawing is None
        viewer.reset_view()
        tool.all_in_view_ok()
        assert all(c.state == af.UNREVIEWED for c in tool.cells.values())
        assert tool.save() and not (train_folder / af.MASKS).exists()  # never written
    finally:
        viewer.close()


def test_failed_save_keeps_the_changes_and_tries_again(train_folder, test_folder, tmp_path, monkeypatch):
    from annocell.app import TOGGLE, Annotator

    viewer = napari.Viewer(show=False)
    try:
        tool = Annotator(viewer, [train_folder, test_folder], 0)
        tool.toggle_mode(TOGGLE)
        _click(tool, viewer, 1, (5, 5))  # cell 1 OK
        save = af.save

        def unreachable(folder, *args):  # the image folder's drive is down; the home folder is fine
            if Path(folder) == train_folder:
                raise OSError("network drive unreachable")
            save(folder, *args)

        monkeypatch.setattr(af, "save", unreachable)
        assert not tool.save() and tool.dirty and "Not saved" in _panel(tool)
        tool.go(1)  # not saved: the image stays open
        assert tool.folder == train_folder and tool.cells[1].state == af.OK
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        backup = tool.save_backup()  # what closing napari does when the drive is still unreachable
        assert backup == tmp_path / "home" / "AnnoCell unsaved" / "alice_round_02_test_image"
        monkeypatch.setattr(af, "save", save)
        assert af.read_cells(backup)[1].state == af.OK
        tool.autosave()  # the drive is back
        assert not tool.dirty and tool.save_error is None and "Not saved" not in _panel(tool)
        assert af.read_cells(train_folder)[1].state == af.OK
    finally:
        viewer.close()


def _panel(tool) -> str:
    """Every text of the side panel."""
    from qtpy.QtWidgets import QLabel

    return " ".join(label.text() for label in tool.widget.findChildren(QLabel))
