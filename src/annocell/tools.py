"""The editing tools, on the masks: modes, the mouse (outlines, delete, OK), and the edits they make.

New cell: right click (in any mode), or left click in draw mode (``N``), then move around the cell with no button
held: the outline follows the mouse, as in the Cellpose GUI. It closes when the mouse comes back to the start, or
with a second click; ``Esc`` drops it. Each outline is a new cell, which takes the pixels it covers from the cells
under it. Dragging pans, except while an outline is being drawn and in the delete and OK modes.

Modes (keys; the same key again, or ``Esc``, back to navigating):

- ``N``, draw: a left click starts a new cell;
- ``D``, delete: a click deletes the cell under it; dragging deletes every cell passed over (one undo). OK cells
  are kept: ``O`` first takes one back to not checked;
- ``O``, OK: a click marks the cell OK, or back to unreviewed if it was OK; dragging does the same to every cell
  passed over, in the direction decided by the first one.

``Shift+O`` marks every unreviewed cell entirely in view OK (``ok_by`` bulk; every other way to OK a cell is by hand).

``Cmd+Z`` (``Ctrl+Z`` on Windows) undoes the last step, ``Cmd+Shift+Z`` redoes it. A step is an outline, a delete, a
napari brush stroke, an OK stroke or a ``Shift+O``: its pixels (napari's own undo) and the cell rows it changed
(OK or not, how, macrophage) go back together, so ``Shift+O`` is undone in one go.

``Tools`` is a part of ``Annotator`` (``class Annotator(Tools)``): its methods use the annotator's state (``masks``,
``cells``, ``mode``, ``drawing``, ...) and call back into it (``changed``, ``widget``).
"""

from __future__ import annotations

from collections import deque
from copy import copy
from typing import Any

import numpy as np

from annocell import folder as af
from annocell.constants import CLOSE_SCREEN_PX, DELETE, DRAW, MACROPHAGE, MIN_CELL_PX, NAVIGATE, TOGGLE

# A step of the tool's undo: whether napari has a pixel step for it, and ``{cell_ID: (row before, row after)}`` (None:
# no row, the cell was not there).
Step = tuple[bool, dict[int, tuple[af.Cell | None, af.Cell | None]]]


class Tools:
    """Modes, mouse and edits of ``Annotator``."""

    # What these methods use of ``Annotator``, which sets it (see its docstring): declared here so that the editor
    # (Pylance) and type checkers know it. Annotations only: no value is set here.
    viewer: Any
    masks: Any  # the editable napari labels layer
    cells: dict[int, af.Cell]
    removed: dict[int, af.Cell]  # the rows of the cells deleted since the image was opened (``_sync_cells``)
    history: deque[Step]  # the steps to undo, the last one last
    future: list[Step]  # the steps undone, to redo
    start: np.ndarray
    snapshot: np.ndarray
    mode: str
    drawing: list | None
    outline: Any
    next_id: int
    read_only: bool
    test_mode: bool
    closed: bool
    is_open: bool
    widget: Any

    def changed(self) -> None:
        """Something to save; ``Annotator.changed`` replaces this."""
        raise NotImplementedError

    # Modes

    def set_mode(self, mode: str) -> None:
        """``navigate`` (napari as usual), ``draw``, ``delete`` or ``toggle`` (OK); dragging pans in navigate and draw
        (except while an outline is being drawn); in delete and OK it acts on the cells passed over. A read-only image
        stays in navigate."""
        mode = NAVIGATE if self.read_only else mode
        self.cancel_outline()  # sets mouse_pan back to True: set it after
        self.masks.mouse_pan = mode not in (DELETE, TOGGLE)
        self.mode = mode
        if mode != NAVIGATE:
            self.masks.mode = "pan_zoom"
            self.viewer.layers.selection.active = self.masks
        self.masks.cursor = {NAVIGATE: "standard", DRAW: "cross", DELETE: "forbidden", TOGGLE: "pointing", MACROPHAGE: "pointing"}[mode]
        self.widget.update_mode()

    def toggle_mode(self, mode: str) -> None:
        """The key of a mode turns it on, and off again."""
        self.set_mode(NAVIGATE if self.mode == mode else mode)

    def _on_napari_mode(self, event=None) -> None:
        """A napari tool (brush, eraser, fill, ...) picked from its buttons: back to navigating, so that a click does
        not both paint and delete or mark."""
        if self.masks.mode != "pan_zoom" and self.mode != NAVIGATE:
            self.set_mode(NAVIGATE)

    # Mouse: drawing and deleting

    def _on_mouse(self, viewer, event):
        """A right click starts an outline in any mode; a left click starts one in draw mode, closes the one being
        drawn, deletes (delete mode) or marks (OK mode) the cells pressed and dragged over."""
        # Buttons: 1 left, 2 right. Clicks on another layer (e.g. a channel being adjusted) are napari's.
        if not self.is_open or event.button not in (1, 2) or viewer.layers.selection.active is not self.masks:
            return
        if self.read_only:  # a closed round: look, do not touch (dragging still pans)
            return
        point = self._data_point(event.position)
        if self.drawing is not None:
            self.close_outline()
        elif event.button == 2 or self.mode == DRAW:
            self.start_outline(point)
        elif self.mode == DELETE:
            yield from self._delete_drag(event)
        elif self.mode == TOGGLE:
            yield from self._toggle_drag(event)
        elif self.mode == MACROPHAGE:
            self._macrophage_at(event)

    def start_outline(self, point) -> None:
        self.drawing = [self._inside(point)]
        self.masks.mouse_pan = False  # while drawing, a small drag must not move the image under the outline

    def _on_move(self, viewer, event) -> None:
        """The outline follows the mouse; it closes when the mouse comes back to its start."""
        if self.drawing is None:
            return
        point = self._inside(self._data_point(event.position))
        if np.abs(point - self.drawing[-1]).max() < 1:  # less than a pixel: not worth a point
            return
        self.drawing.append(point)
        # "Back to the start" is measured on screen (CLOSE_SCREEN_PX), whatever the zoom; the outline must first have
        # gone further away (3 times that), or it would close as soon as it starts.
        close = CLOSE_SCREEN_PX / self.viewer.camera.zoom  # in pixels of the image
        distances = np.abs(np.asarray(self.drawing) - self.drawing[0]).max(axis=1)
        if distances.max() > 3 * close and distances[-1] < close:
            self.close_outline()
        else:
            self._show_outline(self.drawing)

    def close_outline(self) -> int:
        """Fill the outline being drawn as a new cell; returns its ID (0: none, or too small)."""
        path, self.drawing = self.drawing, None
        self._show_outline([])
        self.masks.mouse_pan = True
        return 0 if path is None else self.draw_cell(np.asarray(path))

    def cancel_outline(self) -> None:
        self.drawing = None
        self._show_outline([])
        self.masks.mouse_pan = True

    def escape(self) -> None:
        """``Esc``: drop the outline being drawn, else back to navigating."""
        if self.drawing is not None:
            self.cancel_outline()
        else:
            self.set_mode(NAVIGATE)

    def _data_point(self, position) -> np.ndarray:
        """A mouse position (napari world coordinates) as (y, x) in pixels of the masks."""
        return np.asarray(self.masks.world_to_data(position)[-2:], dtype=float)

    def _inside(self, point) -> np.ndarray:
        """``point`` moved onto the image: an outline that leaves the image follows its edge (and the helper layer
        never grows beyond the image, which made napari recompute the scene at every move)."""
        return np.clip(np.asarray(point, dtype=float), 0, np.asarray(self.masks.data.shape) - 1)

    def _show_outline(self, path: list) -> None:
        """The outline being drawn, in a helper shapes layer on top."""
        if self.outline is None:
            if not path:
                return
            # Created on the first outline only, so the layer list stays short until then.
            self.outline = self.viewer.add_shapes(name="outline being drawn", edge_color="white", face_color=[0] * 4)
            self.viewer.layers.selection.active = self.masks
        self.outline.data = []
        if len(path) > 1:
            # The line width is in image pixels: about 2 screen pixels at any zoom.
            self.outline.add_paths(np.asarray(path), edge_width=max(1.0, 2 / self.viewer.camera.zoom))

    # Edits

    def draw_cell(self, path: np.ndarray) -> int:
        """Fill the closed outline ``path`` ((y, x) in pixels) as a new cell; returns its ID (0: too small)."""
        if len(path) < 3 or _polygon_area(path) < MIN_CELL_PX:
            return 0
        cell = self.next_id
        self.next_id += 1
        # napari fills the polygon over whatever is there (the cells under it lose those pixels) and records it
        # for undo; its ``paint`` event then updates the cell table (``_on_paint``).
        self.masks.paint_polygon(path, cell)
        if self.masks.selected_label == cell:  # napari's brush must not add to this cell: on to a new one
            self.masks.selected_label = self.next_id
        return cell

    def delete_at(self, point) -> None:
        """Delete the cell at ``point`` (undoable, like a napari edit); an OK cell is kept (O first, to delete it)."""
        cell = self._cell_at(point)
        if cell and not self.cells[cell].ok:
            self.masks.data_setitem(np.nonzero(self.masks.data == cell), 0)

    def _delete_drag(self, event):
        """Delete mode: delete the cell under the mouse on press, and every cell it passes over until release. A
        deleted cell is gone, so passing over its place again does nothing."""
        with self.masks.block_history():  # the whole stroke is one undo step (and one paint event)
            self.delete_at(self._data_point(event.position))
            yield
            while event.type == "mouse_move":
                self.delete_at(self._data_point(event.position))
                yield

    def _macrophage_at(self, event) -> None:
        '''Set the cell at ``point`` as a macrophage (only fot test images)'''
        cell = self._cell_at(self._data_point(event.position))
        self._set_macrophage(cell, MACROPHAGE)

    def _toggle_drag(self, event):
        """OK mode: the first cell decides the stroke. Not checked (or background): every cell passed over becomes
        OK; OK: every cell passed over goes back to not checked. A toggle per move event would flip a cell back and
        forth while the mouse crosses it; a fixed direction makes repeated events harmless."""
        before = self._rows()  # the whole stroke is one undo step
        first = self._cell_at(self._data_point(event.position))
        ok = not (first in self.cells and self.cells[first].ok)
        self._set_ok(first, ok)
        yield
        while event.type == "mouse_move":
            self._set_ok(self._cell_at(self._data_point(event.position)), ok)
            yield
        self._remember(before)

    def _set_ok(self, cell: int, ok: bool) -> None:
        if cell in self.cells and self.cells[cell].ok != ok:
            self.cells[cell].mark(ok, af.HAND)
            self.changed()
            
    def _set_macrophage(self, cell: int, isMacrophage: bool) -> None:
        if cell in self.cells:
            before = {cell: copy(self.cells[cell])}
            self.cells[cell].is_macrophage = not self.cells[cell].is_macrophage
            self._remember(before)
            self.changed()

    def all_in_view_ok(self) -> None:
        """``Shift+O``: every unreviewed cell entirely in view becomes OK, in bulk (cells cut by the edge of the view
        are left; cells already OK keep how they became OK)."""
        if self.read_only:
            return
        # ``corner_pixels``: the first and last visible pixel of the layer, (y, x), as napari last drew it.
        (y0, x0), (y1, x1) = np.asarray(self.masks.corner_pixels, dtype=int)[:, -2:]
        marked = [c for c in af.ids_inside(self.masks.data, y0, y1 + 1, x0, x1 + 1) if c in self.cells]
        marked = [c for c in marked if not self.cells[c].ok]
        before = {c: copy(self.cells[c]) for c in marked}
        for cell in marked:
            self.cells[cell].mark(True, af.BULK)
        if marked:  # nothing to mark is not a change (and does not take Done back)
            self._remember(before)  # one undo step for them all
            self.changed()

    def _cell_at(self, point) -> int:
        """The cell ID at ``point`` ((y, x) in pixels); 0 for background or outside the image."""
        y, x = np.round(point).astype(int)
        height, width = self.masks.data.shape
        return int(self.masks.data[y, x]) if 0 <= y < height and 0 <= x < width else 0

    def _on_paint(self) -> None:
        """After an outline, a delete, an undo or redo: the cells that gained pixels are OK (by hand), and so are
        those trimmed to background; cells that only lost pixels to a new cell keep their state, and a deleted cell
        brought back by undo is as it was (``_sync_cells``)."""
        if self.closed:
            return
        before = self._rows()
        data = self.masks.data
        diff = data != self.snapshot  # the pixels this edit changed
        gained = set(np.unique(data[diff]).tolist()) - {0}
        trimmed = set(np.unique(self.snapshot[diff & (data == 0)]).tolist()) - {0}
        back = self._sync_cells()
        # A cell back from an undo is as it was before its delete, not OK just because it gained pixels.
        for cell in (gained | trimmed) & (self.cells.keys() - back):
            self.cells[cell].mark(True, af.HAND)
        self.next_id = max(self.next_id, int(data.max()) + 1)
        self.snapshot = data.copy()
        self._remember(before, pixels=True)  # napari undoes the pixels, ``undo`` the rows they changed
        self.changed()

    def _sync_cells(self) -> set[int]:
        """Match the cell table to the masks (``af.sync_cells``: deleted cells leave it, new ones join it, every
        origin is recomputed), and remember the deleted ones: a cell that comes back (undo) gets its row back, OK or
        not, macrophage or not. IDs are never reused (``next_id`` only grows), so an ID that comes back is the same
        cell. Returns the IDs that came back."""
        data = self.masks.data
        present = set(np.unique(data).tolist())
        self.removed.update({i: cell for i, cell in self.cells.items() if i not in present})
        cells = af.sync_cells(self.cells, data, self.start)
        back = cells.keys() & self.removed.keys()
        for i in back:
            old = self.removed.pop(i)
            cells[i] = af.Cell(old.ok, cells[i].origin, old.ok_by, old.is_macrophage)
        self.cells = cells
        return back

    # Undo

    def _rows(self) -> dict[int, af.Cell]:
        """A copy of every row of the cell table, to tell later what a step changed (``_remember``)."""
        return {i: copy(cell) for i, cell in self.cells.items()}

    def _remember(self, before: dict[int, af.Cell], pixels: bool = False) -> None:
        """Add a step to undo: ``pixels``, napari added one to its own undo for this step; ``before``, the rows as
        they were (at least those that may have changed). A new step drops the steps undone (no redo after it)."""
        rows = {}
        for i in before.keys() | (self.cells.keys() if pixels else set()):
            old, new = before.get(i), self.cells.get(i)
            if old != new:
                rows[i] = (old, copy(new))
        if pixels or rows:
            self.history.append((pixels, rows))
            self.future.clear()

    def undo(self) -> None:
        """``Cmd+Z``: take the last step back, its pixels and the cell rows it changed."""
        self._replay(self.history, self.future, undoing=True)

    def redo(self) -> None:
        """``Cmd+Shift+Z``: do the last step undone again."""
        self._replay(self.future, self.history, undoing=False)

    def _replay(self, source, target, undoing: bool) -> None:
        if self.read_only or not source:
            return
        self.cancel_outline()
        pixels, rows = source.pop()
        target.append((pixels, rows))
        if pixels:
            # napari's own undo: it sends no paint event, so the table is matched to the masks here.
            (self.masks.undo if undoing else self.masks.redo)()
            self._sync_cells()
            self.snapshot = self.masks.data.copy()
        for i, (old, new) in rows.items():
            row = old if undoing else new
            if row is not None and i in self.cells:
                self.cells[i] = copy(row)
        self.changed()


def _polygon_area(path: np.ndarray) -> float:
    """Area of the polygon ``path`` ((y, x) points) in pixels (shoelace formula)."""
    y, x = path[:, 0], path[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
