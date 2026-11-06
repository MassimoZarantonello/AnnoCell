"""The editing tools, on the masks: modes, the mouse (outlines, delete, OK), and the edits they make.

New cell: right click (in any mode), or left click in draw mode (``N``), then move around the cell with no button
held: the outline follows the mouse, as in the Cellpose GUI. It closes when the mouse comes back to the start, or
with a second click; ``Esc`` drops it. Each outline is a new cell, which takes the pixels it covers from the cells
under it. Dragging pans, except while an outline is being drawn and in the delete and OK modes.

Modes (keys; the same key again, or ``Esc``, back to navigating):

- ``N``, draw: a left click starts a new cell;
- ``D``, delete: a click deletes the cell under it; dragging deletes every cell passed over (one undo);
- ``O``, OK: a click marks the cell OK, or back to unreviewed if it was OK; dragging does the same to every cell
  passed over, in the direction decided by the first one.

``Shift+O`` marks every unreviewed cell entirely in view OK (``ok_by`` bulk; every other way to OK a cell is by hand).

``Tools`` is a part of ``Annotator`` (``class Annotator(Tools)``): its methods use the annotator's state (``masks``,
``cells``, ``mode``, ``drawing``, ...) and call back into it (``changed``, ``widget``).
"""

from __future__ import annotations

import numpy as np

from annocell import folder as af
from annocell.constants import CLOSE_SCREEN_PX, DELETE, DRAW, MIN_CELL_PX, NAVIGATE, TOGGLE


class Tools:
    """Modes, mouse and edits of ``Annotator``."""

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
        self.masks.cursor = {NAVIGATE: "standard", DRAW: "cross", DELETE: "forbidden", TOGGLE: "pointing"}[mode]
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
            yield from self._ok_drag(event)

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
        """Delete the cell at ``point`` (undoable, like a napari edit)."""
        cell = self._cell_at(point)
        if cell:
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

    def _ok_drag(self, event):
        """OK mode: the first cell decides the stroke. Not checked (or background): every cell passed over becomes
        OK; OK: every cell passed over goes back to not checked. A toggle per move event would flip a cell back and
        forth while the mouse crosses it; a fixed direction makes repeated events harmless."""
        first = self._cell_at(self._data_point(event.position))
        state = af.UNREVIEWED if first in self.cells and self.cells[first].state == af.OK else af.OK
        self._set_state(first, state)
        yield
        while event.type == "mouse_move":
            self._set_state(self._cell_at(self._data_point(event.position)), state)
            yield

    def _set_state(self, cell: int, state: str) -> None:
        # Review states are not pixels: not in napari's undo history (O again undoes a mark).
        if cell in self.cells and self.cells[cell].state != state:
            self.cells[cell].mark(state, af.HAND)
            self.changed()

    def all_in_view_ok(self) -> None:
        """``Shift+O``: every unreviewed cell entirely in view becomes OK, in bulk (cells cut by the edge of the view
        are left; cells already OK keep how they became OK)."""
        if self.read_only:
            return
        # ``corner_pixels``: the first and last visible pixel of the layer, (y, x), as napari last drew it.
        (y0, x0), (y1, x1) = np.asarray(self.masks.corner_pixels, dtype=int)[:, -2:]
        marked = [c for c in af.ids_inside(self.masks.data, y0, y1 + 1, x0, x1 + 1) if c in self.cells]
        marked = [c for c in marked if self.cells[c].state == af.UNREVIEWED]
        for cell in marked:
            self.cells[cell].mark(af.OK, af.BULK)
        if marked:  # nothing to mark is not a change (and does not take Done back)
            self.changed()

    def _cell_at(self, point) -> int:
        """The cell ID at ``point`` ((y, x) in pixels); 0 for background or outside the image."""
        y, x = np.round(point).astype(int)
        height, width = self.masks.data.shape
        return int(self.masks.data[y, x]) if 0 <= y < height and 0 <= x < width else 0

    def _on_paint(self) -> None:
        """After an outline, a delete, an undo or redo: the cells that gained pixels are OK (by hand), and so are
        those trimmed to background; cells that only lost pixels to a new cell keep their state."""
        if self.closed:
            return
        data = self.masks.data
        diff = data != self.snapshot  # the pixels this edit changed
        gained = set(np.unique(data[diff]).tolist()) - {0}
        trimmed = set(np.unique(self.snapshot[diff & (data == 0)]).tolist()) - {0}
        # Deleted cells leave the table, new ones join it, and every origin is recomputed against the start masks.
        self.cells = af.sync_cells(self.cells, data, self.start)
        for cell in (gained | trimmed) & self.cells.keys():
            self.cells[cell].mark(af.OK, af.HAND)
        self.next_id = max(self.next_id, int(data.max()) + 1)
        self.snapshot = data.copy()
        self.changed()


def _polygon_area(path: np.ndarray) -> float:
    """Area of the polygon ``path`` ((y, x) points) in pixels (shoelace formula)."""
    y, x = path[:, 0], path[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
