"""The napari annotation tool: ``annocell <folder>``.

``<folder>`` is the annotator's own folder (the tool opens their latest round that is not closed), a round (or
test) folder, or one image folder; the tool opens the first image not done yet and moves through the
others of the same round with Previous / Next. Without ``<folder>`` it opens empty, and the panel's "Open folder"
button picks one. An image of a closed round (``closed_on``) opens read-only: it can be looked at, not changed; a
round closed while one of its images is open turns it read-only at the next save.

Layers: every channel of the image (napari's own controls) and the editable masks (``masks (from <segmentation>)``),
outlined in yellow (unreviewed) or green (OK). Read-only comparison masks of other segmentations are added to the
layer list only on request (a checkbox), hidden.

New cell: right click (in any mode), or left click in draw mode (``N``), then move around the cell with no button
held: the outline follows the mouse, as in the Cellpose GUI. It closes when the mouse comes back to the start, or
with a second click; ``Esc`` drops it. Each outline is a new cell, which takes the pixels it covers from the cells
under it. Dragging pans, except while an outline is being drawn and in the delete and OK modes.

Modes (keys; the same key again, or ``Esc``, back to navigating):

- ``N``, draw: a left click starts a new cell;
- ``D``, delete: a click deletes the cell under it; dragging deletes every cell passed over (one undo);
- ``O``, OK: a click marks the cell OK, or back to unreviewed if it was OK; dragging does the same to every cell
  passed over, in the direction decided by the first one.

Keys on the cell under the mouse:

- ``Shift+O``: mark every unreviewed cell entirely in view OK (``ok_by`` bulk; every other way to OK a cell is by
  hand).

napari's own editing tools (brush, eraser, fill, polygon, picker) are there, as buttons only: their keys (P, E, F,
L, 1 to 7, M, ...) do nothing (``NAPARI_KEYS``), so a key pressed by mistake never changes the masks. Picking one of
them leaves the tool's mode.

**Done**: a button. The annotator presses it when every cell is OK; with cells still not checked, a message says
how many, and nothing is marked. Any change after Done takes it back: press it again. "Image >" goes on only from
an image done (no skipping); closing napari or opening another folder with an image not done asks first.

The masks are saved every few seconds, when changing image and when closing napari. The folders may be on a
network drive: a save that fails (drive unreachable) keeps everything in memory, says so in the panel, and is tried
again at the next autosave; the tool does not change image until it has saved. If it still fails when napari closes,
a copy goes to ``AnnoCell unsaved`` in the home folder.

How it fits together: ``Annotator`` holds the state (current image, masks layer, cell table) and reacts to napari's
mouse, key and paint events; ``_Widget`` is the side panel; ``annocell.folder`` reads and writes the files.
Qt and napari are imported inside functions, so that this module (and its tests' helpers) import without a display.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from annocell import folder as af

# Outline colours (RGBA) by review state.
COLOURS = {
    af.UNREVIEWED: (1.0, 0.85, 0.0, 1.0),
    af.OK: (0.1, 0.9, 0.3, 1.0),
}
WARNING = (1.0, 0.2, 1.0, 1.0)  # "Not saved" in the panel
CHANNEL_COLORMAPS = ("red", "green", "blue")  # channels without a colour in annotation.json, in turn
SAVE_EVERY_MS = 3000
MIN_CELL_PX = 10  # a smaller outline is a slip, not a cell
NAVIGATE, DRAW, DELETE, TOGGLE = "navigate", "draw", "delete", "toggle"
CLOSE_SCREEN_PX = 8  # an outline closes when the mouse comes back this close to its start
# napari's keys on a labels layer (napari 0.9: tools, label IDs, brush size), switched off on the masks layer; undo,
# redo, Escape (bound by the tool) and Enter (closes napari's polygon) stay.
NAPARI_KEYS = ("1", "2", "3", "4", "5", "6", "7", "E", "P", "F", "L", "Z", "M", "X", "B", "-", "=", "[", "]")


class Annotator:
    """The state of one napari session: the images of a round, the current one's masks and cell table.

    State of the current image:

    - ``masks``: the editable labels layer; its ``data`` is the current masks;
    - ``start``: the proposed masks (all background in test mode), to tell proposed, edited and drawn cells apart;
    - ``cells``: ``{cell_ID: Cell}``, the review state of every cell in ``masks``;
    - ``snapshot``: the masks after the last handled edit, to find what the next edit changed;
    - ``saved``: the masks as last written to disk, and ``dirty``: a review state changed since then;
    - ``next_id``: the ID of the next drawn cell, never reusing an ID of the start masks;
    - ``mode``: navigate, draw, delete or OK; ``drawing``: the outline being drawn, or None;
    - ``read_only``: the image's round is closed (when opened, or since): nothing can be changed or saved;
    - ``done``: the annotator pressed Done (every cell OK) and changed nothing since;
    - ``save_error``: why the last save failed (e.g. the network drive unreachable), or None.

    With no folder open (``folders`` empty), ``masks`` is None and the keys, buttons, mouse and autosave do nothing.
    """

    def __init__(self, viewer, folders: list[Path] | None = None, index: int = 0):
        from qtpy.QtCore import QTimer

        self.viewer = viewer
        self.folders = list(folders or [])
        self.masks = None  # no image open yet
        self.folder, self.index, self.cells, self.test_mode = None, 0, {}, False
        self.record, self.read_only, self.save_error, self.done = {}, False, None, False
        self.mode = NAVIGATE
        self.drawing = None  # the outline being drawn, (y, x) points
        self.show_compare = False  # kept when moving to the next image
        self.widget = _Widget(self)
        viewer.window.add_dock_widget(self.widget, name="Annotation", area="right")
        self._bind_keys()
        # Closing napari with an image not done asks first (kept here: Qt does not keep the filter alive).
        self._close_guard = _close_guard(self)
        if (window := self._window()) is not None:
            window.installEventFilter(self._close_guard)
        # On the viewer, not on a layer: they run whatever layer is selected, before the layer's own callbacks.
        viewer.mouse_drag_callbacks.append(self._on_mouse)
        viewer.mouse_move_callbacks.append(self._on_move)
        self.timer = QTimer()
        self.timer.timeout.connect(self.autosave)
        self.timer.start(SAVE_EVERY_MS)
        # napari closed: its panel is destroyed. No autosave after that, and a paint handler still queued does nothing
        # (it would update labels that no longer exist).
        self.closed = False
        self.widget.destroyed.connect(self._on_closed)
        if self.folders:
            self.load(index)
        else:
            self.widget.update_status()
            self.widget.update_mode()

    @property
    def is_open(self) -> bool:
        return self.masks is not None

    def open_folder(self, path: str | Path) -> None:
        """Open a round (or test) folder, or one image folder, as ``annocell <folder>`` does; the image open so far
        is saved first. A folder in another layout is refused with a message, and the current image stays."""
        try:
            folders, index = folders_to_open(path)
        except FileNotFoundError as error:
            from napari.utils.notifications import show_error

            show_error(str(error))
            return
        if not self.may_leave("Open the other folder anyway?"):
            return
        if self.is_open and not self.save():
            _cannot_leave()
            return
        self.folders = folders
        self.load(index)

    # Images

    def load(self, index: int) -> None:
        """Show image ``index`` of the round: its channels and masks, with the review saved so far."""
        from qtpy.QtCore import QTimer

        self.index = index
        self.folder = self.folders[index]
        self.record = af.read_record(self.folder)
        self.test_mode = af.is_test(self.record)
        self.read_only = af.is_closed(self.record)
        image, names = af.read_image(self.folder)
        shape = image.shape[1:]
        self.start = af.read_start(self.folder, shape)
        masks = af.read_masks(self.folder, shape)  # the saved corrections, else the proposal
        self.cells = af.sync_cells(af.read_cells(self.folder), masks, self.start)
        self.done = af.is_complete(af.read_review(self.folder))
        self.next_id = int(max(masks.max(), self.start.max())) + 1

        self.viewer.layers.clear()
        self.outline = None
        self.compare = []
        # One image layer per channel, added together (additive blending); contrast from the 0.5 and 99.8
        # percentiles, so that a few very bright pixels do not leave the rest dark.
        visible = set(self.record.get("visible_channels") or names[:3])
        colours = self.record.get("colours") or {}
        for i, (name, channel) in enumerate(zip(names, image, strict=True)):
            low, high = np.percentile(channel, (0.5, 99.8))
            self.viewer.add_image(
                channel,
                name=name,
                colormap=colours.get(name) or CHANNEL_COLORMAPS[i % len(CHANNEL_COLORMAPS)],
                blending="additive",
                contrast_limits=(float(low), float(max(high, low + 1))),
                visible=name in visible,
            )
        source = (self.record.get("start_segmentation") or {}).get("labels")
        name = "masks (from scratch)" if self.test_mode else f"masks (from {source})" if source else "masks"
        self.masks = self.viewer.add_labels(masks, name=name, opacity=1.0)
        self.masks.editable = not self.read_only
        self.masks.contour = 1  # outlines only, so the channels stay visible inside the cells
        self.masks.show_selected_label = False
        self.masks.selected_label = self.next_id  # napari's brush starts a new cell, not cell 1
        # napari emits ``paint`` before writing the new pixels: look at them once the event is over.
        self.masks.events.paint.connect(lambda event: QTimer.singleShot(0, self._on_paint))
        self.masks.events.mode.connect(self._on_napari_mode)
        # Keys bound on the layer win over napari's own (its class keymap): no-ops, and our Esc. napari's own Esc on
        # a labels layer (cancel its polygon) would also win over the viewer's.
        for key in NAPARI_KEYS:
            self.masks.bind_key(key, lambda layer: None, overwrite=True)
        self.masks.bind_key("Escape", lambda layer: self.escape(), overwrite=True)
        self.set_compare(self.show_compare)
        self.set_mode(self.mode)
        self.viewer.title = f"AnnoCell: {self.folder.name}"
        self.snapshot = masks.copy()
        self.saved = masks.copy()
        self.dirty = False
        self.viewer.reset_view()
        self.refresh()

    def go(self, step: int) -> None:
        """Save, then open the previous (``-1``) or next (``1``) image; nothing past the first or last. The next image
        only once this one is done: with cells not OK, Done's message (how many are left); with every cell OK but
        Done not pressed, a warning to press it. A read-only image does not stop anyone, nor does going back."""
        index = self.index + step
        if not (self.is_open and 0 <= index < len(self.folders)):
            return
        if step > 0 and not self.read_only and not self.done:
            problem = self._not_finished()
            self._tell(
                problem.replace("press Done", "press Done and go to the next image")
                if problem
                else "This image is not marked as done: press Done, then go to the next image."
            )
            return
        if not self.save():
            _cannot_leave()
            return
        self.load(index)

    def set_compare(self, show: bool) -> None:
        """Add the comparison masks to the layer list (hidden, below the masks), or remove them."""
        self.show_compare = show
        if not self.is_open:  # remembered for the first image opened
            return
        for layer in self.compare:
            self.viewer.layers.remove(layer)
        self.compare = []
        if show and not self.test_mode:  # test images are drawn from scratch: no other masks, ever
            for name, labels, trained in af.compare_layers(self.folder):
                layer = self.viewer.add_labels(
                    labels, name=f"{name} (trained on this image)" if trained else name, visible=False, opacity=0.8
                )
                layer.contour = 1
                layer.locked = True  # read-only
                self.viewer.layers.move(self.viewer.layers.index(layer), self.viewer.layers.index(self.masks))
                self.compare.append(layer)
        # Adding a layer selects it; keys and clicks act on the selected layer, which must stay the masks.
        self.viewer.layers.selection.active = self.masks

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

    def press_done(self) -> None:
        """The Done button: the image is finished, if every cell is OK; else a message says how many are left."""
        if self.read_only:
            return
        if problem := self._not_finished():
            self._tell(problem)
            return
        self.done, self.dirty = True, True
        self.save()
        self.widget.update_status()

    def may_leave(self, question: str) -> bool:
        """Before closing napari or opening another folder: an image not done asks ``question`` first (it would stay
        out of training, e.g. Done taken back by a late fix and not pressed again). True: go ahead."""
        if not self.is_open or self.read_only or self.done:
            return True
        return _ask(
            f"This image is not marked as done: it stays out of training until you press Done. {question}",
            self._window(),
        )

    def _tell(self, text: str) -> None:
        _message(text, self._window())

    def _window(self):
        """napari's main window (napari internals: None if that changes), so that messages open on top of it."""
        return getattr(self.viewer.window, "_qt_window", None)

    def _not_finished(self) -> str | None:
        """Why this image is not finished (no cells, or cells not OK), for the annotator; None if every cell is OK."""
        counts = af.summary(self.cells)
        if not counts["cells"]:
            return "There are no cells yet: draw them, then press Done."
        left = counts[af.UNREVIEWED]
        if left:
            return (
                f"There {'is' if left == 1 else 'are'} still {left} cell{'' if left == 1 else 's'} not OK. Mark "
                "them OK (or delete them), then press Done; Next unchecked cell takes you to them."
            )
        return None

    def next_unreviewed(self) -> None:
        """Centre the view on the unreviewed cell closest to the current centre."""
        from scipy import ndimage

        ids = [i for i, c in self.cells.items() if c.state == af.UNREVIEWED]
        if not ids:
            return
        centres = np.asarray(ndimage.center_of_mass(np.ones_like(self.masks.data), self.masks.data, ids))
        here = np.asarray(self.viewer.camera.center[-2:])
        self.viewer.camera.center = tuple(centres[np.argmin(((centres - here) ** 2).sum(axis=1))])

    def _cell_at(self, point) -> int:
        """The cell ID at ``point`` ((y, x) in pixels); 0 for background or outside the image."""
        y, x = np.round(point).astype(int)
        height, width = self.masks.data.shape
        return int(self.masks.data[y, x]) if 0 <= y < height and 0 <= x < width else 0

    def _on_closed(self, *args) -> None:
        self.closed = True
        self.timer.stop()

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

    # Display and saving

    def changed(self) -> None:
        """Something to save: recolour the outlines and update the panel. It takes Done back."""
        self.dirty, self.done = True, False
        self.refresh()

    def refresh(self) -> None:
        """Colour every outline by its review state, and update the counts in the panel."""
        from napari.utils.colormaps import DirectLabelColormap

        # One colour per cell ID; ``None`` is the colour of IDs not in the table (none, normally), 0 the background.
        colours = {None: COLOURS[af.UNREVIEWED], 0: (0, 0, 0, 0)}
        for i, cell in self.cells.items():
            colours[i] = COLOURS[cell.state]
        self.masks.colormap = DirectLabelColormap(color_dict=colours)
        self.widget.update_status()

    def autosave(self) -> None:
        """Every ``SAVE_EVERY_MS``: save if anything changed since the last save."""
        if not self.is_open:
            return
        data = self.masks.data
        if not np.array_equal(data, self.snapshot):  # undo / redo leave no paint event
            self.cells = af.sync_cells(self.cells, data, self.start)
            self.snapshot = data.copy()
            self.changed()
        if self.dirty or not np.array_equal(data, self.saved):
            self.save()

    def save(self) -> bool:
        """Write the masks, the cell table and the review summary of the current image (``folder.save``); True if
        written (or nothing to write: no image open, or a read-only image, which is never written).

        If the write fails (e.g. the network drive is unreachable), everything stays in memory and ``dirty``, the
        panel says so (one warning, not one per autosave), and the next autosave tries again; returns False.

        If the round was closed since the image was opened (``closed_on`` on disk), nothing is written: the image
        turns read-only and the panel says so (its masks may already be in a training). Returns True, so the
        annotator can move on.
        """
        from napari.utils.notifications import show_info, show_warning

        if not self.is_open or self.read_only:
            return True
        data = self.masks.data
        self.cells = af.sync_cells(self.cells, data, self.start)
        try:
            record = af.read_record(self.folder)  # one small file, every few seconds: cheap
            if af.is_closed(record):
                self._closed_meanwhile(record)
                return True
            af.save(self.folder, data, self.cells, self.start, self.done)
        except OSError as error:
            if self.save_error is None:
                show_warning(f"Not saved ({error}). Your changes are kept; trying again every few seconds.")
            self.save_error = str(error)
            self.dirty = True
            self.widget.update_status()
            return False
        if self.save_error is not None:
            show_info("Saved again.")
            self.save_error = None
        self.saved = data.copy()
        self.dirty = False
        self.widget.update_status()
        return True

    def _closed_meanwhile(self, record: dict) -> None:
        """The round was closed while this image was open: read-only from now on, the last changes not saved."""
        from napari.utils.notifications import show_warning

        self.record, self.read_only, self.dirty, self.save_error = record, True, False, None
        self.masks.editable = False
        self.set_mode(NAVIGATE)  # read-only: also drops an outline being drawn
        show_warning("This round was closed: the image is now read-only; changes since the last save are not saved.")
        self.widget.update_status()

    def save_backup(self) -> Path:
        """When the image folder cannot be written as napari closes: the masks, the cell table and the record go to
        ``~/AnnoCell unsaved/<annotator>_<round>_<image>/``, to be copied back into the image folder."""
        folder = Path.home() / "AnnoCell unsaved" / "_".join(self.folder.parts[-3:])
        af.write_record(folder, self.record)
        af.save(folder, self.masks.data, self.cells, self.start, self.done)
        return folder

    def _bind_keys(self) -> None:
        """The tool's keys, on the viewer (they work whatever layer is selected; Esc is also bound on the masks
        layer in ``load``, where napari's own Esc would win)."""
        actions = {
            "n": lambda: self.toggle_mode(DRAW),
            "d": lambda: self.toggle_mode(DELETE),
            "Escape": self.escape,
            "o": lambda: self.toggle_mode(TOGGLE),
            "Shift-O": self.all_in_view_ok,
        }
        for key, action in actions.items():
            # Nothing to act on before a folder is open.
            self.viewer.bind_key(key, lambda viewer, action=action: self.is_open and action(), overwrite=True)


class _Widget:
    """The dock widget: image, review counts, colours, buttons and keys. Built lazily so that the module imports
    without Qt."""

    # ``__new__`` returns a plain QWidget (not a _Widget) with two functions attached, ``update_status`` and
    # ``update_mode``, which the Annotator calls after every change.
    def __new__(cls, annotator: Annotator):
        from qtpy.QtWidgets import QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

        widget = QWidget()
        layout = QVBoxLayout(widget)

        def choose_folder():
            # The system's own folder picker (Finder on Mac, Explorer on Windows); empty if cancelled.
            path = QFileDialog.getExistingDirectory(widget, "Open a round folder or an image folder")
            if path:
                annotator.open_folder(path)

        open_button = QPushButton("Open folder")
        open_button.clicked.connect(lambda _=False: choose_folder())
        layout.addWidget(open_button)
        # Image and counts, then the current mode: filled in by update_status / update_mode.
        title, status, mode = QLabel(), QLabel(), QLabel()
        for label in (title, status, mode):
            label.setWordWrap(True)
            layout.addWidget(label)

        # Legend: a square in each outline colour (the same COLOURS as the outlines).
        def swatch(key, text):
            return f"<span style='color:{_hex(COLOURS[key])}'>&#9633;</span> {text}"

        legend = QLabel(
            "<br>".join(
                [
                    swatch(af.UNREVIEWED, "not checked"),
                    swatch(af.OK, "OK"),
                ]
            )
        )
        layout.addWidget(legend)
        # ``action=action`` / ``step=step``: bind the current value, not the loop variable's last one.
        for text, action in (
            ("All in view OK", annotator.all_in_view_ok),
            ("Next unchecked cell", annotator.next_unreviewed),
            ("Done", annotator.press_done),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, action=action: annotator.is_open and action())
            layout.addWidget(button)
        row = QHBoxLayout()
        for text, step in (("< Image", -1), ("Image >", 1)):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, step=step: annotator.go(step))
            row.addWidget(button)
        layout.addLayout(row)
        compare = QCheckBox("Other masks")
        compare.toggled.connect(annotator.set_compare)
        layout.addWidget(compare)
        # Mouse and keys, as a reminder for the annotator (Qt labels take a subset of HTML).
        keys = QLabel(
            "<b>Mouse</b><table>"
            "<tr><td>Right&nbsp;</td><td>new cell: go around it, back to the start or click</td></tr>"
            "<tr><td>Drag</td><td>move the image</td></tr>"
            "<tr><td>Scroll</td><td>zoom</td></tr>"
            "</table><b>Keys</b> (on the cell under the mouse)<table>"
            "<tr><td>N</td><td>draw mode: left click draws</td></tr>"
            "<tr><td>D</td><td>delete mode: click or drag to delete</td></tr>"
            "<tr><td>Esc</td><td>drop outline / leave mode</td></tr>"
            "<tr><td>O</td><td>OK mode: click or drag to mark OK / not checked</td></tr>"
            "<tr><td>Shift+O&nbsp;</td><td>all in view OK</td></tr>"
            "<tr><td>Cmd+Z&nbsp;</td><td>undo (Ctrl+Z on Windows)</td></tr>"
            "</table>napari's brush, eraser and fill: their buttons, no keys. Saved automatically."
        )
        keys.setWordWrap(True)
        layout.addWidget(keys)
        layout.addStretch()  # everything to the top of the panel

        def update_status():
            if not annotator.is_open:
                title.setText("<b>No folder open</b>")
                status.setText("Open a round folder or an image folder.")
                return
            counts = af.summary(annotator.cells)
            record = annotator.record
            kind = "test image (from scratch)" if annotator.test_mode else "correction"
            # Whose image and which round: a mix-up of folders shows here.
            where = "test" if annotator.test_mode else f"round {record['round']}" if "round" in record else ""
            who = ", ".join(x for x in (record.get("annotator"), where) if x)
            title.setText(
                f"<b>{annotator.folder.name}</b><br>{who + '<br>' if who else ''}"
                f"image {annotator.index + 1} of {len(annotator.folders)}, {kind}"
            )
            text = f"{counts['cells']} cells: {counts[af.OK]} OK, {counts[af.UNREVIEWED]} not checked"
            if annotator.read_only:
                text = f"<b>{where.capitalize() or 'Round'} closed: read-only</b><br>{text}"
            if annotator.done:
                text += "<br><b>Done</b>"  # every cell checked: untouched cells now count as correct
            elif counts["cells"] and not counts[af.UNREVIEWED] and not annotator.read_only:
                text += "<br>Every cell OK: press <b>Done</b> when finished."
            if annotator.save_error is not None:
                text += f"<br><b style='color:{_hex(WARNING)}'>Not saved: folder unreachable.</b> "
                text += "Your changes are kept; trying again every few seconds."
            status.setText(text)

        def update_mode():
            names = {
                NAVIGATE: "navigate",
                DRAW: "draw (N)",
                DELETE: "delete (D): click or drag over the cells to delete",
                TOGGLE: "OK (O): click or drag over the cells to mark",
            }
            mode.setText(f"Mode: <b>{'read-only' if annotator.read_only else names[annotator.mode]}</b>")
            compare.setEnabled(not annotator.test_mode)

        widget.update_status = update_status
        widget.update_mode = update_mode
        return widget


def _hex(colour) -> str:
    """An RGBA colour (0 to 1) as ``#rrggbb`` for the panel's HTML."""
    return "#" + "".join(f"{round(255 * c):02x}" for c in colour[:3])


def _polygon_area(path: np.ndarray) -> float:
    """Area of the polygon ``path`` ((y, x) points) in pixels (shoelace formula)."""
    y, x = path[:, 0], path[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _message(text: str, parent=None) -> None:
    """A message the annotator has to close (Done or the next image refused), on top of ``parent``."""
    from qtpy.QtWidgets import QMessageBox

    QMessageBox.information(parent, "AnnoCell", text)


def _ask(text: str, parent=None) -> bool:
    """A yes / no question on top of ``parent``; No is the default (Enter does not leave by accident)."""
    from qtpy.QtWidgets import QMessageBox

    buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    answer = QMessageBox.question(parent, "AnnoCell", text, buttons, QMessageBox.StandardButton.No)
    return answer == QMessageBox.StandardButton.Yes


def _close_guard(annotator: Annotator):
    """An event filter for napari's window: closing it with an image not done asks first, and No keeps it open."""
    from qtpy.QtCore import QEvent, QObject

    class CloseGuard(QObject):
        def eventFilter(self, watched, event) -> bool:
            if event.type() == QEvent.Type.Close and not annotator.may_leave("Close napari anyway?"):
                event.ignore()
                return True  # handled: napari does not close
            return False

    return CloseGuard()


def _cannot_leave() -> None:
    from napari.utils.notifications import show_error

    show_error("Not saved: this image stays open until it can be saved (is the drive reachable?).")


def folders_to_open(path: str | Path) -> tuple[list[Path], int]:
    """The image folders to move through and the one to open first: ``path`` itself among its siblings, or the
    first image of ``path`` (a round or test folder, or the round an annotator's folder opens) not done yet."""
    path = Path(path)
    single = (path / af.RECORD).exists()
    if not single and not af.image_folders(path):
        path = _group_to_open(path)  # an annotator's folder
    folders = [f for f in af.image_folders(path.parent if single else path) if (f / af.IMAGE).exists()]
    if single and path not in folders or not folders:
        raise FileNotFoundError(
            f"No image to open in {path}: open your own folder, a round folder or an image folder in the AnnoCell "
            f"layout (with {af.IMAGE})."
        )
    if single:
        return folders, folders.index(path)
    pending = [i for i, f in enumerate(folders) if not af.is_complete(af.read_review(f))]
    return folders, pending[0] if pending else 0


def _group_to_open(folder: Path) -> Path:
    """In an annotator's folder: the latest round with an image not closed, else the test images, else the latest
    round (read-only); ``folder`` itself if it holds none of them."""
    closed = {}  # round or test folder: whether each of its images is closed
    for record in folder.glob(f"*/*/{af.RECORD}"):
        closed.setdefault(record.parent.parent, []).append(af.is_closed(af.read_record(record.parent)))
    rounds = sorted(
        (g for g in closed if g.name.removeprefix("round_").isdigit()), key=lambda g: int(g.name.removeprefix("round_"))
    )
    open_rounds = [g for g in rounds if not all(closed[g])]
    test = folder / af.TEST
    return open_rounds[-1] if open_rounds else test if test in closed else rounds[-1] if rounds else folder


def main(argv: list[str] | None = None) -> None:
    """``annocell [<folder>]``: open the tool (empty without a folder) and save when napari closes."""
    parser = argparse.ArgumentParser(prog="annocell", description="Correct segmentations in napari.")
    parser.add_argument(
        "folder", nargs="?", help="your own folder, a round folder or an image folder (default: choose in the tool)"
    )
    args = parser.parse_args(argv)
    folders, index = [], 0
    if args.folder:
        try:
            folders, index = folders_to_open(args.folder)
        except FileNotFoundError as error:
            parser.error(str(error))  # a plain message for the annotator, not a traceback
    import napari  # here, so that ``--help`` and errors answer without loading napari

    viewer = napari.Viewer()
    annotator = Annotator(viewer, folders, index)
    napari.run()  # returns when the napari window is closed
    if not annotator.save():
        backup = annotator.save_backup()
        print(f"Could not save to {annotator.folder}. A copy is in {backup}: copy its files back into that folder.")


if __name__ == "__main__":
    sys.exit(main())
