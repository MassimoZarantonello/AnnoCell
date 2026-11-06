"""``Annotator``: the state of one napari session and the images it moves through.

It opens images (channels, masks, comparison masks), keeps the cell table, colours the outlines, saves, and decides
when the annotator may go on (Done, the next image, closing napari). The editing tools (modes, mouse, edits) are in
``annocell.tools``, mixed into this class; the side panel is ``annocell.widget``.

**Done**: a button. The annotator presses it when every cell is OK; with cells still not checked, a message says
how many, and nothing is marked. Any change after Done takes it back: press it again. "Image >" goes on only from
an image done (no skipping); closing napari or opening another folder with an image not done asks first.

The masks are saved every few seconds, when changing image and when closing napari. The folders may be on a
network drive: a save that fails (drive unreachable) keeps everything in memory, says so in the panel, and is tried
again at the next autosave; the tool does not change image until it has saved. If it still fails when napari closes,
a copy goes to ``AnnoCell unsaved`` in the home folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from annocell import dialogs
from annocell import folder as af
from annocell.constants import CHANNEL_COLORMAPS, COLOURS, DELETE, DRAW, NAPARI_KEYS, NAVIGATE, SAVE_EVERY_MS, TOGGLE
from annocell.opening import folders_to_open
from annocell.tools import Tools
from annocell.widget import Widget


class Annotator(Tools):
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
        self.widget = Widget(self)
        viewer.window.add_dock_widget(self.widget, name="Annotation", area="right")
        self._bind_keys()
        # Closing napari with an image not done asks first (kept here: Qt does not keep the filter alive).
        self._close_guard = dialogs.close_guard(self)
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
            dialogs.cannot_leave()
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
            dialogs.cannot_leave()
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

    # Done, and leaving an image

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
        return dialogs.ask(
            f"This image is not marked as done: it stays out of training until you press Done. {question}",
            self._window(),
        )

    def _tell(self, text: str) -> None:
        dialogs.message(text, self._window())

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

    def _on_closed(self, *args) -> None:
        self.closed = True
        self.timer.stop()

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
