"""The side panel: image, review counts, colours, buttons and keys."""

from __future__ import annotations

from annocell import folder as af
from annocell.constants import COLOURS, DELETE, DRAW, NAVIGATE, TOGGLE, WARNING


class Widget:
    """The dock widget. Built lazily so that the module imports without Qt."""

    # ``__new__`` returns a plain QWidget (not a Widget) with two functions attached, ``update_status`` and
    # ``update_mode``, which the Annotator calls after every change.
    def __new__(cls, annotator):
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
            return f"<span style='color:{hex_colour(COLOURS[key])}'>&#9633;</span> {text}"

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
                text += f"<br><b style='color:{hex_colour(WARNING)}'>Not saved: folder unreachable.</b> "
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


def hex_colour(colour) -> str:
    """An RGBA colour (0 to 1) as ``#rrggbb`` for the panel's HTML."""
    return "#" + "".join(f"{round(255 * c):02x}" for c in colour[:3])
