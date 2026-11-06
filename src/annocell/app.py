"""The napari annotation tool: ``annocell <folder>``.

``<folder>`` is the annotator's own folder (the tool opens their latest round that is not closed), a round (or
test) folder, or one image folder; the tool opens the first image not done yet and moves through the
others of the same round with Previous / Next. Without ``<folder>`` it opens empty, and the panel's "Open folder"
button picks one. An image of a closed round (``closed_on``) opens read-only: it can be looked at, not changed; a
round closed while one of its images is open turns it read-only at the next save.

Layers: every channel of the image (napari's own controls) and the editable masks (``masks (from <segmentation>)``),
outlined in yellow (unreviewed) or green (OK). Read-only comparison masks of other segmentations are added to the
layer list only on request (a checkbox), hidden.

napari's own editing tools (brush, eraser, fill, polygon, picker) are there, as buttons only: their keys (P, E, F,
L, 1 to 7, ...) do nothing (``NAPARI_KEYS``), so a key pressed by mistake never changes the masks. Picking one of
them leaves the tool's mode.

How it fits together:

- ``annocell.annotator``: ``Annotator``, the state of the session (current image, masks layer, cell table), opening
  images, Done, saving, colours and keys;
- ``annocell.tools``: the modes, the mouse and the edits (drawing, deleting, marking OK), mixed into ``Annotator``;
- ``annocell.widget``: the side panel;
- ``annocell.dialogs``: messages, questions and the guard before napari closes;
- ``annocell.opening``: which image folders to open from the folder given;
- ``annocell.constants``: modes, colours, timings, napari's keys switched off;
- ``annocell.folder``: reads and writes the files.

Qt and napari are imported inside functions, so that these modules (and their tests' helpers) import without a
display.
"""

from __future__ import annotations

import argparse
import sys

from annocell.annotator import Annotator
from annocell.constants import DELETE, DRAW, NAVIGATE, TOGGLE
from annocell.opening import folders_to_open

__all__ = ["DELETE", "DRAW", "NAVIGATE", "TOGGLE", "Annotator", "folders_to_open", "main"]


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
