"""The tool's fixed values: modes, colours, timings, and napari's keys switched off."""

from __future__ import annotations

# Outline colours (RGBA), by ``Cell.ok``: yellow not checked, green OK.
COLOURS = {
    False: (1.0, 0.85, 0.0, 1.0),
    True: (0.1, 0.9, 0.3, 1.0),
}
MACROPHAGE_COLOUR = (1.0, 0.0, 1.0, 1.0)  # magenta outline: a cell marked as a macrophage (wins over its state)
WARNING = (1.0, 0.2, 1.0, 1.0)  # "Not saved" in the panel
CHANNEL_COLORMAPS = ("red", "green", "blue")  # channels without a colour in annotation.json, in turn
SAVE_EVERY_MS = 3000
UNDO_STEPS = 100  # as many as napari keeps of the pixels
MIN_CELL_PX = 10  # a smaller outline is a slip, not a cell
NAVIGATE, DRAW, DELETE, TOGGLE, MACROPHAGE = "navigate", "draw", "delete", "toggle", "macropahge"
CLOSE_SCREEN_PX = 8  # an outline closes when the mouse comes back this close to its start
# napari's keys on a labels layer (napari 0.9: tools, label IDs, brush size), switched off on the masks layer; undo,
# redo, Escape (bound by the tool) and Enter (closes napari's polygon) stay.
NAPARI_KEYS = ("1", "2", "3", "4", "5", "6", "7", "E", "P", "F", "L", "Z", "X", "B", "-", "=", "[", "]")
