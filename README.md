# AnnoCell

Correct a proposed cell segmentation in [napari](https://napari.org), cell by cell, and record which cells were
checked. Built for annotation rounds of [macroseg](../macroseg) (fine-tuning Cellpose on MACSima images), but it only
needs a folder in the AnnoCell layout: any pipeline can write one.

See [docs/annotation_tool.md](docs/annotation_tool.md) for the design and every decision.

## For the annotator

Install once with [uv](https://docs.astral.sh/uv/) (Mac or Windows), then open the folder you received:

```bash
uv tool install "annocell[gui] @ <wheel or git URL>"
annocell <folder>
```

Or just `annocell`, then **Open folder** in the panel to pick the folder.

- **Right click**: new cell; move around it, back to the start (or click again) to close it. `Esc` drops it.
- **N** draw mode (left click draws), **D** delete mode (click or drag over cells to delete), **O** OK mode (click or
  drag over cells to mark them OK, or back to not checked when the first one was OK); **Esc** back to navigating.
- **Shift+O** all in view OK.
- **Shift+M** macrophage (test images only).
- A wrong outline: delete it, draw it again. Saved automatically.

## The folder (`annocell.folder`)

One folder per annotator (`<root>/<annotator>/round_<NN>/<roi>/`, and `test/<roi>/` for test regions), one
folder per ROI inside, self-contained: `image.tif` (all channels), `start_masks.tif` (the proposal), `compare/`
(read-only masks of other segmentations), `masks.tif` and `cells.csv` (written by the tool: masks, and review state
and origin of every cell) and `annotation.json` (provenance, channels, review summary, `format_version`). The
pipeline writes the folder and reads the corrections back with `annocell.folder` (numpy and tifffile only, no
napari).

## Development

```bash
uv sync --extra gui --extra dev
.venv/bin/python -m pytest
```
