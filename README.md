# AnnoCell

Correct a proposed cell segmentation in [napari](https://napari.org), cell by cell, and record which cells were
checked. Built for annotation rounds of [macroseg](../macroseg) (fine-tuning Cellpose on MACSima images), but it only
needs a folder in the AnnoCell layout: any pipeline can write one.

See [docs/annotation_tool.md](docs/annotation_tool.md) for the design and every decision.

## For the annotator

Install once with [uv](https://docs.astral.sh/uv/) (Mac or Windows), then open your own folder: the tool opens
your latest round.

```bash
uv tool install "annocell[gui] @ <wheel or git URL>"
annocell <your folder>
```

Or just `annocell`, then **Open folder** in the panel to pick it.

- **Right click**: new cell; move around it, back to the start (or click again) to close it. `Esc` drops it.
- **N** draw mode (left click draws), **D** delete mode (click or drag over cells to delete; OK cells are kept), **O** OK mode (click or
  drag over cells to mark them OK, or back to not checked when the first one was OK); **Esc** back to navigating.
- **Shift+O** all in view OK.
- **Cmd+Z** / **Cmd+Shift+Z** undo / redo, OK marks included (Ctrl on Windows).
- A wrong outline: delete it, draw it again (or napari's brush and eraser, from their buttons: their keys are off).
- **Done** (button) when every cell is OK: the image is finished. With cells still not checked, it says how many.
  **Image >** goes on only once the image is marked done; otherwise it says why. Closing napari with an image not
  done asks first.
- Saved automatically. If the panel says **Not saved** (drive unreachable), keep working: your changes are kept and
  saved as soon as the drive is back.
- A **closed** round opens read-only: you can look, not change.

## The folder (`annocell.folder`)

One folder per annotator (`<root>/<annotator>/round_<NN>/<image>/`, and `test/<image>/` for test images), one
folder per image inside, self-contained: `image.tif` (all channels), `start_masks.tif` (the proposal), `compare/`
(read-only masks of other segmentations), `annotation.json` (provenance, channels, `closed_on`, `format_version`:
written by the pipeline only), and, written by the tool only, `masks.tif`, `cells.csv` (OK or not, origin, how each
cell became OK: by hand or in bulk, and the macrophage mark of test images) and `review.json` (the review summary). The pipeline writes the folder
and reads the corrections back with `annocell.folder` (numpy and tifffile only, no napari).

## Development

```bash
uv sync --extra gui --extra dev
.venv/bin/python -m pytest
```
