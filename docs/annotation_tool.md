# Annotation tool: correcting segmentations in napari

Status: version 1 built (`annocell`, command `annocell`); tried by Massimo, not yet by an annotator.
Replaces the Cellpose GUI step (notebook 02) of the annotation rounds.
Written for the annotation loop of [AnnoLoop](../../AnnoLoop) (the pipeline, which started as a copy of macroseg and
still has the package name `macroseg`; `macroseg.annotation` exports and closes rounds); the tool itself only needs
a folder in the AnnoCell layout.

## Use

Massimo exports, in Python, from AnnoLoop (`macroseg.annotation`):

```python
export_round(sdata, cfg, "membrane_dapi", root, 2, "alice", compare=["exp201_test_crop_r01"], models_root=models)
export_test(sdata, cfg, [y0, y1, x0, x1], root, "carla")  # a test region, drawn from scratch
close_round(root, 2)  # the round is over: its ROIs become read-only
```

The annotator opens their own folder; the tool opens their latest round that is not closed (else their test
regions, else their latest round, read-only):

```bash
annocell <root>/alice
```

A round folder or a single image folder opens too. Or `annocell` alone: napari opens with an empty panel, and its
**Open folder** button picks the folder in the system's folder picker. napari's own File → Open is not used: it would open the files as plain layers,
outside the tool.

Three verbs, three owners:

- **closing** a round (`close_round(root, n)` in AnnoLoop, which calls `annocell.folder.close` on each ROI) writes
  `closed_on` in `annotation.json`: the round is over, its ROIs open read-only from then on, and AnnoLoop refuses to
  export into it. Nothing else changes: the masks and the review stay as the annotator left them;
- **complete** (`annocell.folder.is_complete`) is not a step but a test on the review: at least one cell and none
  unreviewed. Training takes the complete ROIs, closed or not;
- **finishing** an ROI (`finish(folder)` in AnnoLoop, optional) writes in `annotation.json` its `status`
  (`complete` / `partial`) and `start_score`, the score of the proposal against the corrections. It does not count
  the corrections: those are AnnoCell's review.

## Goal

Let a biologist correct a proposed cell segmentation quickly and completely, and record which cells were
checked. **The focus is segmentation, not classification**: the tool says whether each outline is right,
not what the cell is.

Why a new tool: the Cellpose GUI cannot mark a cell as checked, so "untouched = correct" is an assumption,
not data; and it shows at most 3 channels. (Fixing an outline is still delete and redraw, as in Cellpose: the
brush and eraser were tried and dropped.)

## Users and setting

- Biologists (not programmers), several, each with their own ROIs in their own folder: one annotator per ROI.
- Runs **locally** on the annotator's computer, **Mac or Windows** (unknown in advance): nothing OS-specific,
  shortcuts documented as Ctrl / Cmd.
- No training, no GPU needed in the tool.

## What the annotator sees

1. **Every image channel as its own napari image layer** (DAPI, NaK_ATPase, F4/80, autofluorescence, ...),
   read from the folder's `image.tif`, each toggled on and off on its own.
   Colours: red, green, blue, set per channel at export (`colours=`, e.g. F4/80 red, NaK_ATPase green, DAPI blue
   as in the Cellpose GUI), else red, green, blue in turn.
   napari's own controls are kept as they are: contrast limits, gamma, colormap, blending, opacity.
2. **The proposed masks**, editable, shown as outlines: the baseline segmentation in round 1, the latest
   fine-tuned model's in later rounds. The layer is named after it, e.g. `masks (from membrane_dapi)`.
   Outline colour shows the review state (unreviewed / OK); exact colours do not matter.
3. **Comparison layers** (read-only), **only on request**: a checkbox adds them to the layer list, hidden. Shown
   all the time they confused more than they helped. They are the masks of earlier segmentations and models of the
   same image, copied into the folder at export, to see the changes from model to model. Each is labelled with
   whether that model was **trained on this image** (then its improvement there does not count).
4. **The side panel**: image, annotator and round (a mix-up of folders shows there), counts, a legend of the colours, the two bulk buttons (all in view OK, go to an
   unreviewed cell), previous / next image, the comparison checkbox, and the mouse and keys. The modes have no
   buttons: they are keys (`N`, `D`), and the panel shows the current one.

## Actions (version 1)

Three modes, besides navigating: **draw** (`N`), **delete** (`D`) and **OK** (`O`), keys only; `Esc` (or the same
key again) goes back to navigating. Dragging pans while navigating and drawing (except while an outline is being
drawn); in delete and OK mode it acts on every cell passed over. The other keys act on the cell under the mouse (no
need to select it first). Tried and dropped: left click to
draw with the space bar to pan (losing the plain drag to pan was worse).

| Action | How | Effect |
|---|---|---|
| Draw a cell | right click (any mode; or left click in draw mode), move around the cell with no button held (the outline follows the mouse, as in the Cellpose GUI); it closes back at the start or with a second click; `Esc` drops it | new cell; it takes the pixels it covers from overlapping cells, whose remainders stay; each outline is a new cell. napari's point-by-point polygon was too slow, and holding the button down was tiring |
| Delete a cell | delete mode (`D`): click it, or drag over several | cells removed; a drag is one undo step |
| Fix an outline | delete it (`D`) and draw it again | napari's brush, eraser, fill, polygon and picker are hidden and switched off (decided after trying them: they made it easy to damage the masks by accident) |
| Mark OK | OK mode (`O`): click a cell, or drag over several | the outline is right; a stroke starting on an OK cell takes the cells back to unreviewed (the first cell decides the direction, so passing over a cell twice does not flip it back) |
| Mark all remaining in view as OK | `Shift+O` or button | every unreviewed cell **entirely** in view becomes OK, recorded as OK **in bulk** (cells cut by the view edge are left; cells already OK keep how they became OK) |
| Macrophage on / off | `Shift+M`, test mode only | the cell is marked as a macrophage |
| Go to an unreviewed cell | button | centres the view on the nearest unreviewed cell (to finish an image) |
| Next / previous image | two buttons | saves, then moves through the images of the round |

Every cell is either **unreviewed** or **OK**; a drawn or edited cell is OK. A cell that only lost pixels to a new
cell keeps its state (the annotator has not looked at it). An image is **complete** when no cell is unreviewed.

**How a cell became OK** (`ok_by` in `cells.csv`): `hand` (marked in OK mode, edited or drawn) or `bulk` (all in view
OK); the last action wins, and a cell taken back to unreviewed forgets it. So that an annotator may clear the easy
cells quickly (allowed: AnnoLoop D6) while the statistics can still tell cells looked at one by one from cells
cleared in bulk (D7). Not a monitoring of the annotator: no time, no order, only the last action per cell.
Outlines: yellow unreviewed, green OK, magenta macrophage (test mode).

`origin` (`proposed` / `edited` / `drawn`) is not tracked from the edits but recomputed against the start masks at
every save, so undo and redo cannot leave it wrong. For training, drawn and edited cells both count as corrected by
hand (`hand_drawn`, which the train / test split of the tiles spreads over both sides).

There is **no "uncertain" state**, on purpose: the annotator always decides. Cellpose cannot learn from
"don't know" anyway (every pixel is a cell or background). Borderline cases are decided by a fixed rule in the
annotation guidelines, so that forced choices are at least consistent, e.g. *not sure it is a cell: delete it;
not sure about the outline: keep the proposal*.

Saving is automatic (every 3 seconds when something changed, when changing image and when closing napari), so a
crash loses little; undo / redo as in napari.

**Network drive.** The folders are on a network drive (fs4), reached through the VPN from home, so a save can fail
(drive or VPN down). Then everything stays in memory, the panel says "Not saved", the next autosave tries again,
and the tool does not change image until it has saved. If it still cannot save when napari closes, it writes a copy
to `AnnoCell unsaved` in the home folder and says so. Saves write a temporary file and rename it, so a dropped
connection never leaves half a file.

**Closed rounds.** An ROI whose round is over (`closed_on`, written by the pipeline) opens **read-only**: no editing
mode, no drawing, no marking, never saved; the panel says "Round N closed: read-only". Its masks were already used
for training: changing them afterwards would make that training impossible to reproduce. A round closed while an
ROI of it is open: the tool reads `closed_on` before every save, and the next save (within 3 seconds) turns the ROI
read-only and writes nothing; the changes since the last save are dropped, with a warning.

## Data

One folder per annotator, the only one they open, with their rounds and test regions inside:
`<root>/<annotator>/round_<NN>/<roi>/` and `<root>/<annotator>/test/<roi>/` (`annocell.folder.rois` lists them;
annotator names are lowercase letters, digits and `-`). Annotator first, so that nobody has to pick their round
among everyone's. When a round is over, the pipeline closes its ROIs (`closed_on`, `annocell.folder.close`).
Overviews (AnnOverview) read the tree and never write to it.

One folder per ROI, **self-contained**: the annotator's laptop has no stores, so the export copies everything the
tool shows, and the tool needs only napari, tifffile and numpy (no spatialdata, Harpy or Cellpose). A 1024 x 1024,
6-channel uint16 crop is 12 MB; a mask is 4 MB.

- `image.tif`: every channel of the crop, with the channel names;
- `start_masks.tif`: the proposed masks as given to the annotator (to count corrections); absent in test mode;
- `compare/<segmentation>.tif`: the comparison layers; "trained on this image" is decided at export from each
  model's `model.json`; absent in test mode;
- `masks.tif`: the current masks (uint32 labels). Replaces `<image>_seg.npy` of the Cellpose GUI (a pickle; the
  GUI is no longer used);
- `cells.csv`: one row per cell: `cell_ID`, `state` (`unreviewed` / `ok`), `origin` (`proposed` / `edited` /
  `drawn`; replaces the GUI's `ismanual`, used by training as hand-drawn cells), `macrophage` (test mode only),
  `ok_by` (`hand` / `bulk`, empty while unreviewed);
- `annotation.json`: written by the pipeline only: annotator, round, `mode` (`train` / `test`), provenance,
  channels, start segmentation, `closed_on` when the round is over, and what AnnoLoop's `finish` adds (`status`,
  `start_score`);
- `review.json`: written by the tool only, at every save: the counts (cells, OK, unreviewed, OK by hand and in
  bulk, kept / edited / drawn, `start_cells`, `deleted`, macrophages) and `saved_at`, `started_at`, `completed_at`.
  The **one definition** of the correction counts: AnnoLoop and AnnOverview read them, and do not recount.

**One writer per file**: the pipeline writes `annotation.json`, the tool writes `masks.tif`, `cells.csv` and
`review.json`. Format 1 kept the review in `annotation.json`, so a save and a `finish` or a `close` running at the
same moment could overwrite each other's fields; format 2 (`format_version`) moved it out. Format-1 folders are not
read (no real round used them): export them again.

Saves write a temporary file and rename it, so a crash during a save cannot corrupt the masks.

Images in the loop: **1024 x 1024 px crops**, each its own config and store (like `exp201_test_crop`); the store
stays on Massimo's side and the export reads from it.

## Macrophages

Not annotated. Macrophage identity comes from **F4/80 on the corrected masks**: after an image is finished,
per-cell intensities are recomputed on the corrected masks (a correct outline makes the per-cell mean reliable).
Before correction (on proposed masks) per-cell F4/80 is unreliable, so the tool does not use it to guide the
annotator; the F4/80 channel itself is visible. To check before fixing a threshold: that F4/80 splits into two
groups on corrected cells, and how much of it is autofluorescence (`autofluo_APC`).

## Training

Not in the tool. Fine-tuning runs in AnnoLoop (`macroseg.train`, notebook 04 for now, a cluster script later). The
tool only picks up a new model's segmentation of the next image as its proposal, when it exists.

## Test ROI

A test set annotated **from scratch**, never used for training, to score every model every round (AP on all
cells and on macrophages, one curve over rounds); correction effort per round is the supporting evidence.
Drawn once and then fixed.

- **From scratch**: a ground truth corrected from a segmentation favours that segmentation, so the tool has a
  **test mode** that starts from empty: the image channels only, **no mask layer at all** (no proposal, no
  comparison layers, not even hidden ones).
- **Size: by macrophages, not by area.** Several **512 x 512 patches** (about 100 cells each) in F4/80-rich areas,
  until about **100 macrophages** in total (with 40, the standard error of a recall around 0.8 is 0.06, larger
  than the expected gain per round). Patches are placed with the baseline's per-cell F4/80; this choice of
  *where* does not bias the labels. Trade-off: F4/80-rich areas are crowded and over-represent hard cases.
  The annotator can spend several hours, so the cost of drawing is acceptable.
- **One test set per slide / tissue**: more tissues are coming; a test set on one slide only measures that slide.
- **No overlap with training, enforced in code**: patch coordinates are stored in slide pixels; crops and tiles
  within one cell diameter (60 px) of a patch are rejected. Patches lie outside every crop already annotated
  (Exp201: outside `[2000:4048, 2000:4048]`).
- **Fully excluded from training**: test patches never enter `training_set`, not even as Cellpose's test loss:
  AnnoLoop's `training_set` reads only the complete ROIs of the rounds, never `<annotator>/test/`.
- **Macrophages marked by hand here only**: one key toggles a macrophage flag per cell, in test mode only. It also
  calibrates the F4/80 rule (agreement with high F4/80 on the drawn masks, effect of autofluorescence).
- **Drawn by the biologist**, the same person who corrects the training images (otherwise part of every score is
  the difference between two annotators' styles), and the same person for every test set.
- **Noise ceiling**: about 50 cells redrawn by the same annotator weeks later, and by Massimo. Their agreement is the
  AP explained by annotation noise; differences between models below it are not real.

## Out of version 1

- Training or segmenting from the GUI.
- Per-cell confidence (needs the segmentation step to save Cellpose's probability map).
- Several annotators on the same ROI (blending them is the pipeline's: macroseg `docs/concurrent_annotations.md`),
  locking, time per image (decided: not measured; the review records only when an ROI was started and finished,
  `started_at` / `completed_at`).
- Classification of cells (macrophage / other): a later phase (except the macrophage flag of test mode).

## Packaging and installation

- **Its own package and repository, AnnoCell** (moved out of macroseg, where it started as `macroseg.annotate`): the
  tool has its own users, dependencies (napari, Qt) and pace. `annocell.folder` (numpy and tifffile only) is the
  folder format, the contract with the pipeline: macroseg depends on it to export and read back, so the format is
  defined in one place and cannot drift; `format_version` in `annotation.json` marks the layout. The napari tool is
  the `gui` extra: a pipeline never installs napari.
- **A launcher, not a napari-menu plugin**: `annocell <round folder>` opens napari with every layer and the
  widget set up; no plugin manifest for the annotator to find.
- **Installation with uv**: one line installs uv (Mac or Windows), then `uv tool install` from a wheel handed over
  with the data or from the git URL. To test on a Windows laptop before the first round.
- **Performance** is not a concern: on the 2048 x 2048 Exp201 crop (4x the target, 1587 cells) on the Mac, opening
  takes 1.2 s, an edit 0.05 s, a save 0.04 s. The risks are UX and OpenGL drivers on some Windows laptops: to try
  on the annotator's laptop.

## Open questions

- Is the repository private (install from a wheel or from git)? Is a Windows laptop available for testing?
- How data reach the annotator and come back (shared drive, OneDrive, USB).
