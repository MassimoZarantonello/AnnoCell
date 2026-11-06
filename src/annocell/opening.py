"""Which image folders to open, from the folder given (``annocell <folder>`` or the panel's "Open folder")."""

from __future__ import annotations

from pathlib import Path

from annocell import folder as af


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
