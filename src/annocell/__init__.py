"""AnnoCell: correct a proposed cell segmentation in napari, cell by cell, and record what was checked.

``folder``: the self-contained image folder (numpy and tifffile only), the contract with the pipeline that exports
the images and reads the corrections back; ``app``: the napari tool, started with ``annocell <folder>``. See
``docs/annotation_tool.md``.
"""

__version__ = "0.1.0"
