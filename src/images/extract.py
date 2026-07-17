"""Image feature extraction — the contract between the image track and the app.

FOR TAPS
========
Implement `extract_image_features` below. It is the single entry point for image features, and it
is used in two places:

  1. building `image_features.csv` (loop over the raw photos and their augmentations), and
  2. the live demo, where the CLI must featurise a photo it has never seen.

That second use is why this has to be a function you can call with a path, not just cells in a
notebook. If it only exists as notebook cells, the app cannot authenticate a new face and the
system simulation does not work.

Requirements:
  * return a 1-D float array of exactly len(schemas.IMAGE_FEATURE_COLUMNS) == 388 values
  * the order must match schemas.IMAGE_FEATURE_COLUMNS exactly: 64 histogram bins, then 324 HOG
  * all params (face size, bin count, HOG settings) come from src/config.py — don't hardcode
  * raise FaceNotFound if the cascade finds no face, so callers can report it properly

The recipe, already verified to produce these dimensions:
  detect face with cv2 Haar cascade -> crop -> grayscale -> resize to config.FACE_SIZE
  -> cv2.calcHist 64 bins, normalised -> skimage.feature.hog with the config params
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FaceNotFound(Exception):
    """Raised when no face can be located in an image."""


def extract_image_features(path: Path | str) -> np.ndarray:
    """Return the 388-dim feature vector for one image. See module docstring."""
    raise NotImplementedError(
        "Taps: implement this — see the module docstring in src/images/extract.py. "
        "The CLI and the CSV builder both call it."
    )
