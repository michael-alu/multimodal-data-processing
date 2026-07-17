"""Image feature extraction.

FOR TAPS: implement extract_image_features below.

It is called in two places, building image_features.csv and the live demo, where the CLI has to
featurise a photo it has never seen. That second use is why it must be a function, not notebook
cells. Without it the system simulation cannot authenticate a new face.

Requirements:
  - return a 1-D float array of exactly 388 values
  - order must match schemas.IMAGE_FEATURE_COLUMNS: 64 histogram bins, then 324 HOG
  - take all params from src/config.py, do not hardcode
  - raise FaceNotFound if no face is detected

Recipe, already checked to give these dimensions:
  cv2 Haar cascade to find the face, crop, grayscale, resize to config.FACE_SIZE,
  cv2.calcHist with config.HIST_BINS normalised, then skimage.feature.hog with the config params.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FaceNotFound(Exception):
    """Raised when no face can be located in an image."""


def extract_image_features(path: Path | str) -> np.ndarray:
    raise NotImplementedError(
        "Taps: implement this, see the module docstring. The CLI and the CSV builder both call it."
    )
