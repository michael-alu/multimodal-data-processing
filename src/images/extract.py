"""Image feature extraction.

Turns a face photo into the 388 numbers the face model uses: a 64-bin brightness histogram plus a
324-value HOG (histogram of oriented gradients) shape descriptor.

`extract_image_features` is the entry point the CLI and the CSV builder both call. It has to be a
function, not notebook cells, because the CLI calls it at demo time on a photo it has never seen.

FOR TAPS: this is a first version to read through and rewrite in your own words. The recipe is
detect the face, crop it, make it grayscale, resize to a fixed size, then compute the two feature
sets. All the numbers (face size, bin count, HOG settings) come from src/config.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

from .. import config


class FaceNotFound(Exception):
    """Raised when no face can be located in an image."""


def _cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_and_prepare(image_bgr: np.ndarray) -> np.ndarray:
    """Find the face, crop it, grayscale it, resize to config.FACE_SIZE.

    Returns a grayscale square the size of config.FACE_SIZE. Raises FaceNotFound if there is no
    detectable face.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise FaceNotFound("no face detected")

    # If several faces are found, keep the largest one.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = gray[y:y + h, x:x + w]
    return cv2.resize(face, config.FACE_SIZE)


def features_from_face(face_gray: np.ndarray) -> np.ndarray:
    """Compute the 388-value feature vector from a prepared grayscale face."""
    hist = cv2.calcHist([face_gray], [0], None, [config.HIST_BINS], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)  # normalise so brighter or larger faces stay comparable

    shape = hog(
        face_gray,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    )

    return np.concatenate([hist, shape]).astype(float)


def extract_image_features(path: Path | str) -> np.ndarray:
    """Load an image file and return its 388-value feature vector."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return features_from_face(detect_and_prepare(image))
