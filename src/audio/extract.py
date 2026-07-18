"""Audio feature extraction.

FOR TEDLA: implement extract_audio_features below.

Same as the image side. It is called both to build audio_features.csv and by the CLI to featurise
a clip recorded live during the demo, so it must be a function, not notebook cells.

Requirements:
  - return a 1-D float array of exactly 34 values
  - order must match schemas.AUDIO_FEATURE_COLUMNS: 13 mfcc means, 13 mfcc stds, then rolloff
    mean/std, centroid mean/std, rms mean/std, zcr mean/std
  - load at config.SAMPLE_RATE and use config.N_MFCC, do not hardcode
  - raise AudioTooShort rather than returning NaNs for a clip too short to frame

Checked working here with librosa 0.11:
  librosa.load, then librosa.feature.mfcc / spectral_rolloff / spectral_centroid / rms /
  zero_crossing_rate, then .mean(axis=1) and .std(axis=1) over the time axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class AudioTooShort(Exception):
    """Raised when a clip is too short to extract framed features from."""


def extract_audio_features(path: Path | str) -> np.ndarray:
    raise NotImplementedError(
        "Tedla: implement this, see the module docstring. The CLI and the CSV builder both call it."
    )
