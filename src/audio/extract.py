"""Audio feature extraction — the contract between the audio track and the app.

FOR TEDLA
=========
Implement `extract_audio_features` below. Same deal as the image side: it is the single entry
point, used both to build `audio_features.csv` and by the CLI to featurise a clip recorded live
during the demo. Notebook cells alone won't work — the app has to be able to call it.

Requirements:
  * return a 1-D float array of exactly len(schemas.AUDIO_FEATURE_COLUMNS) == 34 values
  * the order must match schemas.AUDIO_FEATURE_COLUMNS exactly:
      13 mfcc means, 13 mfcc stds, then rolloff mean/std, centroid mean/std,
      rms mean/std, zcr mean/std
  * load at config.SAMPLE_RATE and use config.N_MFCC — don't hardcode
  * raise AudioTooShort rather than returning NaNs for a clip too short to frame

Verified working on this machine (librosa 0.11):
  librosa.load(path, sr=config.SAMPLE_RATE)
  librosa.feature.mfcc / spectral_rolloff / spectral_centroid / rms / zero_crossing_rate
  then .mean(axis=1) and .std(axis=1) over the time axis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class AudioTooShort(Exception):
    """Raised when a clip is too short to extract framed features from."""


def extract_audio_features(path: Path | str) -> np.ndarray:
    """Return the 34-dim feature vector for one clip. See module docstring."""
    raise NotImplementedError(
        "Tedla: implement this — see the module docstring in src/audio/extract.py. "
        "The CLI and the CSV builder both call it."
    )
