"""Audio feature extraction.

Turns a voice clip into the 34 numbers the voice model uses: the mean and standard deviation of 13
MFCCs (the standard summary of a voice's timbre), plus four more spectral measures as mean and std.

`extract_audio_features` is the entry point the CLI and the CSV builder both call. It has to be a
function, not notebook cells, because the CLI calls it at demo time on a clip it has never heard.

FOR TEDLA: this is a first version to read through and rewrite in your own words. The recipe is
load the clip, compute each feature as a value per short time frame, then summarise each one by its
average and spread over time. All the settings come from src/config.py.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .. import config

# Shortest clip we will accept. Below this there are too few frames for stable statistics.
MIN_DURATION_SECONDS = 0.2


class AudioTooShort(Exception):
    """Raised when a clip is too short to extract framed features from."""


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def features_from_signal(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute the 34-value feature vector from a loaded audio signal."""
    if len(y) < MIN_DURATION_SECONDS * sr:
        raise AudioTooShort(f"clip is shorter than {MIN_DURATION_SECONDS}s")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds = mfcc.std(axis=1)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    spectral = []
    for values in (rolloff, centroid, rms, zcr):
        mean, std = _mean_std(values)
        spectral.extend([mean, std])

    # Order must match schemas.AUDIO_FEATURE_COLUMNS: 13 means, 13 stds, then the spectral pairs.
    return np.concatenate([mfcc_means, mfcc_stds, spectral]).astype(float)


def extract_audio_features(path: Path | str) -> np.ndarray:
    """Load an audio file and return its 34-value feature vector."""
    y, sr = librosa.load(str(path), sr=config.SAMPLE_RATE)
    return features_from_signal(y, sr)
