"""Frozen column schemas for image_features.csv and audio_features.csv.

These are the contract between the image track (Taps), the audio track (Tedla), and the models
(Michael). Build your DataFrame with the column list from this module and run the matching
validate_* function before writing to disk — that way the three tracks merge without a rename
pass at the end.

If a schema genuinely needs to change, raise it with Michael first; the model code reads these
column names directly.
"""

from __future__ import annotations

import pandas as pd

from . import config

# --- image_features.csv --------------------------------------------------

IMAGE_META_COLUMNS = ["member", "expression", "augmentation", "source_file"]

IMAGE_HIST_COLUMNS = [f"hist_{i}" for i in range(config.HIST_BINS)]          # 64
IMAGE_HOG_COLUMNS = [f"hog_{i}" for i in range(config.HOG_DIM)]              # 324

IMAGE_FEATURE_COLUMNS = IMAGE_HIST_COLUMNS + IMAGE_HOG_COLUMNS              # 388
IMAGE_COLUMNS = IMAGE_META_COLUMNS + IMAGE_FEATURE_COLUMNS                  # 392

# --- audio_features.csv --------------------------------------------------

AUDIO_META_COLUMNS = ["member", "phrase", "augmentation", "source_file"]

AUDIO_MFCC_COLUMNS = (
    [f"mfcc_{i}_mean" for i in range(1, config.N_MFCC + 1)]
    + [f"mfcc_{i}_std" for i in range(1, config.N_MFCC + 1)]
)                                                                            # 26

AUDIO_SPECTRAL_COLUMNS = [
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "rms_energy_mean", "rms_energy_std",
    "zcr_mean", "zcr_std",
]                                                                            # 8

AUDIO_FEATURE_COLUMNS = AUDIO_MFCC_COLUMNS + AUDIO_SPECTRAL_COLUMNS          # 34
AUDIO_COLUMNS = AUDIO_META_COLUMNS + AUDIO_FEATURE_COLUMNS                   # 38


# --- validation ----------------------------------------------------------


def _validate(df: pd.DataFrame, expected: list[str], feature_cols: list[str], kind: str) -> None:
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        raise ValueError(
            f"{kind}: column mismatch.\n"
            f"  missing ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}\n"
            f"  unexpected ({len(extra)}): {extra[:8]}{'...' if len(extra) > 8 else ''}\n"
            f"  or the order differs from schemas.{kind.upper()}_COLUMNS"
        )
    if df.empty:
        raise ValueError(f"{kind}: no rows")

    feats = df[feature_cols]
    if feats.isna().any().any():
        bad = feats.columns[feats.isna().any()].tolist()
        raise ValueError(f"{kind}: NaN in feature columns: {bad[:8]}")

    # A silently-failed extractor still writes a well-formed CSV of zeros; this is the check
    # that catches it.
    dead = feats.index[(feats == 0).all(axis=1)].tolist()
    if dead:
        raise ValueError(f"{kind}: {len(dead)} all-zero feature row(s) at index {dead[:5]} ")


def validate_image_features(df: pd.DataFrame) -> None:
    _validate(df, IMAGE_COLUMNS, IMAGE_FEATURE_COLUMNS, "image")

    unknown_members = set(df["member"]) - set(config.MEMBERS) - {config.UNKNOWN}
    if unknown_members:
        raise ValueError(f"image: unrecognised member(s): {sorted(unknown_members)}")

    bad_expr = set(df["expression"]) - set(config.EXPRESSIONS)
    if bad_expr:
        raise ValueError(f"image: unrecognised expression(s): {sorted(bad_expr)}")


def validate_audio_features(df: pd.DataFrame) -> None:
    _validate(df, AUDIO_COLUMNS, AUDIO_FEATURE_COLUMNS, "audio")

    unknown_members = set(df["member"]) - set(config.MEMBERS) - {config.UNKNOWN}
    if unknown_members:
        raise ValueError(f"audio: unrecognised member(s): {sorted(unknown_members)}")

    bad_phrase = set(df["phrase"]) - set(config.PHRASES)
    if bad_phrase:
        raise ValueError(f"audio: unrecognised phrase(s): {sorted(bad_phrase)}")
