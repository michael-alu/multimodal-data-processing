"""Frozen column schemas for image_features.csv and audio_features.csv.

Build your DataFrame with the column list from this module and call the matching validate function
before writing to disk. Talk to Michael before changing a schema, the model code reads these names.
"""

from __future__ import annotations

import pandas as pd

from . import config

# --- image_features.csv ---

IMAGE_META_COLUMNS: list[str] = ["member", "expression", "augmentation", "source_file"]

IMAGE_HIST_COLUMNS: list[str] = [f"hist_{i}" for i in range(config.HIST_BINS)]
IMAGE_HOG_COLUMNS: list[str] = [f"hog_{i}" for i in range(config.HOG_DIM)]

IMAGE_FEATURE_COLUMNS: list[str] = IMAGE_HIST_COLUMNS + IMAGE_HOG_COLUMNS
IMAGE_COLUMNS: list[str] = IMAGE_META_COLUMNS + IMAGE_FEATURE_COLUMNS

# --- audio_features.csv ---

AUDIO_META_COLUMNS: list[str] = ["member", "phrase", "augmentation", "source_file"]

AUDIO_MFCC_COLUMNS: list[str] = [
    f"mfcc_{i}_mean" for i in range(1, config.N_MFCC + 1)
] + [f"mfcc_{i}_std" for i in range(1, config.N_MFCC + 1)]

AUDIO_SPECTRAL_COLUMNS: list[str] = [
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "rms_energy_mean", "rms_energy_std",
    "zcr_mean", "zcr_std",
]

AUDIO_FEATURE_COLUMNS: list[str] = AUDIO_MFCC_COLUMNS + AUDIO_SPECTRAL_COLUMNS
AUDIO_COLUMNS: list[str] = AUDIO_META_COLUMNS + AUDIO_FEATURE_COLUMNS


# --- validation ---


def _validate(
    df: pd.DataFrame,
    expected: list[str],
    feature_cols: list[str],
    kind: str,
) -> None:
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        raise ValueError(
            f"{kind}: column mismatch.\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(extra)}): {extra[:8]}\n"
            f"  or the order differs from schemas.{kind.upper()}_COLUMNS"
        )

    if df.empty:
        raise ValueError(f"{kind}: no rows")

    feats = df[feature_cols]
    if feats.isna().any().any():
        bad = feats.columns[feats.isna().any()].tolist()
        raise ValueError(f"{kind}: NaN in feature columns: {bad[:8]}")

    # A failed extractor still writes a well formed CSV of zeros. This catches that.
    dead = feats.index[(feats == 0).all(axis=1)].tolist()
    if dead:
        raise ValueError(f"{kind}: {len(dead)} all-zero feature row(s) at index {dead[:5]}")


def validate_image_features(df: pd.DataFrame) -> None:
    _validate(df, IMAGE_COLUMNS, IMAGE_FEATURE_COLUMNS, "image")

    bad_members = set(df["member"]) - set(config.MEMBERS) - {config.UNKNOWN}
    if bad_members:
        raise ValueError(f"image: unrecognised member(s): {sorted(bad_members)}")

    bad_expressions = set(df["expression"]) - set(config.EXPRESSIONS)
    if bad_expressions:
        raise ValueError(f"image: unrecognised expression(s): {sorted(bad_expressions)}")


def validate_audio_features(df: pd.DataFrame) -> None:
    _validate(df, AUDIO_COLUMNS, AUDIO_FEATURE_COLUMNS, "audio")

    bad_members = set(df["member"]) - set(config.MEMBERS) - {config.UNKNOWN}
    if bad_members:
        raise ValueError(f"audio: unrecognised member(s): {sorted(bad_members)}")

    bad_phrases = set(df["phrase"]) - set(config.PHRASES)
    if bad_phrases:
        raise ValueError(f"audio: unrecognised phrase(s): {sorted(bad_phrases)}")
