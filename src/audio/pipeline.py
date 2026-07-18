"""Build audio_features.csv from the collected voice clips.

For every clip in data/raw/audio, we apply each augmentation to the signal and extract features from
it. Both .wav and .mp3 are accepted (wav is preferred, it is lossless, but mp3 loads fine here).

    python -m src.audio.pipeline

FOR TEDLA: the augmentations live in AUGMENTATIONS below, one small function each. Read them, then
rewrite in your own style. The filenames are parsed as <member>_<phrase>.wav.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from .. import config, schemas
from .extract import AudioTooShort, features_from_signal


def _pitch_shift(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=2)


def _time_stretch(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=1.1)


def _add_noise(y: np.ndarray, sr: int) -> np.ndarray:
    noise = np.random.default_rng(0).normal(0, 0.005, len(y))
    return (y + noise).astype(np.float32)


# Each name here matches config.AUDIO_AUGMENTATIONS.
AUGMENTATIONS = {
    "original": lambda y, sr: y,
    "pitch_shift": _pitch_shift,
    "time_stretch": _time_stretch,
    "background_noise": _add_noise,
}


def augment(y: np.ndarray, sr: int, name: str) -> np.ndarray:
    """Apply one named augmentation to a loaded signal."""
    if name not in AUGMENTATIONS:
        raise ValueError(f"unknown augmentation {name!r}")
    return AUGMENTATIONS[name](y, sr)


def _parse_name(filename: str) -> tuple[str, str]:
    """Turn 'taps_approve.wav' into ('taps', 'approve')."""
    stem = Path(filename).stem
    member, _, phrase = stem.partition("_")
    return member, phrase


AUDIO_EXTENSIONS = ("*.wav", "*.mp3")


def build_audio_features(raw_dir: Path) -> pd.DataFrame:
    """Extract features for every clip and augmentation into one DataFrame."""
    paths = sorted(p for pattern in AUDIO_EXTENSIONS for p in raw_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no audio ({', '.join(AUDIO_EXTENSIONS)}) in {raw_dir}")

    rows = []
    skipped = []
    for path in paths:
        member, phrase = _parse_name(path.name)
        y, sr = librosa.load(str(path), sr=config.SAMPLE_RATE)

        for aug_name in AUGMENTATIONS:
            try:
                features = features_from_signal(augment(y, sr, aug_name), sr)
            except AudioTooShort:
                skipped.append(f"{path.name} ({aug_name})")
                continue
            row = {
                "member": member,
                "phrase": phrase,
                "augmentation": aug_name,
                "source_file": path.name,
            }
            row.update(dict(zip(schemas.AUDIO_FEATURE_COLUMNS, features)))
            rows.append(row)

    if skipped:
        print(f"warning: {len(skipped)} clip(s) too short to use: {skipped}")
    if not rows:
        raise RuntimeError("no features extracted; every clip was too short")

    return pd.DataFrame(rows)[schemas.AUDIO_COLUMNS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build audio_features.csv")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_AUDIO)
    parser.add_argument("--out", type=Path, default=config.AUDIO_FEATURES_CSV)
    args = parser.parse_args(argv)

    df = build_audio_features(args.raw_dir)
    schemas.validate_audio_features(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} rows from {df['source_file'].nunique()} clips "
          f"({df['member'].nunique()} identities)")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
