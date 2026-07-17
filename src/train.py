"""Train the face and voice models and report their evaluation.

    python -m src.train

Validation happens here on purpose. A feature CSV with a typo'd member or a column of zeros from
a failed extractor would otherwise train a plausible looking model and surface later as a
confusing demo failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config, schemas
from .models.biometric import BiometricModel, Metrics, face_model, voice_model

Validator = Callable[[pd.DataFrame], None]


def _load(csv: Path, validate: Validator) -> pd.DataFrame:
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} does not exist yet, the feature extraction pipeline has not been run."
        )
    df = pd.read_csv(csv)
    validate(df)
    return df


def train_one(model: BiometricModel, df: pd.DataFrame, path: Path) -> Metrics:
    """Cross-validate for honest metrics, then refit on everything for deployment.

    The reported numbers come from the CV, never from the deployed model scoring its own training
    data, which would read about 1.00 and mean nothing.
    """
    metrics = model.cross_validate(df)
    model.fit(df)
    model.save(path)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the face and voice models")
    parser.add_argument("--image-features", type=Path, default=config.IMAGE_FEATURES_CSV)
    parser.add_argument("--audio-features", type=Path, default=config.AUDIO_FEATURES_CSV)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS)
    args = parser.parse_args(argv)

    jobs: list[tuple[str, Path, Validator, BiometricModel, Path]] = [
        (
            "face",
            args.image_features,
            schemas.validate_image_features,
            face_model(),
            args.models_dir / config.FACE_MODEL_PATH.name,
        ),
        (
            "voice",
            args.audio_features,
            schemas.validate_audio_features,
            voice_model(),
            args.models_dir / config.VOICE_MODEL_PATH.name,
        ),
    ]

    results: list[tuple[str, Metrics]] = []
    for name, csv, validate, model, out in jobs:
        print(f"\n=== {name} model ===")
        try:
            df = _load(csv, validate)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            return 1

        print(
            f"  {len(df)} rows from {df['member'].nunique()} members, "
            f"{df['source_file'].nunique()} distinct sources"
        )
        metrics = train_one(model, df, out)
        print(f"  {metrics}")
        print(f"  saved -> {out}")
        results.append((name, metrics))

    print("\n" + "=" * 62)
    print(f"  {'model':<8} {'accuracy':>9} {'f1_macro':>9} {'log_loss':>9}   evaluation")
    print("-" * 62)
    for name, m in results:
        print(
            f"  {name:<8} {m.accuracy:>9.3f} {m.f1_macro:>9.3f} {m.log_loss:>9.3f}   "
            f"{m.n_folds}-fold grouped CV"
        )
    print("=" * 62)
    print("\nMetrics are from grouped cross-validation: augmentations of one photo or clip never")
    print("span the train/test split, so these measure recognising the person, not the recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
