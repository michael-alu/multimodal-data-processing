"""Build image_features.csv from the collected photos.

For every photo in data/raw/images, we detect and crop the face once, then apply each augmentation
to that crop and extract features from it. Augmenting the crop rather than the whole photo means an
augmentation can never break face detection, so every photo yields the full set of rows.

    python -m src.images.pipeline

FOR TAPS: the augmentations live in AUGMENTATIONS below, one small function each. Read them, then
rewrite in your own style. The filenames are parsed as <member>_<expression>.jpg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .. import config, schemas
from .extract import FaceNotFound, detect_and_prepare, features_from_face


def _rotate(face: np.ndarray, degrees: float) -> np.ndarray:
    h, w = face.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(face, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def _flip(face: np.ndarray) -> np.ndarray:
    return cv2.flip(face, 1)


def _brighten(face: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(face, alpha=1.0, beta=40)


def _add_noise(face: np.ndarray) -> np.ndarray:
    noise = np.random.default_rng(0).normal(0, 15, face.shape)
    return np.clip(face.astype(float) + noise, 0, 255).astype(np.uint8)


# Each name here matches config.IMAGE_AUGMENTATIONS.
AUGMENTATIONS = {
    "original": lambda face: face,
    "rotate_p15": lambda face: _rotate(face, 15),
    "rotate_m15": lambda face: _rotate(face, -15),
    "flip_horizontal": _flip,
    "brightness_up": _brighten,
    "gaussian_noise": _add_noise,
}


def augment(face_gray: np.ndarray, name: str) -> np.ndarray:
    """Apply one named augmentation to a prepared grayscale face."""
    if name not in AUGMENTATIONS:
        raise ValueError(f"unknown augmentation {name!r}")
    return AUGMENTATIONS[name](face_gray)


def _parse_name(filename: str) -> tuple[str, str]:
    """Turn 'taps_neutral.jpg' into ('taps', 'neutral')."""
    stem = Path(filename).stem
    member, _, expression = stem.partition("_")
    return member, expression


def build_image_features(raw_dir: Path) -> pd.DataFrame:
    """Extract features for every photo and augmentation into one DataFrame."""
    paths = sorted(raw_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no .jpg images in {raw_dir}")

    rows = []
    skipped = []
    for path in paths:
        member, expression = _parse_name(path.name)
        image = cv2.imread(str(path))
        try:
            face = detect_and_prepare(image)
        except FaceNotFound:
            skipped.append(path.name)
            continue

        for aug_name in AUGMENTATIONS:
            features = features_from_face(augment(face, aug_name))
            row = {
                "member": member,
                "expression": expression,
                "augmentation": aug_name,
                "source_file": path.name,
            }
            row.update(dict(zip(schemas.IMAGE_FEATURE_COLUMNS, features)))
            rows.append(row)

    if skipped:
        print(f"warning: no face found in {len(skipped)} image(s): {skipped}")
    if not rows:
        raise RuntimeError("no features extracted; every image failed face detection")

    return pd.DataFrame(rows)[schemas.IMAGE_COLUMNS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build image_features.csv")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_IMAGES)
    parser.add_argument("--out", type=Path, default=config.IMAGE_FEATURES_CSV)
    args = parser.parse_args(argv)

    df = build_image_features(args.raw_dir)
    schemas.validate_image_features(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} rows from {df['source_file'].nunique()} photos "
          f"({df['member'].nunique()} identities)")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
