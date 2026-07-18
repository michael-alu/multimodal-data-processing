"""Tests for the image feature pipeline.

Real photos have not been collected yet, so these use a stand-in face (the skimage astronaut
sample, which the Haar cascade detects) to exercise the real extraction and augmentation code.
"""

import cv2
import numpy as np
import pytest
from skimage import data

from src import config, schemas
from src.images import extract, pipeline
from src.images.extract import FaceNotFound
from src.models.biometric import face_model


@pytest.fixture(scope="module")
def face_bgr():
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="module")
def prepared_face(face_bgr):
    return extract.detect_and_prepare(face_bgr)


# --- the core extractor ---


def test_detect_and_prepare_returns_a_fixed_size_grayscale_face(prepared_face):
    assert prepared_face.shape == config.FACE_SIZE
    assert prepared_face.ndim == 2  # grayscale


def test_feature_vector_has_the_contracted_length(prepared_face):
    vec = extract.features_from_face(prepared_face)
    assert len(vec) == len(schemas.IMAGE_FEATURE_COLUMNS) == 388
    assert not np.isnan(vec).any()
    assert not (vec == 0).all()


def test_extract_from_a_file(tmp_path, face_bgr):
    path = tmp_path / "taps_neutral.jpg"
    cv2.imwrite(str(path), face_bgr)
    vec = extract.extract_image_features(path)
    assert len(vec) == 388


def test_no_face_raises(tmp_path):
    blank = np.full((200, 200, 3), 127, dtype=np.uint8)
    with pytest.raises(FaceNotFound):
        extract.detect_and_prepare(blank)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract.extract_image_features(tmp_path / "nope.jpg")


# --- augmentations ---


def test_every_configured_augmentation_exists():
    assert set(pipeline.AUGMENTATIONS) == set(config.IMAGE_AUGMENTATIONS)


def test_augmentations_keep_the_face_shape_and_add_signal(prepared_face):
    original = extract.features_from_face(prepared_face)
    for name in config.IMAGE_AUGMENTATIONS:
        out = pipeline.augment(prepared_face, name)
        assert out.shape == config.FACE_SIZE
        diff = np.abs(extract.features_from_face(out) - original).sum()
        if name == "original":
            assert diff == 0
        else:
            assert diff > 0, f"{name} produced no change, so it adds no training signal"


def test_unknown_augmentation_raises(prepared_face):
    with pytest.raises(ValueError, match="unknown augmentation"):
        pipeline.augment(prepared_face, "sepia")


# --- the CSV builder ---


def _fill_raw_dir(raw_dir, face_bgr, members=None):
    members = members or config.MEMBERS
    for m in members:
        for e in config.EXPRESSIONS:
            cv2.imwrite(str(raw_dir / f"{m}_{e}.jpg"), face_bgr)


def test_build_produces_one_row_per_photo_and_augmentation(tmp_path, face_bgr):
    _fill_raw_dir(tmp_path, face_bgr)
    df = pipeline.build_image_features(tmp_path)

    expected = len(config.MEMBERS) * len(config.EXPRESSIONS) * len(config.IMAGE_AUGMENTATIONS)
    assert len(df) == expected
    assert list(df.columns) == schemas.IMAGE_COLUMNS


def test_build_output_passes_the_schema(tmp_path, face_bgr):
    _fill_raw_dir(tmp_path, face_bgr)
    schemas.validate_image_features(pipeline.build_image_features(tmp_path))


def test_filename_is_parsed_into_member_and_expression(tmp_path, face_bgr):
    _fill_raw_dir(tmp_path, face_bgr, members=["taps"])
    df = pipeline.build_image_features(tmp_path)
    assert set(df["member"]) == {"taps"}
    assert set(df["expression"]) == set(config.EXPRESSIONS)


def test_photos_without_a_face_are_skipped_not_fatal(tmp_path, face_bgr, capsys):
    _fill_raw_dir(tmp_path, face_bgr, members=["taps"])
    cv2.imwrite(str(tmp_path / "anthony_neutral.jpg"), np.full((200, 200, 3), 127, np.uint8))

    df = pipeline.build_image_features(tmp_path)
    assert "no face found" in capsys.readouterr().out
    assert set(df["member"]) == {"taps"}


def test_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .jpg"):
        pipeline.build_image_features(tmp_path)


def test_main_writes_the_csv(tmp_path, face_bgr):
    raw = tmp_path / "raw"
    raw.mkdir()
    _fill_raw_dir(raw, face_bgr)
    out = tmp_path / "image_features.csv"

    assert pipeline.main(["--raw-dir", str(raw), "--out", str(out)]) == 0
    assert out.exists()


# --- it feeds the face model ---


def test_extracted_features_train_the_face_model(tmp_path, face_bgr):
    _fill_raw_dir(tmp_path, face_bgr)
    df = pipeline.build_image_features(tmp_path)

    model = face_model().fit(df)
    assert set(model.classes_) == set(config.MEMBERS)

    sample = df[schemas.IMAGE_FEATURE_COLUMNS].iloc[0].to_numpy()
    result = model.predict(sample)
    assert result.identity in config.MEMBERS
    assert 0.0 <= result.confidence <= 1.0
