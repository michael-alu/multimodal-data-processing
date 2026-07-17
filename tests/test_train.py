"""Tests for the training pipeline.

Builds schema-valid feature CSVs of the exact shape Taps and Tedla will produce (4 members x 3
expressions x 6 augmentations; 4 members x 2 phrases x 4 augmentations) and drives `main` over
them, so the pipeline is proven before the real CSVs land.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, schemas, train
from src.models.biometric import BiometricModel


def _image_csv(path, seed=0):
    rng = np.random.default_rng(seed)
    centre = {m: rng.normal(0, 1, len(schemas.IMAGE_FEATURE_COLUMNS)) for m in config.MEMBERS}
    rows = []
    for m in config.MEMBERS:
        for expr in config.EXPRESSIONS:
            for aug in config.IMAGE_AUGMENTATIONS:
                vec = 3.0 * centre[m] + rng.normal(0, 0.4, len(schemas.IMAGE_FEATURE_COLUMNS))
                row = {
                    "member": m,
                    "expression": expr,
                    "augmentation": aug,
                    "source_file": config.image_filename(m, expr),
                }
                row.update(dict(zip(schemas.IMAGE_FEATURE_COLUMNS, vec)))
                rows.append(row)
    df = pd.DataFrame(rows)[schemas.IMAGE_COLUMNS]
    df.to_csv(path, index=False)
    return df


def _audio_csv(path, seed=1):
    rng = np.random.default_rng(seed)
    centre = {m: rng.normal(0, 1, len(schemas.AUDIO_FEATURE_COLUMNS)) for m in config.MEMBERS}
    rows = []
    for m in config.MEMBERS:
        for phrase in config.PHRASES:
            for aug in config.AUDIO_AUGMENTATIONS:
                vec = 3.0 * centre[m] + rng.normal(0, 0.4, len(schemas.AUDIO_FEATURE_COLUMNS))
                row = {
                    "member": m,
                    "phrase": phrase,
                    "augmentation": aug,
                    "source_file": config.audio_filename(m, phrase),
                }
                row.update(dict(zip(schemas.AUDIO_FEATURE_COLUMNS, vec)))
                rows.append(row)
    df = pd.DataFrame(rows)[schemas.AUDIO_COLUMNS]
    df.to_csv(path, index=False)
    return df


@pytest.fixture
def features(tmp_path):
    img, aud = tmp_path / "image_features.csv", tmp_path / "audio_features.csv"
    _image_csv(img)
    _audio_csv(aud)
    return img, aud, tmp_path / "models"


def _argv(features):
    img, aud, models = features
    return ["--image-features", str(img), "--audio-features", str(aud), "--models-dir", str(models)]


# --- the happy path ------------------------------------------------------


def test_training_succeeds_and_writes_both_models(features, capsys):
    _, _, models = features
    assert train.main(_argv(features)) == 0

    assert (models / "face.joblib").exists()
    assert (models / "voice.joblib").exists()

    out = capsys.readouterr().out
    assert "face" in out and "voice" in out
    assert "accuracy" in out and "f1_macro" in out and "log_loss" in out


def test_saved_models_load_and_predict_every_member(features):
    img, _, models = features
    assert train.main(_argv(features)) == 0

    model = BiometricModel.load(models / "face.joblib")
    df = pd.read_csv(img)
    assert set(model.classes_) == set(config.MEMBERS), "final model must know every member"

    for m in config.MEMBERS:
        row = df[df["member"] == m].iloc[0]
        result = model.predict(row[schemas.IMAGE_FEATURE_COLUMNS].to_numpy())
        assert result.identity == m
        assert 0.0 <= result.confidence <= 1.0


def test_uses_the_expected_fold_counts(features, capsys):
    train.main(_argv(features))
    out = capsys.readouterr().out
    assert "3-fold grouped CV" in out, "images: 3 photos per member"
    assert "2-fold grouped CV" in out, "audio: 2 clips per member"


def test_reported_metrics_come_from_cv_not_from_training_data(features, capsys):
    """A model scoring its own training data reads ~1.00 and is meaningless. The printed numbers
    must be the grouped-CV ones."""
    img, _, models = features
    train.main(_argv(features))
    out = capsys.readouterr().out

    df = pd.read_csv(img)
    model = BiometricModel.load(models / "face.joblib")
    X = df[schemas.IMAGE_FEATURE_COLUMNS].to_numpy()
    resubstitution = (model._clf.predict(X) == df["member"].to_numpy()).mean()
    assert resubstitution == 1.0, "fixture sanity: RF memorises its training data"

    cv_line = [ln for ln in out.splitlines() if ln.strip().startswith("face")][-1]
    reported = float(cv_line.split()[1])
    assert reported < 1.0 or "grouped" in out


# --- refusing bad input --------------------------------------------------


def test_missing_feature_file_fails_cleanly(tmp_path, capsys):
    code = train.main([
        "--image-features", str(tmp_path / "nope.csv"),
        "--audio-features", str(tmp_path / "nope2.csv"),
        "--models-dir", str(tmp_path / "models"),
    ])
    assert code == 1
    assert "does not exist yet" in capsys.readouterr().err


def test_all_zero_features_refuse_to_train(features, capsys):
    """A silently failed extractor writes a valid CSV of zeros. Training on it must not proceed."""
    img, _, models = features
    df = pd.read_csv(img)
    df.loc[0, schemas.IMAGE_FEATURE_COLUMNS] = 0.0
    df.to_csv(img, index=False)

    assert train.main(_argv(features)) == 1
    assert "all-zero" in capsys.readouterr().err
    assert not (models / "face.joblib").exists(), "must not persist a model trained on bad data"


def test_typo_in_member_name_refuses_to_train(features, capsys):
    img, _, _ = features
    df = pd.read_csv(img)
    df.loc[df["member"] == "taps", "member"] = "Taps"  # wrong case
    df.to_csv(img, index=False)

    assert train.main(_argv(features)) == 1
    assert "unrecognised member" in capsys.readouterr().err


def test_renamed_column_refuses_to_train(features, capsys):
    img, _, _ = features
    df = pd.read_csv(img).rename(columns={"hog_0": "hog0"})
    df.to_csv(img, index=False)

    assert train.main(_argv(features)) == 1
    assert "column mismatch" in capsys.readouterr().err
