"""Tests for the face and voice models.

test_grouped_cv_does_not_leak_augmentations builds data with no member signal at all, only a
per-photo signature, and shows a naive random split scoring near perfect on it while the grouped
split correctly scores near chance.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src import config, schemas
from src.models.biometric import BiometricModel, face_model, voice_model

RNG = np.random.default_rng(0)
FEATS = [f"f_{i}" for i in range(20)]


def _table(member_signal: float, source_signal: float, n_sources=3, n_augs=6, seed=0):
    """Synthetic feature table shaped like the real one: member x source x augmentation."""
    rng = np.random.default_rng(seed)
    member_centre = {m: rng.normal(0, 1, len(FEATS)) for m in config.MEMBERS}

    rows = []
    for member in config.MEMBERS:
        for s in range(n_sources):
            source = f"{member}_src{s}.jpg"
            source_centre = rng.normal(0, 1, len(FEATS))
            for a in range(n_augs):
                vec = (
                    member_signal * member_centre[member]
                    + source_signal * source_centre
                    + rng.normal(0, 0.05, len(FEATS))
                )
                row = {"member": member, "source_file": source, "augmentation": f"aug{a}"}
                row.update(dict(zip(FEATS, vec)))
                rows.append(row)
    return pd.DataFrame(rows)


def _model():
    return BiometricModel(feature_columns=FEATS, n_estimators=50)


# --- basic behaviour -----------------------------------------------------


def test_predict_returns_a_modality_result():
    df = _table(member_signal=3.0, source_signal=0.2)
    m = _model().fit(df)
    r = m.predict(df[FEATS].iloc[0].to_numpy())
    assert r.identity in config.MEMBERS
    assert 0.0 <= r.confidence <= 1.0


def test_predict_rejects_wrong_feature_width():
    m = _model().fit(_table(3.0, 0.2))
    with pytest.raises(ValueError, match="Expected 20 features, got 3"):
        m.predict(np.array([1.0, 2.0, 3.0]))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        _model().predict(np.zeros(20))


def test_missing_feature_columns_raises():
    df = _table(3.0, 0.2).drop(columns=["f_0", "f_1"])
    with pytest.raises(ValueError, match="missing 2 column"):
        _model().fit(df)


def test_separable_members_are_learned():
    df = _table(member_signal=4.0, source_signal=0.1)
    metrics = _model().cross_validate(df)
    assert metrics.accuracy > 0.9
    assert metrics.f1_macro > 0.9


# --- the leak ------------------------------------------------------------


def test_grouped_cv_does_not_leak_augmentations():
    """Having seen five augmentations of a photo tells you nothing about a different photo of
    that person, so an honest evaluation must score at chance here."""
    df = _table(member_signal=0.0, source_signal=3.0)
    X, y, groups = df[FEATS].to_numpy(), df["member"].to_numpy(), df["source_file"].to_numpy()

    naive = cross_val_score(
        RandomForestClassifier(n_estimators=50, random_state=0), X, y, cv=5
    ).mean()
    grouped = _model().cross_validate(df).accuracy
    chance = 1.0 / len(config.MEMBERS)

    assert naive > 0.9, f"random split should look near-perfect on leaked data, got {naive:.2f}"
    assert grouped < chance + 0.15, f"grouped split should be ~chance, got {grouped:.2f}"
    assert naive - grouped > 0.5, "the gap is the leak the grouping prevents"


def test_no_source_file_spans_train_and_test():
    from sklearn.model_selection import GroupKFold

    df = _table(2.0, 1.0)
    groups = df["source_file"].to_numpy()
    for train_idx, test_idx in GroupKFold(n_splits=3).split(df[FEATS], df["member"], groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])


# --- metrics -------------------------------------------------------------


def test_metrics_are_populated_and_sane():
    m = _model().cross_validate(_table(3.0, 0.2))
    assert 0.0 <= m.accuracy <= 1.0
    assert 0.0 <= m.f1_macro <= 1.0
    assert m.log_loss >= 0.0
    assert m.n_samples == len(config.MEMBERS) * 3 * 6
    assert m.n_folds == 3
    assert "accuracy=" in str(m) and "log_loss=" in str(m)


def test_cross_validate_needs_at_least_two_groups():
    df = _table(3.0, 0.2, n_sources=1)
    single = df[df["member"] == "taps"]
    with pytest.raises(ValueError, match="at least 2 distinct"):
        _model().cross_validate(single)


def test_fold_that_trains_without_a_member_is_rejected():
    """Unbalanced groups can strand a member outside every training set. That is a broken
    evaluation, not a low score."""
    df = _table(3.0, 0.2)
    # Leave tedla with a single source, so a 3-fold split must train some fold without him.
    thin = df[(df["member"] != "tedla") | (df["source_file"] == "tedla_src0.jpg")]
    with pytest.raises(ValueError, match="trains without"):
        _model().cross_validate(thin, n_folds=3)


# --- the unknown identity ------------------------------------------------


def _with_unknown(n_sources):
    """The team plus a fifth identity trained as a real class."""
    df = _table(member_signal=3.0, source_signal=0.3)
    rng = np.random.default_rng(99)
    centre = rng.normal(0, 1, len(FEATS)) * 3.0
    rows = []
    for s in range(n_sources):
        for a in range(6):
            vec = centre + rng.normal(0, 0.4, len(FEATS))
            row = {
                "member": config.UNKNOWN,
                "source_file": f"unknown_src{s}.jpg",
                "augmentation": f"aug{a}",
            }
            row.update(dict(zip(FEATS, vec)))
            rows.append(row)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def test_unknown_trained_as_a_class_is_predicted_and_denied():
    """This is what makes the unauthorized demo deterministic instead of threshold luck."""
    from src.models.decision import check_face

    registry = {m: f"C{i:03d}" for i, m in enumerate(config.MEMBERS)}
    df = _with_unknown(3)
    model = _model().fit(df)

    intruders = df[df["member"] == config.UNKNOWN][FEATS].to_numpy()
    for vec in intruders[:10]:
        result = model.predict(vec)
        assert result.identity == config.UNKNOWN
        assert check_face(result, registry) is not None, "an unknown face must be denied"


def test_a_single_unknown_photo_is_rejected_not_silently_mis_evaluated():
    """One unknown photo cannot support the split. Better to fail than to score nonsense."""
    with pytest.raises(ValueError, match="trains without"):
        _model().cross_validate(_with_unknown(1))


def test_three_unknown_photos_keep_the_face_model_at_three_folds():
    """The fold count follows the least represented identity, so the intruder needs 3 photos too."""
    assert _model().cross_validate(_with_unknown(2)).n_folds == 2
    assert _model().cross_validate(_with_unknown(3)).n_folds == 3


# --- persistence and wiring ---------------------------------------------


def test_save_load_round_trip_preserves_predictions(tmp_path):
    df = _table(3.0, 0.2)
    m = _model().fit(df)
    sample = df[FEATS].iloc[3].to_numpy()
    before = m.predict(sample)

    path = tmp_path / "m.joblib"
    m.save(path)
    after = BiometricModel.load(path).predict(sample)

    assert before == after


def test_refuses_to_save_unfitted(tmp_path):
    with pytest.raises(RuntimeError, match="unfitted"):
        _model().save(tmp_path / "m.joblib")


def test_face_and_voice_models_bind_the_frozen_schemas():
    assert face_model().feature_columns == schemas.IMAGE_FEATURE_COLUMNS
    assert len(face_model().feature_columns) == 388
    assert voice_model().feature_columns == schemas.AUDIO_FEATURE_COLUMNS
    assert len(voice_model().feature_columns) == 34
