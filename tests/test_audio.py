"""Tests for the audio feature pipeline.

These synthesise simple tones instead of using the collected clips, so the tests are self-contained.
Each identity gets a different pitch so the voice model has something to learn.
"""

import numpy as np
import pytest
import soundfile as sf

from src import config, schemas
from src.audio import extract, pipeline
from src.audio.extract import AudioTooShort
from src.models.biometric import voice_model

SR = config.SAMPLE_RATE


def _tone(freq, seconds=2.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.fixture(scope="module")
def signal():
    return _tone(220.0)


# --- the core extractor ---


def test_feature_vector_has_the_contracted_length(signal):
    vec = extract.features_from_signal(signal, SR)
    assert len(vec) == len(schemas.AUDIO_FEATURE_COLUMNS) == 34
    assert not np.isnan(vec).any()


def test_extract_from_a_file(tmp_path, signal):
    path = tmp_path / "taps_approve.wav"
    sf.write(path, signal, SR)
    assert len(extract.extract_audio_features(path)) == 34


def test_clip_too_short_raises(tmp_path):
    with pytest.raises(AudioTooShort):
        extract.features_from_signal(_tone(220.0, seconds=0.05), SR)


# --- augmentations ---


def test_every_configured_augmentation_exists():
    assert set(pipeline.AUGMENTATIONS) == set(config.AUDIO_AUGMENTATIONS)


def test_augmentations_produce_valid_features_and_add_signal(signal):
    original = extract.features_from_signal(signal, SR)
    for name in config.AUDIO_AUGMENTATIONS:
        out = pipeline.augment(signal, SR, name)
        vec = extract.features_from_signal(out, SR)
        assert not np.isnan(vec).any()
        diff = np.abs(vec - original).sum()
        if name == "original":
            assert diff == 0
        else:
            assert diff > 0, f"{name} produced no change"


def test_unknown_augmentation_raises(signal):
    with pytest.raises(ValueError, match="unknown augmentation"):
        pipeline.augment(signal, SR, "reverb")


# --- the CSV builder ---


def _fill_raw_dir(raw_dir, members=None):
    members = members or config.MEMBERS
    # a distinct pitch per identity so the voice model can separate them
    pitches = {m: 180 + 40 * i for i, m in enumerate(members)}
    for m in members:
        for phrase in config.PHRASES:
            sf.write(raw_dir / f"{m}_{phrase}.wav", _tone(pitches[m]), SR)


def test_build_produces_one_row_per_clip_and_augmentation(tmp_path):
    _fill_raw_dir(tmp_path)
    df = pipeline.build_audio_features(tmp_path)

    expected = len(config.MEMBERS) * len(config.PHRASES) * len(config.AUDIO_AUGMENTATIONS)
    assert len(df) == expected
    assert list(df.columns) == schemas.AUDIO_COLUMNS


def test_build_output_passes_the_schema(tmp_path):
    _fill_raw_dir(tmp_path)
    schemas.validate_audio_features(pipeline.build_audio_features(tmp_path))


def test_filename_is_parsed_into_member_and_phrase(tmp_path):
    _fill_raw_dir(tmp_path, members=["taps"])
    df = pipeline.build_audio_features(tmp_path)
    assert set(df["member"]) == {"taps"}
    assert set(df["phrase"]) == set(config.PHRASES)


def test_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no audio"):
        pipeline.build_audio_features(tmp_path)


def test_main_writes_the_csv(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _fill_raw_dir(raw)
    out = tmp_path / "audio_features.csv"
    assert pipeline.main(["--raw-dir", str(raw), "--out", str(out)]) == 0
    assert out.exists()


# --- it feeds the voice model ---


def test_extracted_features_train_the_voice_model(tmp_path):
    _fill_raw_dir(tmp_path)
    df = pipeline.build_audio_features(tmp_path)

    model = voice_model().fit(df)
    assert set(model.classes_) == set(config.MEMBERS)

    sample = df[schemas.AUDIO_FEATURE_COLUMNS].iloc[0].to_numpy()
    result = model.predict(sample)
    assert result.identity in config.MEMBERS
    assert 0.0 <= result.confidence <= 1.0
