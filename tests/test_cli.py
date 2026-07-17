"""End to end tests for the CLI simulation.

Uses the real decision policy with faked extractors and models.
"""

from pathlib import Path

import numpy as np

from src.audio.extract import AudioTooShort
from src.images.extract import FaceNotFound
from src.cli.app import AuthApp, main
from src.models.decision import ModalityResult, Stage
from src.models.recommender import Recommendation, Recommender, StubRecommender

REGISTRY = {"michael": "C001", "taps": "C002", "anthony": "C003", "tedla": "C004"}

FACE = Path("data/raw/images/taps_neutral.jpg")
VOICE = Path("data/raw/audio/taps_approve.wav")


class FakeModel:
    def __init__(self, result):
        self.result = result

    def predict(self, features):
        return self.result


class CountingExtractor:
    def __init__(self, raises=None):
        self.calls = 0
        self.raises = raises

    def __call__(self, path):
        self.calls += 1
        if self.raises:
            raise self.raises
        return np.zeros(4)


class CountingRecommender(Recommender):
    def __init__(self):
        self.calls = []

    def recommend(self, customer_id):
        self.calls.append(customer_id)
        return Recommendation("Wireless Earbuds", 0.83)


def build(face_result, voice_result, image_extractor=None, audio_extractor=None, recommender=None):
    return AuthApp(
        face_model=FakeModel(face_result),
        voice_model=FakeModel(voice_result),
        recommender=recommender or CountingRecommender(),
        registry=REGISTRY,
        image_extractor=image_extractor or CountingExtractor(),
        audio_extractor=audio_extractor or CountingExtractor(),
    )


def run(app):
    lines = []
    decision = app.run_transaction(FACE, VOICE, echo=lines.append)
    return decision, "\n".join(lines)


# --- the authorized transaction -----------------------------------------


def test_full_authorized_transaction_reaches_a_recommendation():
    rec = CountingRecommender()
    app = build(ModalityResult("taps", 0.91), ModalityResult("taps", 0.88), recommender=rec)
    decision, out = run(app)

    assert decision.granted
    assert decision.customer_id == "C002"
    assert rec.calls == ["C002"], "recommender should be asked about exactly the resolved customer"
    assert "ACCESS GRANTED" in out
    assert "Wireless Earbuds" in out


def test_authorized_transaction_narrates_all_three_stages():
    decision, out = run(build(ModalityResult("taps", 0.91), ModalityResult("taps", 0.88)))
    assert "[1/3] Face recognition" in out
    assert "[2/3] Voice verification" in out
    assert "[3/3] Product recommendation" in out


# --- the unauthorized attempt -------------------------------------------


def test_unknown_face_is_denied_and_never_asks_for_a_voice():
    """The flow must stop at the face, as in the assignment diagram."""
    audio = CountingExtractor()
    app = build(ModalityResult("unknown", 0.99), ModalityResult("taps", 0.9), audio_extractor=audio)
    decision, out = run(app)

    assert not decision.granted and decision.stage is Stage.FACE
    assert audio.calls == 0, "voice extractor ran despite the face being rejected"
    assert "ACCESS DENIED" in out


def test_low_confidence_face_is_denied_and_short_circuits():
    audio = CountingExtractor()
    app = build(ModalityResult("taps", 0.31), ModalityResult("taps", 0.9), audio_extractor=audio)
    decision, _ = run(app)
    assert not decision.granted and decision.stage is Stage.FACE
    assert audio.calls == 0


def test_stranger_voice_after_valid_face_is_denied():
    decision, out = run(build(ModalityResult("taps", 0.91), ModalityResult("unknown", 0.95)))
    assert not decision.granted and decision.stage is Stage.VOICE
    assert "ACCESS DENIED" in out


def test_face_of_one_member_with_voice_of_another_is_denied():
    decision, out = run(build(ModalityResult("taps", 0.91), ModalityResult("tedla", 0.93)))
    assert not decision.granted and decision.stage is Stage.VOICE
    assert "mismatch" in out


def test_no_denial_path_ever_calls_the_recommender():
    cases = [
        (ModalityResult("unknown", 0.9), ModalityResult("taps", 0.9)),
        (ModalityResult("taps", 0.2), ModalityResult("taps", 0.9)),
        (ModalityResult("taps", 0.9), ModalityResult("unknown", 0.9)),
        (ModalityResult("taps", 0.9), ModalityResult("taps", 0.2)),
        (ModalityResult("taps", 0.9), ModalityResult("tedla", 0.9)),
    ]
    for face, voice in cases:
        rec = CountingRecommender()
        decision, out = run(build(face, voice, recommender=rec))
        assert not decision.granted
        assert rec.calls == [], f"recommender called on a denied attempt: {face} / {voice}"
        assert "Recommended product" not in out


# --- extractor failures --------------------------------------------------


def test_undetectable_face_is_reported_not_crashed():
    app = build(
        ModalityResult("taps", 0.9),
        ModalityResult("taps", 0.9),
        image_extractor=CountingExtractor(raises=FaceNotFound("no face")),
    )
    decision, out = run(app)
    assert not decision.granted and decision.stage is Stage.FACE
    assert "no face could be detected" in out


def test_too_short_clip_is_reported_not_crashed():
    app = build(
        ModalityResult("taps", 0.9),
        ModalityResult("taps", 0.9),
        audio_extractor=CountingExtractor(raises=AudioTooShort("0.1s")),
    )
    decision, out = run(app)
    assert not decision.granted and decision.stage is Stage.VOICE
    assert "too short" in out


# --- stub visibility -----------------------------------------------------


def test_stub_recommender_announces_itself_in_the_output():
    """A stub must never be mistaken for a real prediction in the demo video."""
    app = build(ModalityResult("taps", 0.91), ModalityResult("taps", 0.88), recommender=StubRecommender())
    _, out = run(app)
    assert "STUB RECOMMENDER" in out


def test_real_recommender_shows_confidence_instead():
    _, out = run(build(ModalityResult("taps", 0.91), ModalityResult("taps", 0.88)))
    assert "STUB" not in out
    assert "Confidence: 83%" in out


# --- argv handling -------------------------------------------------------


def test_main_returns_2_when_a_file_is_missing(capsys):
    code = main(["--face", "nope.jpg", "--voice", "nope.wav"])
    assert code == 2
    assert "not found" in capsys.readouterr().err
