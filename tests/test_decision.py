"""Decision-table coverage for the multimodal gate.

The gate is pure policy over two verdicts, so every path is reachable without a trained model.
"""

import pandas as pd
import pytest

from src import config
from src.models import registry as reg
from src.models.decision import Decision, ModalityResult, Stage, authenticate

REGISTRY = {"michael": "C001", "taps": "C002", "anthony": "C003", "tedla": "C004"}


def ok(identity, conf=0.95):
    return ModalityResult(identity, conf)


# --- the granted path ----------------------------------------------------


def test_matching_face_and_voice_grants_and_resolves_customer():
    d = authenticate(ok("taps"), ok("taps"), REGISTRY)
    assert d.granted
    assert d.stage is Stage.GRANTED
    assert d.identity == "taps"
    assert d.customer_id == "C002"


def test_every_member_can_authenticate():
    for member in config.MEMBERS:
        d = authenticate(ok(member), ok(member), REGISTRY)
        assert d.granted, f"{member} should authenticate"
        assert d.customer_id == REGISTRY[member]


# --- checkpoint 1: face --------------------------------------------------


def test_unknown_face_denied_at_face_stage():
    d = authenticate(ok(config.UNKNOWN), ok("taps"), REGISTRY)
    assert not d.granted and d.stage is Stage.FACE
    assert d.customer_id is None


def test_no_face_match_denied_at_face_stage():
    d = authenticate(ModalityResult(None, 0.0), ok("taps"), REGISTRY)
    assert not d.granted and d.stage is Stage.FACE


def test_low_confidence_face_denied_even_when_identity_is_right():
    d = authenticate(ok("taps", 0.42), ok("taps"), REGISTRY)
    assert not d.granted and d.stage is Stage.FACE
    assert "too weak" in d.reason


def test_face_at_exactly_threshold_is_accepted():
    d = authenticate(ok("taps", 0.60), ok("taps"), REGISTRY)
    assert d.granted, "threshold is a floor, not an exclusive bound"


def test_recognised_member_without_customer_record_denied():
    d = authenticate(ok("tedla"), ok("tedla"), {"taps": "C002"})
    assert not d.granted and d.stage is Stage.FACE
    assert "no linked customer" in d.reason


# --- checkpoint 2: voice -------------------------------------------------


def test_unknown_voice_denied_at_voice_stage():
    d = authenticate(ok("taps"), ok(config.UNKNOWN), REGISTRY)
    assert not d.granted and d.stage is Stage.VOICE


def test_low_confidence_voice_denied():
    d = authenticate(ok("taps"), ok("taps", 0.11), REGISTRY)
    assert not d.granted and d.stage is Stage.VOICE
    assert "too weak" in d.reason


# --- the multimodal check ------------------------------------------------


def test_known_face_with_a_different_members_voice_is_denied():
    """The whole point of the design: two valid credentials from different people must fail."""
    d = authenticate(ok("taps"), ok("tedla"), REGISTRY)
    assert not d.granted
    assert d.stage is Stage.VOICE
    assert "mismatch" in d.reason
    assert d.customer_id is None, "a denied attempt must never leak a customer_id"


def test_mismatch_is_denied_for_every_cross_pairing():
    for face_id in config.MEMBERS:
        for voice_id in config.MEMBERS:
            d = authenticate(ok(face_id), ok(voice_id), REGISTRY)
            assert d.granted == (face_id == voice_id), f"{face_id} face + {voice_id} voice"


def test_denied_decisions_never_carry_a_customer_id():
    denials = [
        authenticate(ok(config.UNKNOWN), ok("taps"), REGISTRY),
        authenticate(ok("taps", 0.1), ok("taps"), REGISTRY),
        authenticate(ok("taps"), ok(config.UNKNOWN), REGISTRY),
        authenticate(ok("taps"), ok("taps", 0.1), REGISTRY),
        authenticate(ok("taps"), ok("tedla"), REGISTRY),
    ]
    for d in denials:
        assert not d.granted
        assert d.customer_id is None


# --- registry ------------------------------------------------------------


def _merged(n=10):
    return pd.DataFrame({"customer_id": [f"C{i:03d}" for i in range(1, n + 1)]})


def test_build_registry_is_deterministic_and_one_to_one():
    a = reg.build_registry(_merged())
    b = reg.build_registry(_merged())
    assert a == b, "must be reproducible or the report's worked example rots"
    assert len(set(a.values())) == len(config.MEMBERS)
    assert set(a) == set(config.MEMBERS)


def test_build_registry_rejects_too_few_customers():
    with pytest.raises(ValueError, match="only 2 distinct customers"):
        reg.build_registry(_merged(n=2))


def test_validate_registry_catches_duplicate_customer():
    dupe = {"michael": "C001", "taps": "C001", "anthony": "C003", "tedla": "C004"}
    with pytest.raises(ValueError, match="1:1"):
        reg.validate_registry(dupe, _merged())


def test_validate_registry_catches_dangling_customer():
    dangling = dict(REGISTRY, tedla="C999")
    with pytest.raises(ValueError, match="absent from the merged dataset"):
        reg.validate_registry(dangling, _merged())


def test_round_trip_through_disk(tmp_path):
    path = tmp_path / "identity_registry.json"
    reg.save_registry(REGISTRY, path)
    assert reg.load_registry(path) == REGISTRY


def test_load_registry_rejects_incomplete_mapping(tmp_path):
    path = tmp_path / "r.json"
    reg.save_registry({"taps": "C002"}, path)
    with pytest.raises(ValueError, match="missing members"):
        reg.load_registry(path)
