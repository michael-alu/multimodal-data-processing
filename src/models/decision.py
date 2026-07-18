"""The multimodal authentication gate: face, then voice, then a recommendation.

This is not two independent gates. Two independent gates would accept any known face plus any
known voice, so one member's face with another's voice would unlock the first member's
recommendations. Both modalities must identify the same person.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .. import config

# With 4 candidates the argmax is always someone, so an unknown face still resolves to a member.
# These floors are what make "no match" possible at all.
FACE_THRESHOLD: float = 0.60
VOICE_THRESHOLD: float = 0.60


class Stage(str, Enum):
    FACE = "face"
    VOICE = "voice"
    GRANTED = "granted"


@dataclass(frozen=True)
class ModalityResult:
    """One model's verdict. identity is the predicted member, or None if no match."""

    identity: str | None
    confidence: float


@dataclass(frozen=True)
class Decision:
    granted: bool
    stage: Stage
    reason: str
    identity: str | None = None
    customer_id: str | None = None

    def __str__(self) -> str:
        verdict = "ACCESS GRANTED" if self.granted else "ACCESS DENIED"
        who = f" [{self.identity}]" if self.identity else ""
        return f"{verdict}{who} at {self.stage.value}: {self.reason}"


def check_face(
    face: ModalityResult,
    registry: dict[str, str],
    threshold: float = FACE_THRESHOLD,
) -> Decision | None:
    """Checkpoint 1. Returns a denial, or None if the face passes."""
    if face.identity is None or face.identity == config.UNKNOWN:
        return Decision(False, Stage.FACE, "face did not match any known user")

    if face.confidence < threshold:
        return Decision(
            False,
            Stage.FACE,
            f"face match too weak ({face.confidence:.2f} < {threshold:.2f})",
        )

    if face.identity not in registry:
        return Decision(
            False,
            Stage.FACE,
            f"recognised {face.identity} but they have no linked customer record",
            identity=face.identity,
        )

    return None


def check_voice(
    voice: ModalityResult,
    face_identity: str,
    threshold: float = VOICE_THRESHOLD,
) -> Decision | None:
    """Checkpoint 2. Takes the face's identity because the voice must agree with it."""
    if voice.identity is None or voice.identity == config.UNKNOWN:
        return Decision(
            False,
            Stage.VOICE,
            "voice did not match any approved voiceprint",
            identity=face_identity,
        )

    if voice.confidence < threshold:
        return Decision(
            False,
            Stage.VOICE,
            f"voice match too weak ({voice.confidence:.2f} < {threshold:.2f})",
            identity=face_identity,
        )

    if voice.identity != face_identity:
        return Decision(
            False,
            Stage.VOICE,
            f"identity mismatch: face says {face_identity}, voice says {voice.identity}",
            identity=face_identity,
        )

    return None


def grant(face_identity: str, registry: dict[str, str]) -> Decision:
    return Decision(
        True,
        Stage.GRANTED,
        f"face and voice both confirm {face_identity}",
        identity=face_identity,
        customer_id=registry[face_identity],
    )


def authenticate(
    face: ModalityResult,
    voice: ModalityResult,
    registry: dict[str, str],
    face_threshold: float = FACE_THRESHOLD,
    voice_threshold: float = VOICE_THRESHOLD,
) -> Decision:
    """Run both checkpoints. The CLI calls the checks directly so it can stop before the voice."""
    denied = check_face(face, registry, face_threshold)
    if denied is not None:
        return denied

    assert face.identity is not None
    denied = check_voice(voice, face.identity, voice_threshold)
    if denied is not None:
        return denied

    return grant(face.identity, registry)
