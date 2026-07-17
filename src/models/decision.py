"""The multimodal authentication gate.

Implements the flow from the assignment diagram: face first, then voice, then (and only then) a
product recommendation. Both checkpoints must pass, and they must agree with each other.

The design point worth stating in the report: this is not two independent gates. Two independent
gates would accept any known face plus any known voice, so Taps's face with Tedla's voice would
unlock Taps's recommendations. The cross-modal consistency check in `authenticate` — the voice
must identify the *same* member the face did — is what makes the decision genuinely multimodal
rather than a pair of unimodal checks in a row.

This module holds no model code on purpose. It takes each modality's verdict as a plain
`ModalityResult`, so the policy can be tested exhaustively without training anything, and the
face/voice models can change freely underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .. import config

# Confidence floors. Below these, a prediction is treated as "no match" even if it is the
# argmax — with 4 candidate identities the argmax is never worse than chance, so an unknown
# face will always be *some* member unless we impose a floor.
FACE_THRESHOLD = 0.60
VOICE_THRESHOLD = 0.60


class Stage(str, Enum):
    FACE = "face"
    VOICE = "voice"
    GRANTED = "granted"


@dataclass(frozen=True)
class ModalityResult:
    """One model's verdict. `identity` is the predicted member, or None if no match."""

    identity: str | None
    confidence: float


@dataclass(frozen=True)
class Decision:
    granted: bool
    stage: Stage  # where it was granted, or the checkpoint it failed at
    reason: str
    identity: str | None = None
    customer_id: str | None = None

    def __str__(self) -> str:
        verdict = "ACCESS GRANTED" if self.granted else "ACCESS DENIED"
        who = f" [{self.identity}]" if self.identity else ""
        return f"{verdict}{who} at {self.stage.value}: {self.reason}"


def authenticate(
    face: ModalityResult,
    voice: ModalityResult,
    registry: dict[str, str],
    face_threshold: float = FACE_THRESHOLD,
    voice_threshold: float = VOICE_THRESHOLD,
) -> Decision:
    """Run the two checkpoints and resolve the member to a customer.

    Returns a Decision carrying the customer_id to feed the product model when granted, or the
    checkpoint that rejected the attempt when not.
    """
    # --- checkpoint 1: is this a face we know? ---------------------------
    if face.identity is None or face.identity == config.UNKNOWN:
        return Decision(False, Stage.FACE, "face did not match any known user")

    if face.confidence < face_threshold:
        return Decision(
            False,
            Stage.FACE,
            f"face match too weak ({face.confidence:.2f} < {face_threshold:.2f})",
        )

    if face.identity not in registry:
        return Decision(
            False,
            Stage.FACE,
            f"recognised {face.identity} but they have no linked customer record",
            identity=face.identity,
        )

    # --- checkpoint 2: is this an approved voice? ------------------------
    if voice.identity is None or voice.identity == config.UNKNOWN:
        return Decision(
            False, Stage.VOICE, "voice did not match any approved voiceprint", identity=face.identity
        )

    if voice.confidence < voice_threshold:
        return Decision(
            False,
            Stage.VOICE,
            f"voice match too weak ({voice.confidence:.2f} < {voice_threshold:.2f})",
            identity=face.identity,
        )

    # --- the multimodal check: do the two modalities agree? --------------
    if voice.identity != face.identity:
        return Decision(
            False,
            Stage.VOICE,
            f"identity mismatch: face says {face.identity}, voice says {voice.identity}",
            identity=face.identity,
        )

    return Decision(
        True,
        Stage.GRANTED,
        f"face and voice both confirm {face.identity}",
        identity=face.identity,
        customer_id=registry[face.identity],
    )
