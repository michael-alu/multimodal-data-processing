"""The command-line system simulation.

Drives the assignment's flow end to end: a face unlocks the right to attempt a prediction, a voice
confirms it, and only then is a product recommended.

Dependencies are injected rather than imported at module level. That is what lets the whole
transaction be tested today, with the real gate and real models, before the extractors exist —
and it means swapping the stub recommender for the trained one touches nothing here.

Usage:
    python -m src.cli.app --face data/raw/images/taps_neutral.jpg \\
                          --voice data/raw/audio/taps_approve.wav
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..audio.extract import AudioTooShort
from ..images.extract import FaceNotFound
from ..models.biometric import BiometricModel
from ..models.decision import Decision, Stage, check_face, check_voice, grant
from ..models.recommender import Recommender

Extractor = Callable[[Path], np.ndarray]


@dataclass
class AuthApp:
    face_model: BiometricModel
    voice_model: BiometricModel
    recommender: Recommender
    registry: dict[str, str]
    image_extractor: Extractor
    audio_extractor: Extractor

    def run_transaction(
        self,
        face_path: Path,
        voice_path: Path,
        echo: Callable[[str], None] = print,
    ) -> Decision:
        """Attempt one full transaction. Returns the Decision; prints a narration as it goes."""
        echo("")
        echo("=" * 58)
        echo("  USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM")
        echo("=" * 58)

        # --- checkpoint 1: face -----------------------------------------
        echo(f"\n[1/3] Face recognition  ({Path(face_path).name})")
        try:
            features = self.image_extractor(Path(face_path))
        except FaceNotFound:
            decision = Decision(False, Stage.FACE, "no face could be detected in the image")
            return self._finish(decision, echo)

        face = self.face_model.predict(features)
        echo(f"      best match: {face.identity}  (confidence {face.confidence:.2f})")

        denied = check_face(face, self.registry)
        if denied is not None:
            return self._finish(denied, echo)
        assert face.identity is not None
        echo(f"      -> face accepted, {face.identity} may attempt a prediction")

        # --- checkpoint 2: voice ----------------------------------------
        echo(f"\n[2/3] Voice verification  ({Path(voice_path).name})")
        try:
            features = self.audio_extractor(Path(voice_path))
        except AudioTooShort:
            decision = Decision(
                False, Stage.VOICE, "clip too short to verify", identity=face.identity
            )
            return self._finish(decision, echo)

        voice = self.voice_model.predict(features)
        echo(f"      best match: {voice.identity}  (confidence {voice.confidence:.2f})")

        denied = check_voice(voice, face.identity)
        if denied is not None:
            return self._finish(denied, echo)
        echo("      -> voice confirms the same person as the face")

        # --- checkpoint 3: recommend ------------------------------------
        decision = grant(face.identity, self.registry)
        echo(f"\n[3/3] Product recommendation  (customer {decision.customer_id})")
        assert decision.customer_id is not None
        rec = self.recommender.recommend(decision.customer_id)

        self._finish(decision, echo)
        echo(f"  Recommended product: {rec.product}")
        if getattr(self.recommender, "is_stub", False):
            echo("  !! STUB RECOMMENDER — not a real prediction, model not yet trained")
        else:
            echo(f"  Confidence: {rec.confidence:.0%}")
        echo("")
        return decision

    @staticmethod
    def _finish(decision: Decision, echo: Callable[[str], None]) -> Decision:
        echo("")
        echo("-" * 58)
        if decision.granted:
            echo(f"  ACCESS GRANTED — welcome, {decision.identity}")
        else:
            echo(f"  ACCESS DENIED at the {decision.stage.value} checkpoint")
            echo(f"  Reason: {decision.reason}")
        echo("-" * 58)
        return decision


def build_default_app() -> AuthApp:
    """Wire the app from trained models on disk. Requires the pipeline to have been run."""
    from ..images.extract import extract_image_features
    from ..audio.extract import extract_audio_features
    from ..models.recommender import StubRecommender
    from ..models.registry import load_registry
    from .. import config

    return AuthApp(
        face_model=BiometricModel.load(config.FACE_MODEL_PATH),
        voice_model=BiometricModel.load(config.VOICE_MODEL_PATH),
        recommender=StubRecommender(),
        registry=load_registry(),
        image_extractor=extract_image_features,
        audio_extractor=extract_audio_features,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multimodal authentication demo")
    parser.add_argument("--face", type=Path, required=True, help="path to a face image")
    parser.add_argument("--voice", type=Path, required=True, help="path to a voice clip")
    args = parser.parse_args(argv)

    for label, path in (("face", args.face), ("voice", args.voice)):
        if not path.exists():
            print(f"error: {label} file not found: {path}", file=sys.stderr)
            return 2

    decision = build_default_app().run_transaction(args.face, args.voice)
    return 0 if decision.granted else 1


if __name__ == "__main__":
    raise SystemExit(main())
