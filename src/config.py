"""Shared constants. Import from here instead of redefining values locally."""

from pathlib import Path

# --- paths ---

ROOT: Path = Path(__file__).resolve().parent.parent
DATA: Path = ROOT / "data"

RAW_IMAGES: Path = DATA / "raw" / "images"
RAW_AUDIO: Path = DATA / "raw" / "audio"
RAW_TABULAR: Path = DATA / "raw" / "tabular"

PROCESSED: Path = DATA / "processed"
FEATURES: Path = DATA / "features"
MODELS: Path = DATA / "models"

IMAGE_FEATURES_CSV: Path = FEATURES / "image_features.csv"
AUDIO_FEATURES_CSV: Path = FEATURES / "audio_features.csv"
MERGED_CSV: Path = PROCESSED / "merged_dataset.csv"

FACE_MODEL_PATH: Path = MODELS / "face.joblib"
VOICE_MODEL_PATH: Path = MODELS / "voice.joblib"
RECOMMENDER_MODEL_PATH: Path = MODELS / "recommender.joblib"

# --- identities ---

MEMBERS: list[str] = ["michael", "taps", "anthony", "tedla"]

# The fifth identity, for the unauthorized demo. Never train on it.
UNKNOWN: str = "unknown"

EXPRESSIONS: list[str] = ["neutral", "smiling", "surprised"]

PHRASES: dict[str, str] = {
    "approve": "Yes, approve",
    "confirm": "Confirm transaction",
}

# --- image pipeline ---

FACE_SIZE: tuple[int, int] = (128, 128)
HIST_BINS: int = 64

HOG_ORIENTATIONS: int = 9
HOG_PIXELS_PER_CELL: tuple[int, int] = (32, 32)
HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)

# A 128x128 image with the params above gives exactly this many HOG features.
HOG_DIM: int = 324

IMAGE_AUGMENTATIONS: list[str] = [
    "original",
    "rotate_p15",
    "rotate_m15",
    "flip_horizontal",
    "brightness_up",
    "gaussian_noise",
]

# --- audio pipeline ---

SAMPLE_RATE: int = 22050
N_MFCC: int = 13

AUDIO_AUGMENTATIONS: list[str] = [
    "original",
    "pitch_shift",
    "time_stretch",
    "background_noise",
]

# --- file naming ---


def image_filename(member: str, expression: str) -> str:
    return f"{member}_{expression}.jpg"


def audio_filename(member: str, phrase_key: str) -> str:
    return f"{member}_{phrase_key}.wav"
