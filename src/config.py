"""Shared constants for the Formative 2 pipeline.

Everything that more than one workstream needs to agree on lives here. Import from this
module rather than redefining values locally.
"""

from pathlib import Path

# --- paths ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RAW_IMAGES = DATA / "raw" / "images"
RAW_AUDIO = DATA / "raw" / "audio"
RAW_TABULAR = DATA / "raw" / "tabular"

PROCESSED = DATA / "processed"
FEATURES = DATA / "features"

IMAGE_FEATURES_CSV = FEATURES / "image_features.csv"
AUDIO_FEATURES_CSV = FEATURES / "audio_features.csv"
MERGED_CSV = PROCESSED / "merged_dataset.csv"

# --- identities ----------------------------------------------------------

MEMBERS = ["michael", "taps", "anthony", "tedla"]

# The fifth identity: a face and a voice belonging to nobody on the team, used to drive the
# unauthorized-attempt demo. Never include it in the training set for the face/voice models.
UNKNOWN = "unknown"

EXPRESSIONS = ["neutral", "smiling", "surprised"]
PHRASES = {
    "approve": "Yes, approve",
    "confirm": "Confirm transaction",
}

# --- image pipeline ------------------------------------------------------

FACE_SIZE = (128, 128)
HIST_BINS = 64

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (32, 32)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_DIM = 324  # verified: 128x128 image with the params above yields exactly 324 features

IMAGE_AUGMENTATIONS = [
    "original",
    "rotate_p15",
    "rotate_m15",
    "flip_horizontal",
    "brightness_up",
    "gaussian_noise",
]

# --- audio pipeline ------------------------------------------------------

SAMPLE_RATE = 22050
N_MFCC = 13

AUDIO_AUGMENTATIONS = [
    "original",
    "pitch_shift",
    "time_stretch",
    "background_noise",
]

# --- file naming ---------------------------------------------------------


def image_filename(member: str, expression: str) -> str:
    return f"{member}_{expression}.jpg"


def audio_filename(member: str, phrase_key: str) -> str:
    return f"{member}_{phrase_key}.wav"
