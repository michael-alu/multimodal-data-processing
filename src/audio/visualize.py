"""Waveform and spectrogram plots for the collected voice clips.

    python -m src.audio.visualize

For each identity we plot the waveform (loudness over time) and the spectrogram (which frequencies
are present over time) for both phrases, and print a short interpretation. This covers the
"plotted and interpreted" part of the audio rubric.

FOR TEDLA: a first version to read and rewrite in your own words. librosa.display does the drawing;
the rest is just loading each clip and laying out the axes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files, do not open a window
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from .. import config


def plot_member(member: str, raw_dir: Path, out_dir: Path) -> Path | None:
    """Save a waveform + spectrogram figure for one member's clips."""
    clips = sorted(raw_dir.glob(f"{member}_*.wav")) + sorted(raw_dir.glob(f"{member}_*.mp3"))
    if not clips:
        return None

    fig, axes = plt.subplots(2, len(clips), figsize=(6 * len(clips), 7), squeeze=False)

    for col, clip in enumerate(clips):
        y, sr = librosa.load(str(clip), sr=config.SAMPLE_RATE)

        # top row: waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0][col])
        axes[0][col].set_title(f"{clip.stem} waveform")
        axes[0][col].set_xlabel("Time (s)")
        axes[0][col].set_ylabel("Amplitude")

        # bottom row: spectrogram in decibels
        spec_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(spec_db, sr=sr, x_axis="time", y_axis="log", ax=axes[1][col])
        axes[1][col].set_title(f"{clip.stem} spectrogram")
        fig.colorbar(img, ax=axes[1][col], format="%+2.0f dB")

    fig.suptitle(f"{member}: waveform and spectrogram")
    fig.tight_layout()

    out = out_dir / f"audio_{member}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def interpret(raw_dir: Path) -> None:
    """Print a short data-driven interpretation to drop into the report."""
    print("\n=== reading the plots (draft interpretation for the report)")
    print("  Waveform: shows loudness over time. The spoken phrase appears as bursts of amplitude")
    print("  with quiet gaps between words, so louder or longer speakers have taller, wider bursts.")
    print("  Spectrogram: shows which frequencies are active over time. Brighter bands are stronger")
    print("  frequencies; the stacked horizontal bands are the harmonics of the voice, and their")
    print("  spacing reflects pitch, which is part of what separates one speaker from another.\n")

    for member in config.MEMBERS + [config.UNKNOWN]:
        clips = sorted(raw_dir.glob(f"{member}_*.wav")) + sorted(raw_dir.glob(f"{member}_*.mp3"))
        if not clips:
            continue
        y, sr = librosa.load(str(clips[0]), sr=config.SAMPLE_RATE)
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        print(f"  {member:<10} duration {len(y) / sr:.2f}s, "
              f"average spectral centroid {centroid:.0f} Hz (brightness of the voice)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot audio waveforms and spectrograms")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_AUDIO)
    parser.add_argument("--out-dir", type=Path, default=config.ROOT / "reports")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for member in config.MEMBERS + [config.UNKNOWN]:
        out = plot_member(member, args.raw_dir, args.out_dir)
        if out is not None:
            saved.append(out)
            print(f"saved -> {out}")

    if not saved:
        raise FileNotFoundError(f"no audio clips found in {args.raw_dir}")

    interpret(args.raw_dir)
    print(f"\nsaved {len(saved)} figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
