"""
benchmark.py — VAD method comparison.

Single job: run energy, silero, and WebRTC on the same audio file and
print a comparison table of segment count, total speech, and IoU overlap.
"""

import sys
from pathlib import Path

import numpy as np


def _iou(segs_a: list[dict], segs_b: list[dict], audio_duration: float) -> float:
    """Frame-level Intersection over Union between two segment lists."""
    resolution = 0.01  # 10ms bins
    n = int(audio_duration / resolution) + 1
    mask_a = np.zeros(n, dtype=bool)
    mask_b = np.zeros(n, dtype=bool)

    for s in segs_a:
        mask_a[int(s["start"] / resolution): int(s["end"] / resolution)] = True
    for s in segs_b:
        mask_b[int(s["start"] / resolution): int(s["end"] / resolution)] = True

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 1.0


def run_benchmark(audio: np.ndarray, sr: int, wav_path: str) -> None:
    """
    Run energy, silero, and WebRTC VAD on the same audio.
    Prints a comparison table with segment count, total speech, and IoU vs WebRTC.
    """
    from detectors.energy import detect_energy
    from detectors.silero import detect_silero

    # WebRTC VAD — imported from the sibling offline_pipeline
    _pipeline_dir = str(Path(__file__).parent.parent / "speaker_diarization")
    if _pipeline_dir not in sys.path:
        sys.path.insert(0, _pipeline_dir)
    from offline_pipeline import run_vad as webrtc_vad

    audio_duration = len(audio) / sr

    print("\nRunning benchmark...")
    segs_energy = detect_energy(audio, sr)
    segs_silero = detect_silero(audio, sr)
    segs_webrtc = webrtc_vad(wav_path)

    methods = {
        "energy":  segs_energy,
        "silero":  segs_silero,
        "webrtc":  segs_webrtc,
    }

    iou_vs_webrtc = {
        name: _iou(segs, segs_webrtc, audio_duration)
        for name, segs in methods.items()
    }

    print("\n--- Benchmark: VAD method comparison ---")
    print(f"  {'Method':<10} {'Segments':>9} {'Speech(s)':>10} {'IoU vs WebRTC':>14}")
    print(f"  {'-'*47}")
    for name, segs in methods.items():
        total = sum(s["end"] - s["start"] for s in segs)
        iou = iou_vs_webrtc[name]
        iou_str = "  (reference)" if name == "webrtc" else f"  {iou:.3f}"
        print(f"  {name:<10} {len(segs):>9} {total:>10.2f} {iou_str:>14}")
    print()
