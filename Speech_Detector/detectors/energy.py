"""
detectors/energy.py — Adaptive RMS energy VAD.

Single job: frame the signal, compute per-frame energy, threshold against
an adaptive noise floor, return clean speech segments via postprocess().
"""

import numpy as np
from postprocessing import postprocess

_FRAME_MS = 30


def detect_energy(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = _FRAME_MS,
    min_gap: float = 0.4,
    min_duration: float = 0.5,
) -> list[dict]:
    """
    Adaptive-threshold energy VAD.

    Frames the signal into non-overlapping windows, computes per-frame RMS
    in dB, sets a noise floor at the 10th percentile of all frame energies
    + 15 dB margin, then flags frames above that floor as speech.
    """
    frame_samples = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_samples

    frames = audio[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    energy_db = 20.0 * np.log10(rms + 1e-10)

    noise_floor_db = np.percentile(energy_db, 10) + 15.0
    flags = (energy_db > noise_floor_db).tolist()

    frame_dur = frame_ms / 1000.0
    return postprocess(flags, frame_dur, min_gap=min_gap, min_duration=min_duration)
