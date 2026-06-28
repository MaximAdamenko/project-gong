"""
detectors/silero.py — Neural VAD using Silero via torch.hub.

Single job: load Silero model, get speech timestamps, convert sample
indices to seconds, return clean speech segments via postprocess_segments().
"""

import numpy as np
from postprocessing import postprocess_segments


def detect_silero(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.5,
    min_gap: float = 0.4,
    min_duration: float = 0.5,
) -> list[dict]:
    """
    Neural VAD using Silero VAD via torch.hub.

    Silero's internal filtering is disabled so postprocess_segments() is
    the sole post-processing gate for both detectors.
    """
    import torch

    print("  Loading Silero VAD model (cached after first run)...")
    model, utils = torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        trust_repo=True,
        verbose=False,
    )
    get_speech_timestamps = utils[0]

    wav_tensor = torch.from_numpy(audio).float()
    raw_timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=0,
        min_silence_duration_ms=0,
        speech_pad_ms=0,
        return_seconds=False,
    )

    raw_segments = [
        {"start": t["start"] / sr, "end": t["end"] / sr}
        for t in raw_timestamps
    ]

    return postprocess_segments(raw_segments, min_gap=min_gap, min_duration=min_duration)
