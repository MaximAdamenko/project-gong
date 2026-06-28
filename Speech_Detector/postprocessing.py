"""
postprocessing.py — Shared VAD post-processing pipeline.

Single job: transform raw detector output into a clean, validated segment list.

Pipeline: frame flags -> contiguous segments -> gap merge -> drop short -> round -> sort
"""


def _flags_to_raw_segments(flags: list[bool], frame_dur: float) -> list[dict]:
    """Convert boolean frame flags to a raw list of {start, end} dicts."""
    segments = []
    in_seg = False
    seg_start = 0.0
    for i, is_speech in enumerate(flags):
        if is_speech and not in_seg:
            in_seg = True
            seg_start = i * frame_dur
        elif not is_speech and in_seg:
            in_seg = False
            segments.append({"start": seg_start, "end": i * frame_dur})
    if in_seg:
        segments.append({"start": seg_start, "end": len(flags) * frame_dur})
    return segments


def _merge_gaps(segments: list[dict], min_gap: float) -> list[dict]:
    """Merge consecutive segments whose silence gap is shorter than min_gap."""
    if len(segments) < 2:
        return segments
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        gap = round(seg["start"] - merged[-1]["end"], 6)
        if gap < min_gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


def _drop_short(segments: list[dict], min_duration: float) -> list[dict]:
    """Remove segments shorter than min_duration."""
    return [s for s in segments if s["end"] - s["start"] >= min_duration]


def postprocess_segments(
    segments: list[dict],
    min_gap: float = 0.4,
    min_duration: float = 0.5,
) -> list[dict]:
    """
    Apply merge, drop-short, round to 3dp, and sort to a pre-built segment list.
    Used by detectors that already produce {start, end} pairs (e.g. Silero).
    """
    segments = _merge_gaps(segments, min_gap)
    segments = _drop_short(segments, min_duration)
    segments = [{"start": round(s["start"], 3), "end": round(s["end"], 3)} for s in segments]
    segments.sort(key=lambda s: s["start"])
    return segments


def postprocess(
    frame_flags: list[bool],
    frame_dur: float,
    min_gap: float = 0.4,
    min_duration: float = 0.5,
) -> list[dict]:
    """
    Full pipeline for flag-based detectors (energy).
    Converts frame flags to segments then applies postprocess_segments.
    """
    segments = _flags_to_raw_segments(frame_flags, frame_dur)
    return postprocess_segments(segments, min_gap, min_duration)
