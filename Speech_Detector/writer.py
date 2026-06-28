"""
writer.py — Segment file writer and schema validator.

Single job: serialize a segment list to JSON or CSV, and validate
that the output conforms to the downstream contract.
"""

import csv
import json
from pathlib import Path


def write_segments(segments: list[dict], output_path: str) -> None:
    """Write segments to JSON or CSV based on the output file extension."""
    p = Path(output_path)
    if p.suffix == ".csv":
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["start", "end"])
            writer.writeheader()
            writer.writerows({"start": s["start"], "end": s["end"]} for s in segments)
    else:
        with open(p, "w") as f:
            json.dump(segments, f, indent=2)
    print(f"Segments written -> {p}  ({len(segments)} segments)")


def validate_segments(segments: list[dict], audio_duration: float) -> list[str]:
    """
    Validate a segment list against the downstream contract.

    Returns a list of error strings (empty = valid).
    Checks: start < end, sorted, non-overlapping, within bounds, min-duration.
    """
    errors = []
    for i, seg in enumerate(segments):
        if seg["start"] >= seg["end"]:
            errors.append(f"[{i}] start >= end: {seg}")
        if seg["start"] < 0:
            errors.append(f"[{i}] start < 0: {seg}")
        if seg["end"] > audio_duration:
            errors.append(f"[{i}] end {seg['end']:.3f} > audio duration {audio_duration:.3f}")
        if seg["end"] - seg["start"] < 0.5:
            errors.append(f"[{i}] duration < 0.5s: {seg}")

    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i - 1]["start"]:
            errors.append(f"[{i}] not sorted: {segments[i - 1]} -> {segments[i]}")
        if segments[i]["start"] < segments[i - 1]["end"]:
            errors.append(f"[{i}] overlaps previous: {segments[i - 1]} -> {segments[i]}")

    return errors
