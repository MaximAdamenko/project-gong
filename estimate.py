"""
estimate.py — Step 5: Gather all team outputs and estimate participant counts.

Reads the outputs produced by all four teams and computes:
  - total_participants  : how many people were in the Zoom call
  - speaking_count      : how many of them actually spoke

Sources
-------
  Team 3  Face_Detection/output/faces_output.json      -> total participants (video)
  Team 2  Speech_Detector/output/speakers.json         -> unique audio speakers
  Team 4  Participants_lips_moving/output/lips_output.json -> speaking count (video)

Formula
-------
  total     = Team 3 face count  (stable tile tracking is most reliable for totals)
  speaking  = round( (audio_speakers + lip_speakers) / 2 )
              clamped to [0, total]

Sanity checks (from task spec):
  - speaking > total  => impossible, capped
  - speaking < 70% of total => warning (in this recording ≥70% must have spoken)

Usage
-----
  python estimate.py
  python estimate.py --faces Face_Detection/output/faces_output.json \
                     --speakers Speech_Detector/output/speakers.json \
                     --lips Participants_lips_moving/output/lips_output.json \
                     --output estimate_output.json
"""

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).parent

DEFAULT_FACES    = HERE / "Face_Detection"          / "output" / "faces_output.json"
DEFAULT_SPEAKERS = HERE / "Speech_Detector"         / "output" / "speakers.json"
DEFAULT_LIPS     = HERE / "Participants_lips_moving" / "output" / "lips_output.json"
DEFAULT_OUTPUT   = HERE / "estimate_output.json"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_faces(path: Path) -> int:
    """Team 3: total_participants_detected from face tile tracking."""
    with open(path) as f:
        data = json.load(f)
    return int(data["total_participants_detected"])


def load_speakers(path: Path) -> int:
    """Team 2: count distinct speaker_ids from diarization output."""
    with open(path) as f:
        data = json.load(f)
    unique = set(seg["speaker_id"] for seg in data)
    return len(unique)


def load_lips(path: Path) -> int:
    """Team 4: speaking_count from lip movement detection."""
    with open(path) as f:
        data = json.load(f)
    return int(data["speaking_count"])


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def estimate(total_participants: int, audio_speakers: int, lip_speakers: int):
    """
    Combine all signals into a final estimate.

    Returns (total, speaking, warnings).
    """
    # Average of audio and video speaking signals
    speaking_raw = (audio_speakers + lip_speakers) / 2
    speaking = round(speaking_raw)

    warnings = []

    # Sanity check 1: speaking cannot exceed total
    if speaking > total_participants:
        warnings.append(
            f"speaking estimate ({speaking}) > total participants ({total_participants}) — "
            f"capping at total."
        )
        speaking = total_participants

    # Sanity check 2: in this recording ≥70% should have spoken
    if total_participants > 0 and speaking / total_participants < 0.70:
        warnings.append(
            f"only {speaking}/{total_participants} = "
            f"{speaking / total_participants:.0%} speaking — "
            f"below the expected 70% threshold. Check thresholds."
        )

    return total_participants, speaking, warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 5 — Combine all team outputs into a final participant estimate."
    )
    p.add_argument("--faces",    default=str(DEFAULT_FACES),
                   help=f"faces_output.json from Team 3 (default: {DEFAULT_FACES})")
    p.add_argument("--speakers", default=str(DEFAULT_SPEAKERS),
                   help=f"speakers.json from Team 2 (default: {DEFAULT_SPEAKERS})")
    p.add_argument("--lips",     default=str(DEFAULT_LIPS),
                   help=f"lips_output.json from Team 4 (default: {DEFAULT_LIPS})")
    p.add_argument("--output",   default=str(DEFAULT_OUTPUT),
                   help=f"Where to write the final JSON result (default: {DEFAULT_OUTPUT})")
    return p.parse_args()


def main():
    args = parse_args()

    missing = [p for p in [args.faces, args.speakers, args.lips] if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing input file(s). Run all teams first:")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)

    print("[1/3] Loading Team 3 — Face Detection output...")
    total_participants = load_faces(Path(args.faces))
    print(f"      total participants (video): {total_participants}")

    print("[2/3] Loading Team 2 — Speaker Diarization output...")
    audio_speakers = load_speakers(Path(args.speakers))
    print(f"      unique audio speakers:      {audio_speakers}")

    print("[3/3] Loading Team 4 — Lip Movement output...")
    lip_speakers = load_lips(Path(args.lips))
    print(f"      speaking by lip movement:   {lip_speakers}")

    total, speaking, warnings = estimate(total_participants, audio_speakers, lip_speakers)

    print("\n" + "=" * 50)
    print("  FINAL ESTIMATE")
    print("=" * 50)
    print(f"  Total participants  : {total}")
    print(f"  Speaking participants: {speaking}  ({speaking / total:.0%} of total)" if total else f"  Speaking participants: {speaking}")
    print("=" * 50)

    for w in warnings:
        print(f"\n  WARNING: {w}")

    result = {
        "total_participants": total,
        "speaking_participants": speaking,
        "sources": {
            "face_detection_total": total_participants,
            "audio_unique_speakers": audio_speakers,
            "lip_movement_speakers": lip_speakers,
        },
        "warnings": warnings,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved -> {args.output}")


if __name__ == "__main__":
    main()
