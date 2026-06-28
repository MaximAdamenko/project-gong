"""
Speaker Diarization
-------------------
Input:
  audio_file       - path to audio file (.wav, .mp3, etc.)
  segmentation_file - JSON or CSV with speech segments [{start, end}, ...]

Output:
  speakers_mapping_file - JSON or CSV: [{start, end, speaker_id}, ...]

Usage:
  python speaker_diarization.py audio.wav segments.json speakers.json
  python speaker_diarization.py audio.wav segments.csv speakers.csv --threshold 0.80

Install deps:
  pip install resemblyzer librosa scikit-learn soundfile numpy
"""

import json
import csv
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_segmentation(path: str) -> list[dict]:
    """Load segmentation file produced by Speech Detection team.

    Accepted formats
    ----------------
    JSON: [{"start": 0.5, "end": 3.2}, ...]
    CSV:  start,end\\n0.5,3.2\\n...
    """
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        # normalise field names: support 'start_time'/'end_time' aliases
        segments = []
        for item in data:
            start = next((item[k] for k in ("start", "start_time") if k in item), None)
            end   = next((item[k] for k in ("end",   "end_time")   if k in item), None)
            if start is None or end is None:
                raise ValueError(f"Segment missing start/end: {item}")
            segments.append({"start": float(start), "end": float(end)})
        return segments

    elif p.suffix == ".csv":
        segments = []
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = row.get("start") or row.get("start_time")
                end = row.get("end") or row.get("end_time")
                segments.append({"start": float(start), "end": float(end)})
        return segments

    else:
        raise ValueError(f"Unsupported segmentation format: {p.suffix} (use .json or .csv)")


def save_mapping(result: list[dict], path: str):
    """Save speakers mapping to JSON or CSV."""
    p = Path(path)
    if p.suffix == ".csv":
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["start", "end", "speaker_id"])
            writer.writeheader()
            writer.writerows(result)
    else:
        # default to JSON
        if p.suffix not in (".json",):
            p = p.with_suffix(".json")
        with open(p, "w") as f:
            json.dump(result, f, indent=2)
    return p


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def load_audio(audio_path: str):
    """Load audio as mono float32 numpy array, return (samples, sample_rate)."""
    import librosa
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    return audio.astype(np.float32), sr


def extract_embedding(audio: np.ndarray, sr: int, start: float, end: float, encoder) -> np.ndarray:
    """Extract d-vector speaker embedding for one speech segment."""
    import librosa

    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = audio[start_sample:end_sample]

    # resemblyzer expects 16 kHz mono
    if sr != 16000:
        segment = librosa.resample(segment, orig_sr=sr, target_sr=16000)

    return encoder.embed_utterance(segment)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_speakers(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """
    Agglomerative clustering on cosine distance.

    threshold controls how aggressively segments are merged into the same speaker:
      - higher threshold (e.g. 0.85) -> fewer speakers (merge more aggressively)
      - lower threshold  (e.g. 0.65) -> more speakers (split more aggressively)
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    if len(embeddings) == 1:
        return np.array([0])

    emb_norm = normalize(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - threshold,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(emb_norm)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def diarize(audio_path: str, segmentation_path: str, output_path: str, threshold: float = 0.75):
    """
    Full speaker diarization pipeline.

    1. Load audio
    2. Load speech segments from segmentation file
    3. Extract speaker embedding per segment
    4. Cluster embeddings -> speaker labels
    5. Save speakers mapping file
    """
    from resemblyzer import VoiceEncoder

    print(f"[1/5] Loading audio: {audio_path}")
    audio, sr = load_audio(audio_path)
    print(f"      Duration: {len(audio)/sr:.1f}s  |  Sample rate: {sr} Hz")

    print(f"[2/5] Loading segmentation: {segmentation_path}")
    segments = load_segmentation(segmentation_path)
    print(f"      {len(segments)} speech segments loaded")

    print("[3/5] Loading speaker encoder (resemblyzer d-vector)...")
    encoder = VoiceEncoder()

    print("[4/5] Extracting speaker embeddings...")
    embeddings, valid_segments = [], []

    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        duration = end - start

        if duration < 0.5:
            print(f"      Skipping segment {i+1} ({start:.2f}-{end:.2f}s): too short (<0.5s)")
            continue

        try:
            emb = extract_embedding(audio, sr, start, end, encoder)
            embeddings.append(emb)
            valid_segments.append(seg)
        except Exception as e:
            print(f"      Warning: segment {start:.2f}-{end:.2f}s failed — {e}")

    if not embeddings:
        print("ERROR: No valid segments to process.")
        return []

    print(f"      {len(embeddings)} embeddings extracted")

    print(f"[5/5] Clustering speakers (threshold={threshold})...")
    labels = cluster_speakers(np.array(embeddings), threshold=threshold)

    n_speakers = len(set(labels))
    print(f"      Detected {n_speakers} unique speaker(s)")

    # Build result
    result = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "speaker_id": f"SPEAKER_{int(label):02d}",
        }
        for seg, label in zip(valid_segments, labels)
    ]

    out_path = save_mapping(result, output_path)
    print(f"\nSpeakers mapping saved -> {out_path}")
    print(f"Summary: {n_speakers} speakers | {len(result)} segments")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Speaker Diarization: audio + segmentation -> speakers mapping"
    )
    parser.add_argument("audio", help="Path to audio file (.wav, .mp3, ...)")
    parser.add_argument("segmentation", help="Segmentation file from Speech Detection team (.json or .csv)")
    parser.add_argument("output", help="Output speakers mapping file (.json or .csv)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Speaker similarity threshold 0-1 (default: 0.75). "
             "Higher -> fewer speakers. Lower -> more speakers.",
    )
    args = parser.parse_args()
    diarize(args.audio, args.segmentation, args.output, args.threshold)


if __name__ == "__main__":
    main()
