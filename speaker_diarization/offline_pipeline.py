"""
Offline Speaker Diarization Pipeline
--------------------------------------
Input:  MP4 Zoom recording (or .wav/.mp3)
Output: speakers_mapping.json  ->  [{start, end, speaker_id}, ...]

Stages:
  1. Extract audio from MP4 via ffmpeg
  2. Run WebRTC VAD -> speech segments (or accept pre-made segmentation file)
  3. Extract d-vector speaker embeddings (resemblyzer)
  4. Agglomerative clustering -> speaker labels
  5. Save speakers mapping

Usage:
  python offline_pipeline.py meeting.mp4 speakers.json
  python offline_pipeline.py meeting.mp4 speakers.csv --threshold 0.78
  python offline_pipeline.py meeting.mp4 speakers.json --segmentation segments.json
  python offline_pipeline.py meeting.wav speakers.json --vad-aggressiveness 3

Install deps:
  pip install -r requirements.txt
  # also requires ffmpeg on PATH: https://ffmpeg.org/download.html
"""

import json
import csv
import glob
import shutil
import subprocess
import wave
import argparse
from pathlib import Path

import numpy as np
import webrtcvad
import librosa
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


SAMPLE_RATE = 16000
FRAME_MS    = 30   # VAD frame size in ms


def find_ffmpeg() -> str:
    """Locate ffmpeg: PATH first, then winget install folder."""
    f = shutil.which("ffmpeg")
    if f:
        return f
    import os
    pattern = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "**", "ffmpeg.exe",
    )
    hits = glob.glob(pattern, recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(
        "ffmpeg not found. Run SETUP.bat or install manually:\n"
        "  winget install Gyan.FFmpeg"
    )


# ---------------------------------------------------------------------------
# Step 1 — Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(input_path: str) -> str:
    """Extract 16 kHz mono WAV from any video/audio file via ffmpeg."""
    out_wav = str(Path(input_path).with_suffix(".wav"))
    cmd = [
        find_ffmpeg(), "-y",
        "-i", input_path,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",   # mono
        "-vn",        # skip video
        "-sample_fmt", "s16",
        out_wav,
    ]
    print(f"[1/4] Extracting audio -> {out_wav}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{r.stderr}")
    return out_wav


# ---------------------------------------------------------------------------
# Step 2 — Voice Activity Detection
# ---------------------------------------------------------------------------

def _read_wave_pcm(path: str) -> tuple[bytes, int]:
    with wave.open(path) as wf:
        assert wf.getnchannels() == 1,  "WAV must be mono"
        assert wf.getsampwidth() == 2,  "WAV must be 16-bit PCM"
        sr = wf.getframerate()
        assert sr in (8000, 16000, 32000, 48000), f"Unsupported rate {sr} for WebRTC VAD"
        return wf.readframes(wf.getnframes()), sr


def run_vad(wav_path: str, aggressiveness: int = 2) -> list[dict]:
    """
    Run WebRTC VAD and return speech segments [{start, end}, ...].

    aggressiveness: 0 (permissive) – 3 (strict noise filtering)
    """
    pcm, sr = _read_wave_pcm(wav_path)
    frame_bytes = int(sr * FRAME_MS / 1000) * 2   # 16-bit = 2 bytes/sample

    vad = webrtcvad.Vad(aggressiveness)
    frames = [pcm[i:i+frame_bytes] for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes)]
    is_speech = [vad.is_speech(f, sr) if len(f) == frame_bytes else False for f in frames]

    # Fill short silence gaps (< 400 ms) between speech bursts
    gap_limit = int(400 / FRAME_MS)
    i = 0
    while i < len(is_speech):
        if not is_speech[i]:
            gap = 0
            j = i
            while j < len(is_speech) and not is_speech[j]:
                gap += 1
                j += 1
            if gap <= gap_limit and i > 0 and j < len(is_speech):
                for k in range(i, j):
                    is_speech[k] = True
            i = j
        else:
            i += 1

    # Convert frame flags -> time segments
    segments, in_seg, seg_start = [], False, 0
    for idx, speech in enumerate(is_speech):
        t = idx * FRAME_MS / 1000.0
        if speech and not in_seg:
            in_seg, seg_start = True, idx
        elif not speech and in_seg:
            in_seg = False
            s = round(seg_start * FRAME_MS / 1000.0, 3)
            e = round(t, 3)
            if e - s >= 0.5:
                segments.append({"start": s, "end": e})
    if in_seg:
        segments.append({
            "start": round(seg_start * FRAME_MS / 1000.0, 3),
            "end":   round(len(is_speech) * FRAME_MS / 1000.0, 3),
        })

    return segments


def load_segmentation_file(path: str) -> list[dict]:
    """Load segmentation file from Speech Detection team (JSON or CSV)."""
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            raw = json.load(f)
        return [{"start": float(next((s[k] for k in ("start","start_time") if k in s), None)),
                 "end":   float(next((s[k] for k in ("end",  "end_time")   if k in s), None))} for s in raw]
    elif p.suffix == ".csv":
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            return [{"start": float(r.get("start") or r.get("start_time")),
                     "end":   float(r.get("end")   or r.get("end_time"))} for r in reader]
    raise ValueError(f"Unsupported segmentation format: {p.suffix}")


# ---------------------------------------------------------------------------
# Step 3 — Speaker embeddings
# ---------------------------------------------------------------------------

def embed_segments(audio: np.ndarray, sr: int,
                   segments: list[dict], encoder: VoiceEncoder) -> tuple:
    """Extract d-vector embedding for each segment. Returns (embeddings, valid_segments)."""
    embeddings, valid = [], []
    for seg in segments:
        s_samp = int(seg["start"] * sr)
        e_samp = int(seg["end"]   * sr)
        chunk  = audio[s_samp:e_samp]
        if sr != 16000:
            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
        try:
            emb = encoder.embed_utterance(chunk)
            embeddings.append(emb)
            valid.append(seg)
        except Exception as ex:
            print(f"  ! Skipping {seg['start']:.2f}–{seg['end']:.2f}s: {ex}")
    return np.array(embeddings), valid


# ---------------------------------------------------------------------------
# Step 4 — Clustering
# ---------------------------------------------------------------------------

def cluster_speakers(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """
    Agglomerative clustering on cosine distance (no need to know # of speakers).

    threshold: cosine similarity above which two embeddings are the same speaker.
      Higher -> fewer, merged speakers. Lower -> more, split speakers.
    """
    if len(embeddings) == 1:
        return np.array([0])
    emb_norm = normalize(embeddings)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - threshold,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(emb_norm)


# ---------------------------------------------------------------------------
# Step 5 — Save output
# ---------------------------------------------------------------------------

def save_mapping(result: list[dict], path: str) -> Path:
    p = Path(path)
    if p.suffix == ".csv":
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["start", "end", "speaker_id"])
            w.writeheader()
            w.writerows(result)
    else:
        if p.suffix != ".json":
            p = p.with_suffix(".json")
        with open(p, "w") as f:
            json.dump(result, f, indent=2)
    return p


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_offline(input_path: str, output_path: str,
                threshold: float = 0.75,
                vad_aggressiveness: int = 2,
                segmentation_path: str = None,
                keep_wav: bool = False) -> list[dict]:

    p = Path(input_path)
    needs_conversion = p.suffix.lower() != ".wav"

    # 1. Extract / convert audio to WAV
    if needs_conversion:
        wav_path = extract_audio(input_path)
    else:
        wav_path = input_path

    # 2. Segmentation
    if segmentation_path:
        print(f"[2/4] Loading segmentation file: {segmentation_path}")
        segments = load_segmentation_file(segmentation_path)
    else:
        print(f"[2/4] Running VAD (aggressiveness={vad_aggressiveness})...")
        segments = run_vad(wav_path, aggressiveness=vad_aggressiveness)
    print(f"       -> {len(segments)} speech segments")

    # 3. Embeddings
    print("[3/4] Extracting speaker embeddings (resemblyzer)...")
    encoder = VoiceEncoder()
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    embeddings, valid_segs = embed_segments(audio, sr, segments, encoder)

    if len(embeddings) == 0:
        print("ERROR: No valid segments after embedding. Check audio quality.")
        return []

    # 4. Cluster + build output
    print(f"[4/4] Clustering speakers (threshold={threshold})...")
    labels = cluster_speakers(embeddings, threshold)
    n_speakers = len(set(labels))

    result = [
        {"start": seg["start"], "end": seg["end"], "speaker_id": f"SPEAKER_{int(l):02d}"}
        for seg, l in zip(valid_segs, labels)
    ]

    out_path = save_mapping(result, output_path)

    if needs_conversion and not keep_wav:
        Path(wav_path).unlink(missing_ok=True)

    print(f"\nDone -> {out_path}")
    print(f"  {n_speakers} unique speaker(s) across {len(result)} segments")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline Speaker Diarization: MP4/WAV -> speakers mapping JSON/CSV"
    )
    parser.add_argument("input",  help="Zoom recording (.mp4, .wav, .mp3, ...)")
    parser.add_argument("output", help="Output speakers mapping (.json or .csv)")
    parser.add_argument("--threshold",          type=float, default=0.75,
                        help="Speaker clustering threshold 0–1 (default: 0.75)")
    parser.add_argument("--vad-aggressiveness", type=int,   default=2, choices=[0,1,2,3],
                        help="WebRTC VAD aggressiveness 0–3 (default: 2). "
                             "Higher = filter more background noise.")
    parser.add_argument("--segmentation",       type=str,   default=None,
                        help="Use pre-made segmentation file instead of running VAD")
    parser.add_argument("--keep-wav",           action="store_true",
                        help="Keep extracted WAV after processing (default: delete)")
    args = parser.parse_args()

    run_offline(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        vad_aggressiveness=args.vad_aggressiveness,
        segmentation_path=args.segmentation,
        keep_wav=args.keep_wav,
    )


if __name__ == "__main__":
    main()
