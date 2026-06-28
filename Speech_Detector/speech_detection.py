"""
speech_detection.py — Pipeline orchestrator.

Single job: run the speech detection pipeline given typed parameters.
No CLI knowledge — that lives in main.py.

Flow:
  LOAD -> DETECT -> WRITE -> (VALIDATE) -> (BENCHMARK)
"""

import os
from pathlib import Path

from audio_io import prepare_audio
from detectors import detect_energy, detect_silero
from writer import write_segments, validate_segments
from benchmark import run_benchmark


# ---------------------------------------------------------------------------
# Verification gates (dev scaffolding)
# ---------------------------------------------------------------------------

def _gate_audio(audio, sr) -> None:
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    duration = len(audio) / sr
    print("\n--- Gate: Audio I/O ---")
    print(f"  Sample rate : {sr} Hz")
    print(f"  Channels    : {channels}")
    print(f"  Duration    : {duration:.2f} s")
    assert sr == 16000, f"FAIL — expected 16000 Hz, got {sr}"
    assert channels == 1, f"FAIL — expected mono, got {channels} channels"
    print("  PASSED\n")


def _gate_postprocessing() -> None:
    from postprocessing import postprocess
    flags = (
        [True] * 5 + [False] * 3 + [True] * 7 +
        [False] * 6 + [True] * 3 +
        [False] * 6 + [True] * 10
    )
    expected = [{"start": 0.0, "end": 1.5}, {"start": 3.0, "end": 4.0}]
    result = postprocess(flags, frame_dur=0.1, min_gap=0.4, min_duration=0.5)
    print("\n--- Gate: Post-processing ---")
    print(f"  Result  : {result}")
    print(f"  Expected: {expected}")
    assert result == expected, f"FAIL — got {result}"
    print("  PASSED\n")


def _gate_detection(segments: list[dict], method: str, audio_duration: float) -> None:
    total = sum(s["end"] - s["start"] for s in segments)
    print(f"\n--- Gate: Detection ({method}) ---")
    print(f"  Segments : {len(segments)}")
    print(f"  Speech   : {total:.2f} s  /  {audio_duration:.2f} s audio")
    assert len(segments) > 0, "FAIL — no segments detected"
    assert all(s["start"] >= 0 and s["end"] <= audio_duration for s in segments), \
        "FAIL — segment out of audio bounds"
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    output_path: str,
    method: str = "silero",
    min_duration: float = 0.5,
    min_gap: float = 0.4,
    validate: bool = False,
    benchmark: bool = False,
    output_dir: str = "output",
) -> bool:
    """
    Run the full speech detection pipeline.

    All output files (meeting.wav, segments, speakers) are written to
    output_dir. Returns True on success, False if validation fails.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    wav_out   = os.path.join(output_dir, "meeting.wav")
    segs_out  = os.path.join(output_dir, Path(output_path).name)

    # --- LOAD ---
    print(f"\n[LOAD] Preparing audio from '{input_path}'...")
    audio, sr = prepare_audio(input_path, output_path=wav_out)
    _gate_audio(audio, sr)
    _gate_postprocessing()

    # --- DETECT ---
    print(f"[DETECT] Running VAD method: {method}")
    if method == "energy":
        segments = detect_energy(audio, sr, min_gap=min_gap, min_duration=min_duration)
    else:
        segments = detect_silero(audio, sr, min_gap=min_gap, min_duration=min_duration)

    audio_duration = len(audio) / sr
    _gate_detection(segments, method, audio_duration)

    # --- WRITE ---
    print(f"[WRITE] Saving segments to '{segs_out}'...")
    write_segments(segments, segs_out)

    # --- VALIDATE ---
    if validate:
        print("[VALIDATE] Checking output against downstream contract...")
        errors = validate_segments(segments, audio_duration)
        if errors:
            print("  FAILED:")
            for e in errors:
                print(f"    {e}")
            return False
        print("  PASSED: output conforms to downstream contract.\n")

    # --- BENCHMARK ---
    if benchmark:
        print("[BENCHMARK] Comparing VAD methods...")
        run_benchmark(audio, sr, wav_out)

    print(f"[DONE] Pipeline complete. Outputs -> {output_dir}/")
    return True
