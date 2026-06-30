"""
audio_worker.py — Teams 1 + 2 (speech detection + speaker diarization) as a
STANDALONE process.

run_all.py launches this as a subprocess instead of a thread. Why a separate
process: the diarization stack (torch / resemblyzer / silero) and the live UI
stack (MediaPipe TF-Lite + YOLO/torch) load conflicting native runtimes. Sharing
one process causes a hard native crash that closes the window and leaves
orphaned audio playing in the background. Separate processes fully isolate them.

Usage:
    python audio_worker.py <video_path> <output_dir>

Writes:
    <output_dir>/meeting.wav, segments.json, speakers.json
    <output_dir>/audio_result.json   -> {"speakers": N}   (the signal run_all reads)
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).parent
for _p in ["Speech_Detector", "speaker_diarization"]:
    sys.path.insert(0, str(HERE / _p))


def main():
    if len(sys.argv) < 3:
        print("usage: python audio_worker.py <video_path> <output_dir>")
        sys.exit(2)

    video_path = sys.argv[1]
    out_dir    = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = out_dir / "meeting.wav"
    seg = out_dir / "segments.json"
    spk = out_dir / "speakers.json"
    res = out_dir / "audio_result.json"

    from audio_io import prepare_audio
    audio, sr = prepare_audio(video_path, output_path=str(wav))

    from detectors.silero import detect_silero
    from writer import write_segments
    segments = detect_silero(audio, sr)
    write_segments(segments, str(seg))

    from speaker_diarization import diarize
    result = diarize(str(wav), str(seg), str(spk), threshold=0.55)
    n = len(set(s["speaker_id"] for s in result)) if result else 0

    with open(res, "w") as f:
        json.dump({"speakers": n}, f)
    print(f"AUDIO_WORKER_DONE speakers={n}")


if __name__ == "__main__":
    main()
