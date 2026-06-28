"""
Live Speaker Diarization
--------------------------
Captures real-time audio from microphone (or virtual audio device for Zoom),
detects who is speaking, and outputs speaker labels in real-time.

Algorithm:
  1. Capture 30ms audio frames via PyAudio
  2. WebRTC VAD detects speech vs. silence per frame
  3. Accumulate frames into utterances (silence-bounded)
  4. Extract d-vector embedding (resemblyzer) per utterance
  5. Compare to known speaker profiles (cosine similarity):
       sim ≥ threshold -> same speaker (update profile)
       sim < threshold -> new speaker (register profile)
  6. Print speaker label instantly + save full log to JSON

Usage:
  python live_pipeline.py                             # mic default
  python live_pipeline.py --list-devices              # show devices
  python live_pipeline.py --device 2                  # specific device
  python live_pipeline.py --output live_log.json      # save output
  python live_pipeline.py --threshold 0.80            # stricter separation

For Zoom audio capture:
  Windows: install VB-Cable (virtual audio device), set Zoom output to CABLE Input,
           then select "CABLE Output" with --device <index>

Install deps:
  pip install -r requirements.txt
"""

import json
import threading
import queue
import struct
import time
import argparse
from pathlib import Path

import numpy as np
import pyaudio
import webrtcvad
from resemblyzer import VoiceEncoder


SAMPLE_RATE    = 16000
FRAME_MS       = 30
FRAME_SAMPLES  = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES    = FRAME_SAMPLES * 2   # int16 = 2 bytes/sample


# ---------------------------------------------------------------------------
# Speaker profile tracker (online, thread-safe)
# ---------------------------------------------------------------------------

class SpeakerTracker:
    """
    Maintains a registry of speaker d-vector profiles.
    Identifies new utterances against registered profiles.
    Profile is updated with exponential moving average on each match.
    """

    def __init__(self, threshold: float = 0.75, ema_alpha: float = 0.3):
        self.threshold  = threshold
        self.ema_alpha  = ema_alpha       # weight of new embedding in profile update
        self._profiles: list[tuple[str, np.ndarray]] = []
        self._lock      = threading.Lock()

    def identify(self, embedding: np.ndarray) -> tuple[str, float]:
        """
        Return (speaker_id, cosine_similarity).
        Registers a new speaker profile if no match above threshold.
        """
        emb = embedding / (np.linalg.norm(embedding) + 1e-9)

        with self._lock:
            best_id, best_sim = None, -1.0

            for spk_id, profile in self._profiles:
                sim = float(np.dot(emb, profile))
                if sim > best_sim:
                    best_sim, best_id = sim, spk_id

            if best_id is not None and best_sim >= self.threshold:
                # Update existing profile with EMA
                idx = next(i for i, (sid, _) in enumerate(self._profiles) if sid == best_id)
                old_prof = self._profiles[idx][1]
                new_prof = self.ema_alpha * old_prof + (1 - self.ema_alpha) * emb
                new_prof /= np.linalg.norm(new_prof)
                self._profiles[idx] = (best_id, new_prof)
                return best_id, best_sim
            else:
                # Register new speaker
                new_id = f"SPEAKER_{len(self._profiles):02d}"
                self._profiles.append((new_id, emb))
                return new_id, 1.0

    @property
    def n_speakers(self) -> int:
        return len(self._profiles)


# ---------------------------------------------------------------------------
# Audio capture (non-blocking, callback-based)
# ---------------------------------------------------------------------------

class AudioCapture:
    """Streams 30ms PCM frames from a PyAudio input device into a queue."""

    def __init__(self, device_index: int = None):
        self._pa          = pyaudio.PyAudio()
        self.device_index = device_index
        self.frame_queue: queue.Queue[bytes] = queue.Queue()
        self._stream      = None

    def start(self):
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=FRAME_SAMPLES,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        self.frame_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()

    @staticmethod
    def list_devices():
        pa = pyaudio.PyAudio()
        print("\nAvailable audio input devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i:2d}]  {info['name']}")
        pa.terminate()
        print()


# ---------------------------------------------------------------------------
# PCM helpers
# ---------------------------------------------------------------------------

def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM bytes to float32 numpy array in [-1, 1]."""
    samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
    return np.array(samples, dtype=np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Main live loop
# ---------------------------------------------------------------------------

def live_diarize(
    device_index: int    = None,
    output_path: str     = None,
    threshold: float     = 0.75,
    vad_aggressiveness   = 2,
    min_speech_ms: int   = 600,    # ignore utterances shorter than this
    max_silence_ms: int  = 500,    # silence duration that ends an utterance
):
    """
    Real-time speaker diarization.
    Press Ctrl+C to stop and save results.
    """
    vad     = webrtcvad.Vad(vad_aggressiveness)
    encoder = VoiceEncoder()
    tracker = SpeakerTracker(threshold=threshold)
    capture = AudioCapture(device_index=device_index)
    results = []

    silence_limit = int(max_silence_ms / FRAME_MS)   # frames of silence to end utterance
    min_frames    = int(min_speech_ms  / FRAME_MS)   # minimum speech frames to process

    print(f"\n{'─'*58}")
    print(f"  Live Speaker Diarization  |  Ctrl+C to stop")
    print(f"{'─'*58}")
    print(f"  Device       : {'default' if device_index is None else device_index}")
    print(f"  Threshold    : {threshold}")
    print(f"  VAD level    : {vad_aggressiveness}")
    print(f"{'─'*58}\n")

    capture.start()
    session_start  = time.time()
    speech_frames  = []
    silence_count  = 0
    in_speech      = False
    seg_start_abs  = 0.0

    try:
        while True:
            try:
                frame = capture.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            now = time.time() - session_start

            try:
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
            except Exception:
                is_speech = False

            if is_speech:
                if not in_speech:
                    in_speech     = True
                    seg_start_abs = now
                    speech_frames = []
                    silence_count = 0
                speech_frames.append(frame)
                silence_count = 0

            else:
                if in_speech:
                    silence_count += 1
                    speech_frames.append(frame)   # keep trailing silence for natural audio

                    if silence_count >= silence_limit:
                        in_speech   = False
                        seg_end_abs = now

                        if len(speech_frames) >= min_frames:
                            pcm_all  = b"".join(speech_frames)
                            audio_np = pcm_bytes_to_float32(pcm_all)

                            try:
                                emb = encoder.embed_utterance(audio_np)
                                speaker_id, sim = tracker.identify(emb)
                                duration = seg_end_abs - seg_start_abs

                                entry = {
                                    "start":      round(seg_start_abs, 2),
                                    "end":        round(seg_end_abs,   2),
                                    "speaker_id": speaker_id,
                                    "confidence": round(float(sim), 3),
                                }
                                results.append(entry)

                                tag = "[NEW]" if sim == 1.0 and tracker.n_speakers > len(results) - 1 else "     "
                                print(f"  {tag} [{seg_start_abs:7.2f}s -> {seg_end_abs:7.2f}s]  "
                                      f"{speaker_id}  "
                                      f"sim={sim:.2f}  dur={duration:.1f}s")

                            except Exception as e:
                                print(f"  ! Embedding error: {e}")

                        speech_frames = []

    except KeyboardInterrupt:
        pass

    finally:
        capture.stop()

    total_speakers = len(set(r["speaker_id"] for r in results)) if results else 0
    print(f"\n{'─'*58}")
    print(f"  Stopped — {total_speakers} speaker(s) | {len(results)} utterance(s)")

    if output_path and results:
        p = Path(output_path)
        with open(p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {p}")

    print(f"{'─'*58}\n")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live Speaker Diarization — real-time who-is-talking detection"
    )
    parser.add_argument("--device",             type=int,   default=None,
                        help="Audio input device index (run --list-devices to find index)")
    parser.add_argument("--output",             type=str,   default=None,
                        help="Save full session log to JSON file")
    parser.add_argument("--threshold",          type=float, default=0.75,
                        help="Speaker similarity threshold 0–1 (default: 0.75). "
                             "Higher = fewer speakers merged.")
    parser.add_argument("--vad-aggressiveness", type=int,   default=2, choices=[0,1,2,3],
                        help="WebRTC VAD aggressiveness 0–3 (default: 2)")
    parser.add_argument("--list-devices",       action="store_true",
                        help="Print available audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        AudioCapture.list_devices()
        return

    live_diarize(
        device_index=args.device,
        output_path=args.output,
        threshold=args.threshold,
        vad_aggressiveness=args.vad_aggressiveness,
    )


if __name__ == "__main__":
    main()
