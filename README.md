# Project Gong — Zoom Meeting Speaker Counter

Estimates **how many participants** were in a Zoom recording and **how many of them actually spoke**, by combining four independent analysis pipelines — two audio-based and two video-based.

---

## Quick Start — One Command

```bash
pip install -r requirements.txt
python run_all.py --sound
```

Opens a live window showing:
- Face bounding boxes updating in real time (green = speaking, gray = silent)
- Side panel with per-participant speaking status and live counts
- Audio speaker analysis (Teams 1+2) running in a background **process**
- Final estimate when the audio analysis finishes

```
python run_all.py                          # uses Gong_Test_VIDEO.mp4, no sound
python run_all.py --sound                  # play the video's audio (implies --realtime)
python run_all.py --realtime               # play the whole video at true speed, muted
python run_all.py meeting.mp4 --sound      # your own video, with sound
python run_all.py meeting.mp4 --no-audio   # skip speaker analysis (video-only, faster)
python run_all.py meeting.mp4 --slot-radius 0.10   # tune two-people-per-tile splitting
```

**Controls (while the window is focused):** `Q` / `Esc` to quit.

> **Requires `ffmpeg` *and* `ffplay` on PATH** (both ship with ffmpeg):
> `winget install Gyan.FFmpeg` (Windows) / `brew install ffmpeg` (macOS).
> After installing on Windows, **open a new terminal** so PATH is picked up.

> Team 4 model `face_landmarker.task` (~30 MB) downloads automatically on first run.

### How it runs (and why it's robust)

`run_all.py` is the GUI/orchestrator: it loads YOLOv8-Face + MediaPipe, discovers
participants by **sampling across the whole video** (so people whose tile is only
visible part of the time are still counted), then plays the video while detecting
faces and lip movement live. The speaker analysis (Teams 1+2 — torch / Silero /
Resemblyzer) runs in a **separate process** (`audio_worker.py`): its native
runtime conflicts with MediaPipe+YOLO in one process and would crash the window,
so it is fully isolated. On exit, every subprocess is terminated — nothing is
left running in the background.

| Tuning (top of `run_all.py`) | Meaning |
|------------------------------|---------|
| `SLOT_RADIUS` (0.14) | tile clustering radius — lower to split two people in one tile, raise if one face splits in two |
| `STD_ON` / `STD_OFF` (0.045 / 0.030) | lip-movement speaking thresholds (hysteresis — stops smiles/jitter false positives) |
| `MIN_PRESENCE` (0.10) | fraction of scanned frames a tile must appear in to count as a participant |

---

## Goal

Given a Zoom meeting recording (`Gong_Test_VIDEO.mp4`), output:

| Metric | Source |
|--------|--------|
| Total participants | Team 3 — face detection (video) |
| Speaking participants | Average of Team 2 (audio diarization) + Team 4 (lip movement) |

---

## System Overview

```
Gong_Test_VIDEO.mp4
        │
        ├──► [Team 1] Speech_Detector          → output/meeting.wav
        │                                         output/segments.json
        │          │
        │          └──► [Team 2] speaker_diarization → output/speakers.json
        │
        └──► [Team 3] Face_Detection           → output/faces_output.json
                   │
                   └──► [Team 4] Participants_lips_moving → output/lips_output.json
                                                                     │
                                                        [Step 5] estimate.py
                                                                     │
                                                           estimate_output.json
                                                    (total + speaking participants)
```

---

## Project Structure

```
project-gong/
├── Gong_Test_VIDEO.mp4                    ← shared input (all teams read from here)
├── run_all.py                             ← one-command live UI (GUI + orchestrator)
├── audio_worker.py                        ← Teams 1+2 as an isolated subprocess
├── presentation.html                      ← self-contained slide deck (open in a browser)
├── estimate.py                            ← Step 5: combine all outputs → final count
├── estimate_output.json                   ← final result (generated)
│
├── Speech_Detector/                       ← Team 1 · Voice Activity Detection
│   ├── main.py
│   ├── speech_detection.py
│   ├── audio_io.py
│   ├── postprocessing.py
│   ├── detectors/
│   │   ├── energy.py                      ← RMS energy VAD
│   │   └── silero.py                      ← Silero neural VAD (primary, chosen)
│   ├── writer.py
│   ├── benchmark.py
│   ├── requirements.txt
│   └── output/
│       ├── meeting.wav                    ← 16 kHz mono WAV (shared with Team 2)
│       └── segments.json                  ← [{start, end}, ...]
│
├── speaker_diarization/                   ← Team 2 · Speaker Diarization
│   ├── speaker_diarization.py
│   ├── offline_pipeline.py
│   ├── live_pipeline.py
│   ├── run_my_video.py
│   └── requirements.txt
│   (output → Speech_Detector/output/speakers.json)
│
├── Face_Detection/                        ← Team 3 · Face Detection + Tile Tracking
│   ├── main.py
│   ├── face_detection.py
│   ├── detector.py                        ← YOLOv8-Face + stable slot clustering
│   ├── renderer.py                        ← annotated video output
│   ├── writer.py
│   ├── requirements.txt
│   └── output/
│       ├── faces_output.json              ← per-frame bounding boxes + stable face_id
│       └── annotated_output.mp4           ← video with face boxes drawn
│
└── Participants_lips_moving/              ← Team 4 · Lip Movement Detection
    ├── main.py
    ├── lip_analysis.py
    ├── loader.py
    ├── landmarker.py                      ← MediaPipe MAR measurement
    ├── detectors/
    │   ├── std_detector.py                ← MAR std deviation (7/7 accuracy)
    │   └── fft_detector.py                ← FFT speech-band power (6/7 accuracy)
    ├── reporter.py
    ├── face_landmarker.task               ← MediaPipe model (download once, see setup)
    ├── requirements.txt
    └── output/
        └── lips_output.json               ← {speaking_count, per_face decisions}
```

---

## Prerequisites

All teams require **Python 3.9+** and **ffmpeg** on PATH:

```bash
# Windows
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg
```

---

## End-to-End Run

Run each team in order, then run `estimate.py`. All commands are from the repo root.

### Team 1 — Speech Detection

```bash
cd Speech_Detector
pip install -r requirements.txt

# Default run (Silero VAD, outputs to output/)
python main.py ../Gong_Test_VIDEO.mp4 output/segments.json

# With validation and benchmark comparison
python main.py ../Gong_Test_VIDEO.mp4 output/segments.json --validate --benchmark
```

Output: `output/meeting.wav`, `output/segments.json`

| Method | Segments | Total Speech | IoU vs WebRTC |
|--------|----------|--------------|---------------|
| energy | 90 | 620.10 s | 0.944 |
| **silero** | **153** | **530.18 s** | **0.880** |
| webrtc | 106 | 601.80 s | (reference) |

**Chosen:** Silero — finer segmentation gives cleaner speaker embeddings for Team 2.

---

### Team 2 — Speaker Diarization

Reads `meeting.wav` + `segments.json` from Team 1.

```bash
cd Speech_Detector
pip install resemblyzer

python ../speaker_diarization/speaker_diarization.py \
    output/meeting.wav \
    output/segments.json \
    output/speakers.json \
    --threshold 0.55
```

Output: `Speech_Detector/output/speakers.json`

> **Threshold note:** lower value = fewer speakers merged. Sweet spot for Zoom: `0.50–0.60`.

---

### Team 3 — Face Detection

```bash
cd Face_Detection
pip install -r requirements.txt

# Default run
python main.py

# Quick test (~30 s of video)
python main.py --max-frames 900

# Full options
python main.py ../Gong_Test_VIDEO.mp4 output/faces_output.json \
    --annotated output/annotated_output.mp4 \
    --slot-radius 0.16 \
    --min-presence 0.08 \
    --validate
```

Output: `output/faces_output.json`, `output/annotated_output.mp4`

**Test result:** 7 participants detected across 22,008 frames.

Output contract (→ Team 4):
```json
{
  "total_participants_detected": 7,
  "fps": 29.26,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "faces_detected": [
        { "face_id": "face_01", "bounding_box": { "x_min": 0.78, "y_min": 0.10, "width": 0.12, "height": 0.10 } }
      ]
    }
  ]
}
```

---

### Team 4 — Lip Movement Detection

Reads `faces_output.json` from Team 3. Requires the MediaPipe model file (one-time download):

```bash
cd Participants_lips_moving

# Download MediaPipe model (once)
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

pip install -r requirements.txt

# Default run (std method, 7/7 accuracy on test video)
python main.py

# FFT method (6/7 accuracy, more robust to smiles/nods)
python main.py --method fft

# Recalibrate without re-running the heavy analysis
python main.py --threshold 0.040
```

Output: `output/lips_output.json`

| Method | How it works | Test accuracy |
|--------|-------------|---------------|
| `std`  | MAR standard deviation — high variance = speaking | **7/7** |
| `fft`  | FFT power in 1.5–6 Hz band — filters smiles/nods | 6/7 |

---

### Step 5 — Final Estimate

Reads all three outputs and computes the final participant counts:

```bash
cd ..   # back to repo root
python estimate.py
```

Or with explicit paths:

```bash
python estimate.py \
    --faces    Face_Detection/output/faces_output.json \
    --speakers Speech_Detector/output/speakers.json \
    --lips     Participants_lips_moving/output/lips_output.json \
    --output   estimate_output.json
```

**Formula:**

| Metric | Formula |
|--------|---------|
| Total participants | Team 3 face count (stable tile tracking is most reliable) |
| Speaking participants | `round((audio_speakers + lip_speakers) / 2)`, clamped to total |

**Sanity checks** (flags a warning but does not override):
- `speaking > total` — impossible; capped at total
- `speaking < 70% of total` — below expected threshold for this recording

**Output `estimate_output.json`:**
```json
{
  "total_participants": 7,
  "speaking_participants": 6,
  "sources": {
    "face_detection_total": 7,
    "audio_unique_speakers": 6,
    "lip_movement_speakers": 6
  },
  "warnings": []
}
```

---

## Output Contracts

### Team 1 → Team 2 (`segments.json`)
```json
[{"start": 0.500, "end": 3.200}, {"start": 4.100, "end": 7.800}]
```
- Sorted, non-overlapping, all segments ≥ 0.5 s

### Team 3 → Team 4 (`faces_output.json`)
```json
{"total_participants_detected": 7, "fps": 29.26, "frames": [...]}
```
- Bounding box values normalized 0–1; `face_id` stable across the entire video

### Team 4 (`lips_output.json`)
```json
{"method": "std", "total_participants": 7, "speaking_count": 6, "per_face": {...}}
```

### Final (`estimate_output.json`)
```json
{"total_participants": 7, "speaking_participants": 6, "sources": {...}, "warnings": []}
```
