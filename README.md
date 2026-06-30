# Project Gong — Zoom Meeting Analyzer

Offline pipeline that takes a Zoom meeting recording and outputs speech segments, speaker identity, face locations, and lip-movement detection.

---

## Project Structure

```
Project_Gong/
├── Gong_Test_VIDEO.mp4           ← shared input video (all modules read from here)
│
├── Speech_Detector/              ← Team 1 · Voice Activity Detection (COMPLETE)
│   ├── main.py
│   ├── speech_detection.py       ← pipeline orchestrator
│   ├── audio_io.py
│   ├── postprocessing.py
│   ├── detectors/
│   │   ├── energy.py             ← RMS energy VAD
│   │   └── silero.py             ← Silero neural VAD (primary)
│   ├── writer.py
│   ├── benchmark.py
│   ├── requirements.txt
│   └── output/
│       ├── meeting.wav           ← 16 kHz mono WAV
│       └── segments.json         ← speech segments [{start, end}]
│
├── speaker_diarization/          ← Team 2 · Speaker Diarization
│   ├── speaker_diarization.py
│   ├── offline_pipeline.py
│   └── requirements.txt
│
├── Face_Detection/               ← Team 3 · Face Detection + Stable Tile Tracking (COMPLETE)
│   ├── main.py
│   ├── face_detection.py         ← pipeline orchestrator
│   ├── detector.py               ← Pass A: YOLOv8-Face tracking + slot clustering
│   ├── renderer.py               ← Pass B: annotated video rendering
│   ├── writer.py
│   ├── requirements.txt
│   └── output/
│       ├── faces_output.json     ← per-frame bounding boxes + stable face_id
│       └── annotated_output.mp4  ← video with boxes drawn on
│
└── Participants_lips_moving/     ← Team 4 · Lip Movement Speaker Detection (COMPLETE)
    ├── main.py
    ├── lip_analysis.py           ← pipeline orchestrator
    ├── loader.py                 ← reads faces_output.json from Team 3
    ├── landmarker.py             ← MediaPipe MAR measurement (shared heavy pass)
    ├── detectors/
    │   ├── std_detector.py       ← MAR std deviation (7/7 result)
    │   └── fft_detector.py       ← FFT speech-band power (6/7 result)
    ├── reporter.py
    └── requirements.txt
```

---

## Data Flow

```
Gong_Test_VIDEO.mp4
        │
        ├──► Speech_Detector      → output/segments.json
        │          │
        │          └──► speaker_diarization → output/speakers.json
        │
        └──► Face_Detection       → output/faces_output.json
                   │                         output/annotated_output.mp4
                   └──► Participants_lips_moving → who is speaking (by lips)
```

---

## Team 1 — Speech Detection

### Setup

```bash
cd Speech_Detector
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

> Requires `ffmpeg` on PATH:
> ```bash
> brew install ffmpeg         # macOS
> winget install Gyan.FFmpeg  # Windows
> ```

### Run

```bash
# Default run (Silero VAD)
venv/bin/python main.py

# With all options
venv/bin/python main.py ../Gong_Test_VIDEO.mp4 output/segments.json \
    --method silero \
    --validate \
    --benchmark
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `silero` | VAD method: `energy` or `silero` |
| `--min-duration` | `0.5` | Drop segments shorter than N seconds |
| `--min-gap` | `0.4` | Merge silence gaps shorter than N seconds |
| `--validate` | off | Validate output against downstream contract |
| `--benchmark` | off | Compare energy, silero, and WebRTC methods |

### Benchmark Results (Gong_Test_VIDEO.mp4)

| Method | Segments | Total Speech | IoU vs WebRTC |
|--------|----------|--------------|---------------|
| energy | 90 | 620.10 s | 0.944 |
| silero | 153 | 530.18 s | 0.880 |
| webrtc | 106 | 601.80 s | (reference) |

---

## Team 2 — Speaker Diarization

Reads `segments.json` + `meeting.wav` from Team 1 output.

```bash
cd Speech_Detector
venv/bin/pip install resemblyzer

venv/bin/python ../speaker_diarization/speaker_diarization.py \
    output/meeting.wav \
    output/segments.json \
    output/speakers.json \
    --threshold 0.55
```

> **Note:** The `--threshold` flag is inverted from its docstring — lower value = fewer speakers. Sweet spot for Zoom meetings: `0.50–0.60`.

---

## Team 3 — Face Detection

Detects faces with YOLOv8-Face and locks each participant to a stable `face_id` based on their Zoom tile position.

### Setup

```bash
cd Face_Detection
pip install -r requirements.txt
# ffmpeg also required (see Team 1 setup)
```

### Run

```bash
# Default — reads ../Gong_Test_VIDEO.mp4, writes to output/
python main.py

# Quick test (~30 s)
python main.py --max-frames 900

# Full options
python main.py ../Gong_Test_VIDEO.mp4 output/faces_output.json \
    --annotated output/annotated_output.mp4 \
    --slot-radius 0.16 \
    --min-presence 0.08 \
    --validate
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--conf` | `0.30` | YOLO detection confidence threshold |
| `--imgsz` | `1280` | YOLO inference image size |
| `--max-frames` | none | Cap frames processed (for quick tests) |
| `--slot-radius` | `0.16` | Tile clustering radius — raise if one person splits into two ids |
| `--min-presence` | `0.08` | Min fraction of frames a face must appear in to be a real participant |
| `--validate` | off | Validate output JSON schema after writing |

### Output contract (→ Team 4)

```json
{
  "total_participants_detected": 7,
  "fps": 29.26,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "faces_detected": [
        {
          "face_id": "face_01",
          "bounding_box": { "x_min": 0.78, "y_min": 0.10, "width": 0.12, "height": 0.10 }
        }
      ]
    }
  ]
}
```

- Bounding box values normalized 0–1
- `face_id` is stable for the entire video duration
- **Test result:** 7 participants detected across 22,008 frames

---

## Team 4 — Lip Movement Detection

Reads `faces_output.json` from Team 3 and determines who is speaking by measuring mouth aspect ratio (MAR) over time.

### Setup

```bash
cd Participants_lips_moving
pip install -r requirements.txt

# Download MediaPipe model
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Run

```bash
# Default — reads ../Gong_Test_VIDEO.mp4 + ../Face_Detection/output/faces_output.json
python main.py

# FFT method
python main.py --method fft

# Recalibrate threshold without re-running the heavy analysis
python main.py --threshold 0.040

# Full options
python main.py ../Gong_Test_VIDEO.mp4 ../Face_Detection/output/faces_output.json \
    --method std \
    --threshold 0.035 \
    --frame-skip 3
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `std` | Detection method: `std` (7/7) or `fft` (6/7) |
| `--threshold` | `0.035` / `300.0` | Override detection threshold (std / fft) |
| `--frame-skip` | `3` | Process every Nth frame — 2=accurate/slow, 4=fast |

### Detection methods

| Method | How it works | Test result |
|--------|-------------|-------------|
| `std` | MAR standard deviation — high variance = speaking | **7/7** |
| `fft` | FFT power in 1.5–6 Hz band — filters smiles/nods | 6/7 |

---

## Output Contracts

### Team 1 → Team 2 (`segments.json`)
```json
[{"start": 0.500, "end": 3.200}, {"start": 4.100, "end": 7.800}]
```

### Team 3 → Team 4 (`faces_output.json`)
```json
{"total_participants_detected": 7, "fps": 29.26, "frames": [...]}
```
