# Project Gong — Zoom Speaker Counter

Offline pipeline that takes a Zoom meeting recording and outputs:
1. **Speech segments** — when speech occurs (Part 1)
2. **Speaker mapping** — who said what (Part 2)

---

## Project Structure

```
Project_Gong/
├── Speech_Detector/          ← Part 1: Voice Activity Detection (COMPLETE)
│   ├── main.py               ← entry point
│   ├── speech_detection.py   ← pipeline orchestrator
│   ├── audio_io.py           ← audio loading + ffmpeg conversion
│   ├── postprocessing.py     ← shared VAD post-processing
│   ├── detectors/
│   │   ├── energy.py         ← RMS energy VAD
│   │   └── silero.py         ← Silero neural VAD (primary)
│   ├── writer.py             ← JSON/CSV output + schema validator
│   ├── benchmark.py          ← method comparison (energy vs silero vs WebRTC)
│   ├── requirements.txt
│   └── output/               ← all generated files land here
└── speaker_diarization/      ← Part 2: Speaker Diarization
    ├── speaker_diarization.py
    ├── offline_pipeline.py
    └── requirements.txt
```

---

## Part 1 — Speech Detection

### Setup

```bash
cd Speech_Detector
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

> Also requires `ffmpeg` on PATH:
> ```bash
> brew install ffmpeg        # macOS
> winget install Gyan.FFmpeg # Windows
> ```

### Run

```bash
# Basic run (Silero VAD, default)
venv/bin/python main.py meeting.mp4 segments.json

# With validation
venv/bin/python main.py meeting.mp4 segments.json --validate

# Energy VAD instead of Silero
venv/bin/python main.py meeting.mp4 segments.json --method energy

# Full run: validate + benchmark all methods
venv/bin/python main.py meeting.mp4 segments.json --validate --benchmark
```

### Output

All files written to `output/`:
- `output/meeting.wav` — canonical 16 kHz mono WAV (shared with Part 2)
- `output/segments.json` — speech segments `[{start, end}, ...]`

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `silero` | VAD method: `energy` or `silero` |
| `--min-duration` | `0.5` | Drop segments shorter than N seconds |
| `--min-gap` | `0.4` | Merge segments with silence gap shorter than N seconds |
| `--validate` | off | Validate output against downstream contract |
| `--benchmark` | off | Compare energy, silero, and WebRTC methods |
| `--output-dir` | `output/` | Directory for all output files |

---

## Part 2 — Speaker Diarization

Reads `output/segments.json` + `output/meeting.wav` from Part 1.

```bash
cd Speech_Detector
venv/bin/pip install resemblyzer

venv/bin/python ../speaker_diarization/speaker_diarization.py \
    output/meeting.wav \
    output/segments.json \
    output/speakers.json \
    --threshold 0.55
```

Output: `output/speakers.json` — `[{start, end, speaker_id}, ...]`

> **Note:** The `--threshold` flag in `speaker_diarization.py` is inverted from its docstring.
> Lower value = fewer speakers. Sweet spot for Zoom meetings: `0.50–0.60`.

---

## Benchmark Results (Gong_Test_VIDEO.mp4)

| Method | Segments | Total Speech | IoU vs WebRTC |
|--------|----------|-------------|---------------|
| energy | 90 | 620.10 s | 0.944 |
| silero | 153 | 530.18 s | 0.880 |
| webrtc | 106 | 601.80 s | (reference) |

**Chosen method:** Silero — finer segmentation gives cleaner speaker embeddings for diarization.

---

## Output Contract (Part 1 → Part 2)

```json
[
  {"start": 0.500, "end": 3.200},
  {"start": 4.100, "end": 7.800}
]
```

- Field names: `start` / `end` (floats, seconds)
- Sorted ascending, non-overlapping
- All segments ≥ 0.5 s duration
- All times within audio bounds
