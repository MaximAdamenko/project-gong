"""
run_all.py — Project Gong: one command, live analysis UI.

Usage:
    python run_all.py                        # uses Gong_Test_VIDEO.mp4
    python run_all.py meeting.mp4
    python run_all.py meeting.mp4 --no-audio
    python run_all.py meeting.mp4 --yolo-every 3   # more frequent detection (slower)

What it does:
    1. Scans the first few hundred frames to discover stable face tile positions
    2. Plays the video live with:
       - Face bounding boxes (green = speaking, gray = silent)
       - Real-time lip-movement speaking detection (MediaPipe rolling window)
       - Side panel: participant list, live counts, audio analysis progress
    3. Runs Teams 1+2 (speech detection + diarization) in a background thread
    4. Computes and displays the final estimate when audio is done

Press Q or ESC to quit.
"""

import sys
import os
import argparse
import threading
import time
import json
import math
import urllib.request
from pathlib import Path
from collections import deque

import cv2
import numpy as np

HERE = Path(__file__).parent
for _p in ["Face_Detection", "Participants_lips_moving", "Speech_Detector", "speaker_diarization"]:
    sys.path.insert(0, str(HERE / _p))

# ── Tuning constants ────────────────────────────────────────────────────────
PANEL_W      = 295        # sidebar width in pixels
YOLO_EVERY   = 5          # run YOLO every N frames (higher = faster but less smooth)
MAR_WINDOW   = 60         # rolling frames for lip-movement detection
STD_THR      = 0.035      # MAR std threshold → speaking
SLOT_RADIUS  = 0.18       # fraction of frame width/height for face slot clustering
MIN_PRESENCE = 0.04       # fraction of scan frames a face must appear in to count
SCAN_EVERY   = 5          # frame skip during the initial participant scan
SCAN_FRAMES  = 400        # how many frames to scan (more = more accurate, slower start)

MP_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MP_UPPER, MP_LOWER, MP_LEFT, MP_RIGHT = 13, 14, 61, 291

# ── Palette ─────────────────────────────────────────────────────────────────
BG       = (22,  20,  18)
PANEL_BG = (30,  27,  25)
DIVIDER  = (52,  48,  46)
WHITE    = (240, 238, 235)
GRAY     = (108, 105, 102)
GREEN    = (68,  200,  88)
YELLOW   = (48,  196, 218)
RED_C    = (72,   72, 210)
BLUE_BAR = (180, 120,  55)


# ── Shared state (thread-safe) ───────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock       = threading.Lock()
        self.slots      = []      # [{label, cx, cy}]  — face tile positions
        self.boxes      = {}      # label -> (nx1, ny1, nx2, ny2)
        self.mar_win    = {}      # label -> deque of MAR values
        self.speaking   = {}      # label -> bool
        self.audio_st   = "pending"   # pending | running | done | skipped | error:...
        self.audio_pct  = 0.0
        self.audio_spk  = None        # int when done
        self.estimate   = None        # dict when ready


# ── Background audio thread (Teams 1 + 2) ───────────────────────────────────
def run_audio(video_path: Path, out_dir: Path, state: State):
    state.audio_st = "running"
    try:
        wav = out_dir / "meeting.wav"
        seg = out_dir / "segments.json"
        spk = out_dir / "speakers.json"

        from audio_io import prepare_audio
        audio, sr = prepare_audio(str(video_path), output_path=str(wav))
        state.audio_pct = 0.30

        from detectors.silero import detect_silero
        from writer import write_segments
        segments = detect_silero(audio, sr)
        write_segments(segments, str(seg))
        state.audio_pct = 0.65

        from speaker_diarization import diarize
        result = diarize(str(wav), str(seg), str(spk), threshold=0.55)
        state.audio_pct = 1.0

        n = len(set(s["speaker_id"] for s in result)) if result else 0
        with state.lock:
            state.audio_spk = n
            state.audio_st  = "done"

    except Exception as e:
        with state.lock:
            state.audio_st = f"error: {e}"


# ── MediaPipe helpers ────────────────────────────────────────────────────────
def load_landmarker(model_path: Path):
    if not model_path.exists():
        print(f"  Downloading face_landmarker.task (~30 MB)...")
        urllib.request.urlretrieve(MP_URL, str(model_path))
        print(f"  Saved to {model_path}")
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def mouth_aspect_ratio(landmarks, w, h):
    pt = lambda i: np.array([landmarks[i].x * w, landmarks[i].y * h])
    vert  = np.linalg.norm(pt(MP_UPPER) - pt(MP_LOWER))
    horiz = np.linalg.norm(pt(MP_LEFT)  - pt(MP_RIGHT))
    return float(vert / (horiz + 1e-6))


# ── Face slot helpers ────────────────────────────────────────────────────────
def nearest_slot(cx, cy, slots):
    best, bd = None, 1e9
    for s in slots:
        d = math.hypot(cx - s["cx"], cy - s["cy"])
        if d < bd:
            best, bd = s, d
    return best["label"] if (best and bd <= SLOT_RADIUS) else None


def scan_participants(model, video_path: Path, W: int, H: int) -> list:
    """Quick scan of first SCAN_FRAMES frames to discover stable face tile positions."""
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw   = []  # list of (cx, cy) normalized detections
    seen  = 0
    fi    = 0

    print(f"  Scanning first {min(SCAN_FRAMES, total)} frames "
          f"(every {SCAN_EVERY}th) to discover participants...", flush=True)

    while seen < SCAN_FRAMES and fi < total:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % SCAN_EVERY == 0:
            res = model.predict(frame, conf=0.25, imgsz=640, verbose=False)
            if res and res[0].boxes is not None:
                for (x1, y1, x2, y2) in res[0].boxes.xyxy.cpu().numpy():
                    raw.append(((x1 + x2) / (2 * W), (y1 + y2) / (2 * H)))
            seen += 1
        fi += 1

    cap.release()

    # Cluster detections into stable tile slots
    slots = []
    for cx, cy in raw:
        best, bd = None, 1e9
        for s in slots:
            d = math.hypot(cx - s["cx"], cy - s["cy"])
            if d < bd:
                best, bd = s, d
        if best and bd <= SLOT_RADIUS:
            n = best["count"]
            best["cx"] = (best["cx"] * n + cx) / (n + 1)
            best["cy"] = (best["cy"] * n + cy) / (n + 1)
            best["count"] += 1
        else:
            slots.append({"cx": cx, "cy": cy, "count": 1})

    # Filter transient detections and sort top-to-bottom, left-to-right
    threshold = MIN_PRESENCE * seen
    real = [s for s in slots if s["count"] >= threshold]
    real.sort(key=lambda s: (round(s["cy"] * 5) / 5, s["cx"]))
    for i, s in enumerate(real):
        s["label"] = f"face_{i + 1:02d}"

    print(f"  Found {len(real)} participant(s) in {seen} frames scanned.", flush=True)
    return real


# ── UI drawing ───────────────────────────────────────────────────────────────
def draw_side_panel(panel, state: State, fps: float, fi: int, total: int):
    panel[:] = PANEL_BG
    W = panel.shape[1]
    y = 20

    def txt(s, row, col=WHITE, sc=0.52, bold=False):
        cv2.putText(panel, s, (12, row), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, col, 2 if bold else 1, cv2.LINE_AA)

    def div(row):
        cv2.line(panel, (8, row), (W - 8, row), DIVIDER, 1)

    # Title
    txt("PROJECT GONG", y, YELLOW, 0.62, bold=True)
    y += 8; div(y + 6); y += 20

    # Participants
    txt("PARTICIPANTS", y, GRAY, 0.42); y += 18
    n_speaking = 0
    for s in state.slots:
        lbl = s["label"]
        spk = state.speaking.get(lbl, False)
        if spk:
            n_speaking += 1
        col  = GREEN if spk else GRAY
        mark = "SPEAKING" if spk else "silent"
        cv2.circle(panel, (18, y - 4), 5, col if spk else DIVIDER, -1)
        txt(f"{lbl}  {mark}", y, col, 0.46)
        y += 21
    if not state.slots:
        txt("scanning...", y, GRAY, 0.44); y += 20

    y += 6; div(y); y += 14

    # Live counts
    txt("LIVE COUNTS", y, GRAY, 0.42); y += 18
    txt(f"Total:    {len(state.slots)}", y, WHITE, 0.54, bold=True); y += 23
    c = GREEN if n_speaking else GRAY
    txt(f"Speaking: {n_speaking}", y, c, 0.54, bold=True); y += 23

    y += 4; div(y); y += 14

    # Audio analysis
    txt("AUDIO ANALYSIS", y, GRAY, 0.42); y += 18
    st = state.audio_st
    if st == "pending":
        txt("waiting to start...", y, GRAY, 0.44)
    elif st == "running":
        bw = W - 24
        cv2.rectangle(panel, (12, y - 10), (12 + bw, y + 4), DIVIDER, -1)
        filled = int(bw * state.audio_pct)
        cv2.rectangle(panel, (12, y - 10), (12 + filled, y + 4), GREEN, -1)
        txt(f"{int(state.audio_pct * 100)}%  analyzing...", y + 18, GREEN, 0.44)
        y += 14
    elif st == "done":
        txt(f"Speakers detected: {state.audio_spk}", y, GREEN, 0.52, bold=True)
    elif st == "skipped":
        txt("--no-audio flag set", y, GRAY, 0.42)
    else:
        txt(st[:32], y, RED_C, 0.38)
    y += 28

    # Final estimate
    if state.estimate:
        div(y); y += 14
        txt("FINAL ESTIMATE", y, YELLOW, 0.44); y += 18
        e = state.estimate
        txt(f"Total:    {e['total_participants']}", y, WHITE, 0.54, bold=True); y += 23
        pct = (e["speaking_participants"] / max(e["total_participants"], 1))
        txt(f"Speaking: {e['speaking_participants']} ({pct:.0%})", y, GREEN, 0.54, bold=True)
        y += 23
        for w in e.get("warnings", []):
            txt(f"! {w[:34]}", y, RED_C, 0.37); y += 16

    # Progress bar at bottom
    bar_y = panel.shape[0] - 20
    bw2   = W - 24
    cv2.rectangle(panel, (12, bar_y), (12 + bw2, bar_y + 8), DIVIDER, -1)
    pct2 = fi / max(total, 1)
    cv2.rectangle(panel, (12, bar_y), (12 + int(bw2 * pct2), bar_y + 8), BLUE_BAR, -1)
    txt(f"FPS {fps:.1f}   {fi} / {total}", bar_y - 6, GRAY, 0.37)


def draw_face_overlays(frame, state: State, disp_W: int, disp_H: int):
    """Draw bounding boxes and speaking status onto the display frame."""
    for s in state.slots:
        lbl = s["label"]
        box = state.boxes.get(lbl)
        if box is None:
            continue
        nx1, ny1, nx2, ny2 = box
        x1, y1 = int(nx1 * disp_W), int(ny1 * disp_H)
        x2, y2 = int(nx2 * disp_W), int(ny2 * disp_H)
        spk   = state.speaking.get(lbl, False)
        color = (0, 210, 80) if spk else (95, 93, 90)
        thick = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        tag = f"{lbl}  {'● SPEAKING' if spk else '○ silent'}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        tag_y = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, tag_y), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 4, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Project Gong — live Zoom meeting analyzer")
    ap.add_argument("input", nargs="?", default=str(HERE / "Gong_Test_VIDEO.mp4"),
                    help="Input video (default: Gong_Test_VIDEO.mp4)")
    ap.add_argument("--no-audio",   action="store_true",
                    help="Skip audio analysis (Teams 1+2) for faster startup")
    ap.add_argument("--yolo-every", type=int, default=YOLO_EVERY,
                    help=f"Run YOLO every N frames (default: {YOLO_EVERY})")
    ap.add_argument("--output-dir", default=str(HERE / "output"),
                    help="Directory for all generated output files (default: output/)")
    args = ap.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        sys.exit(f"ERROR: video not found: {video_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = State()

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading YOLOv8-Face model...")
    from detector import load_model, probe_video
    yolo = load_model()

    mp_model_path = HERE / "Participants_lips_moving" / "face_landmarker.task"
    print("[2/3] Loading MediaPipe face landmarker...")
    import mediapipe as mp
    lm = load_landmarker(mp_model_path)

    fps_v, W, H, total_frames = probe_video(str(video_path))
    print(f"      Video: {W}x{H} @ {fps_v:.1f}fps, {total_frames} frames")

    # ── Scan participants ─────────────────────────────────────────────────────
    print("[3/3] Scanning participants...")
    slots = scan_participants(yolo, video_path, W, H)
    with state.lock:
        state.slots   = slots
        state.mar_win = {s["label"]: deque(maxlen=MAR_WINDOW) for s in slots}
        state.speaking = {s["label"]: False for s in slots}

    # ── Start audio background thread ─────────────────────────────────────────
    if not args.no_audio:
        audio_dir = HERE / "Speech_Detector" / "output"
        audio_dir.mkdir(parents=True, exist_ok=True)
        threading.Thread(
            target=run_audio, args=(video_path, audio_dir, state), daemon=True
        ).start()
    else:
        state.audio_st = "skipped"

    # ── Live display loop ─────────────────────────────────────────────────────
    cap    = cv2.VideoCapture(str(video_path))
    DISP_W = min(W, 960)
    DISP_H = int(H * DISP_W / W)

    cached_boxes = {}  # label -> (nx1,ny1,nx2,ny2) from last YOLO run
    t0    = time.time()
    fi    = 0
    fps_d = 0.0

    print("\nLive display started — press Q or ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fi += 1
        fps_d = fi / max(time.time() - t0, 1e-3)

        # ── YOLO face detection (every N frames) ─────────────────────────────
        if fi % args.yolo_every == 0:
            res = yolo.predict(frame, conf=0.28, imgsz=640, verbose=False)
            new_boxes = {}
            if res and res[0].boxes is not None:
                for (x1, y1, x2, y2) in res[0].boxes.xyxy.cpu().numpy():
                    cx = (x1 + x2) / (2 * W)
                    cy = (y1 + y2) / (2 * H)
                    lbl = nearest_slot(cx, cy, slots)
                    if lbl:
                        new_boxes[lbl] = (x1 / W, y1 / H, x2 / W, y2 / H)
            cached_boxes = new_boxes
            with state.lock:
                state.boxes = dict(cached_boxes)

        # ── MediaPipe lip detection (every frame on cached boxes) ─────────────
        for s in slots:
            lbl = s["label"]
            box = cached_boxes.get(lbl)
            if box is None:
                continue
            nx1, ny1, nx2, ny2 = box
            x_px  = nx1 * W
            y_px  = ny1 * H
            wb    = (nx2 - nx1) * W
            hb    = (ny2 - ny1) * H
            pad   = 0.30
            crop  = frame[
                max(0, int(y_px - hb * pad)) : min(H, int(y_px + hb * (1 + pad))),
                max(0, int(x_px - wb * pad)) : min(W, int(x_px + wb * (1 + pad))),
            ]
            if crop.size == 0:
                continue
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                r   = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if r.face_landmarks:
                    ch, cw = crop.shape[:2]
                    state.mar_win[lbl].append(
                        mouth_aspect_ratio(r.face_landmarks[0], cw, ch)
                    )
            except Exception:
                pass

            # Update speaking status from rolling window
            win = state.mar_win[lbl]
            if len(win) >= 10:
                with state.lock:
                    state.speaking[lbl] = float(np.std(win)) > STD_THR

        # ── Compute final estimate when audio finishes ────────────────────────
        if state.audio_st == "done" and state.estimate is None:
            lip_count = sum(1 for s in slots if state.speaking.get(s["label"], False))
            total_p   = len(slots)
            speaking  = min(round((state.audio_spk + lip_count) / 2), total_p)
            warnings  = []
            if total_p > 0 and speaking / total_p < 0.70:
                warnings.append(f"{speaking}/{total_p} speaking is below 70% threshold")
            if speaking > total_p:
                warnings.append(f"speaking ({speaking}) > total ({total_p}) — capped")
                speaking = total_p
            estimate = {
                "total_participants":   total_p,
                "speaking_participants": speaking,
                "sources": {
                    "face_detection_total": total_p,
                    "audio_unique_speakers": state.audio_spk,
                    "lip_movement_speakers": lip_count,
                },
                "warnings": warnings,
            }
            state.estimate = estimate
            with open(out_dir / "estimate_output.json", "w") as f:
                json.dump(estimate, f, indent=2)
            print(f"\n[ESTIMATE] Total: {total_p}  Speaking: {speaking}")
            if warnings:
                for w in warnings:
                    print(f"  WARNING: {w}")

        # ── Draw frame ────────────────────────────────────────────────────────
        disp  = cv2.resize(frame, (DISP_W, DISP_H))
        draw_face_overlays(disp, state, DISP_W, DISP_H)

        panel = np.zeros((DISP_H, PANEL_W, 3), dtype=np.uint8)
        draw_side_panel(panel, state, fps_d, fi, total_frames)

        window = np.hstack([disp, panel])
        cv2.imshow("Project Gong — Live Analyzer", window)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  FINAL RESULT")
    print("=" * 52)
    if state.estimate:
        e = state.estimate
        print(f"  Total participants:    {e['total_participants']}")
        print(f"  Speaking participants: {e['speaking_participants']}"
              f"  ({e['speaking_participants']/max(e['total_participants'],1):.0%})")
        print(f"  Sources:")
        src = e["sources"]
        print(f"    Face detection:   {src['face_detection_total']}")
        print(f"    Audio diarization:{src['audio_unique_speakers']}")
        print(f"    Lip movement:     {src['lip_movement_speakers']}")
        if e["warnings"]:
            for w in e["warnings"]:
                print(f"  WARNING: {w}")
        print(f"\n  Saved -> {out_dir / 'estimate_output.json'}")
    else:
        lip = sum(1 for s in slots if state.speaking.get(s["label"], False))
        print(f"  Total participants (video): {len(slots)}")
        print(f"  Speaking (lip movement):    {lip}")
        print(f"  Audio analysis: {state.audio_st}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
