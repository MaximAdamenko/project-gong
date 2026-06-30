import math
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def load_model():
    model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        filename="model.pt",
    )
    return YOLO(model_path)


def probe_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, W, H, total


def detect_faces(model, video_path, W, H, total, conf=0.30, imgsz=1280,
                 max_frames=None, tracker="bytetrack.yaml"):
    """Pass A: YOLO tracking — returns per_frame list of (cx, cy, nx1, ny1, nx2, ny2)."""
    per_frame = []
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )
    for fi, res in enumerate(results):
        if max_frames is not None and fi >= max_frames:
            break
        dets = []
        if res.boxes is not None and len(res.boxes):
            for (x1, y1, x2, y2) in res.boxes.xyxy.cpu().numpy():
                nx1, ny1, nx2, ny2 = x1 / W, y1 / H, x2 / W, y2 / H
                cx = (nx1 + nx2) / 2
                cy = (ny1 + ny2) / 2
                dets.append((cx, cy, nx1, ny1, nx2, ny2))
        per_frame.append(dets)
        if fi % 500 == 0:
            print(f"  detect frame {fi}/{total}")
    return per_frame


def cluster_slots(per_frame, slot_radius=0.16, min_presence=0.08):
    """Group detections into fixed tile slots; filter transient/background faces.

    slot_radius  — raise if one person splits into two ids; lower if two tiles merge.
    min_presence — fraction of frames a slot must appear in to count as a real participant.
    """
    slots = []
    n_frames = len(per_frame)

    for dets in per_frame:
        for (cx, cy, *_) in dets:
            best, bestd = None, 1e9
            for s in slots:
                d = math.hypot(cx - s["cx"], cy - s["cy"])
                if d < bestd:
                    best, bestd = s, d
            if best is not None and bestd <= slot_radius:
                n = best["count"]
                best["cx"] = (best["cx"] * n + cx) / (n + 1)
                best["cy"] = (best["cy"] * n + cy) / (n + 1)
                best["count"] += 1
            else:
                slots.append({"cx": cx, "cy": cy, "count": 1})

    real = [s for s in slots if s["count"] >= min_presence * n_frames]
    real.sort(key=lambda s: (round(s["cy"], 1), s["cx"]))
    for i, s in enumerate(real):
        s["label"] = f"face_{i + 1:02d}"

    print(f"\nraw slots: {len(slots)}  ->  real participants: {len(real)}")
    for s in real:
        pct = 100 * s["count"] / n_frames
        print(f'  {s["label"]}: center=({s["cx"]:.2f},{s["cy"]:.2f}) present {pct:.0f}% of frames')

    return real, n_frames
