import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe Face Landmarker lip indices
UPPER, LOWER, LEFT, RIGHT = 13, 14, 61, 291


def mouth_aspect_ratio(landmarks, w, h):
    """Vertical lip opening divided by horizontal mouth width (scale-invariant)."""
    pt = lambda i: np.array([landmarks[i].x * w, landmarks[i].y * h])
    return np.linalg.norm(pt(UPPER) - pt(LOWER)) / (np.linalg.norm(pt(LEFT) - pt(RIGHT)) + 1e-6)


def make_landmarker(model_path):
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def measure_mar(video_path, boxes_by_frame, model_path, frame_skip=3, pad=0.30):
    """Heavy analysis pass: returns {face_id: [MAR time series]} for every participant.

    Runs once — re-run report() / detectors with different thresholds without re-analyzing.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lm = make_landmarker(model_path)
    series, idx = {}, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_skip == 0 and idx in boxes_by_frame:
            if idx % 3000 == 0:
                print(f"  processing... {idx}/{total}")
            H, W = frame.shape[:2]
            for fid, xn, yn, wn, hn in boxes_by_frame[idx]:
                x, y, w, h = xn * W, yn * H, wn * W, hn * H
                px, py = w * pad, h * pad
                crop = frame[max(0, int(y - py)):min(H, int(y + h + py)),
                             max(0, int(x - px)):min(W, int(x + w + px))]
                if crop.size == 0:
                    continue
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                if res.face_landmarks:
                    ch, cw = crop.shape[:2]
                    series.setdefault(fid, []).append(
                        mouth_aspect_ratio(res.face_landmarks[0], cw, ch))
        idx += 1

    cap.release()
    return series
