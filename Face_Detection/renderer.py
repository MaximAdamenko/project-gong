import math
import cv2


def _assign(cx, cy, real_slots, slot_radius):
    best, bestd = None, 1e9
    for s in real_slots:
        d = math.hypot(cx - s["cx"], cy - s["cy"])
        if d < bestd:
            best, bestd = s, d
    if best is not None and bestd <= slot_radius:
        return best["label"]
    return None


def render(video_path, per_frame, real_slots, fps, W, H, raw_output_path, slot_radius=0.16):
    """Pass B: draw bounding boxes + face_id labels; return per-frame JSON data.

    Writes a raw mp4v file to raw_output_path — caller must re-encode to H.264.
    """
    cap = cv2.VideoCapture(video_path)
    video_writer = cv2.VideoWriter(
        raw_output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    frames_out = []
    n_frames = len(per_frame)

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        chosen = {}
        for (cx, cy, nx1, ny1, nx2, ny2) in per_frame[fi]:
            lab = _assign(cx, cy, real_slots, slot_radius)
            if lab is None:
                continue
            s = next(s for s in real_slots if s["label"] == lab)
            d = math.hypot(cx - s["cx"], cy - s["cy"])
            if lab not in chosen or d < chosen[lab][1]:
                chosen[lab] = ((nx1, ny1, nx2, ny2), d)

        faces = []
        for lab, ((nx1, ny1, nx2, ny2), _) in sorted(chosen.items()):
            faces.append({
                "face_id": lab,
                "bounding_box": {
                    "x_min": round(float(nx1), 4),
                    "y_min": round(float(ny1), 4),
                    "width":  round(float(nx2 - nx1), 4),
                    "height": round(float(ny2 - ny1), 4),
                },
            })
            x1, y1 = int(nx1 * W), int(ny1 * H)
            x2, y2 = int(nx2 * W), int(ny2 * H)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, lab, (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        video_writer.write(frame)
        frames_out.append({
            "frame_index": fi,
            "timestamp_ms": int(round(fi / fps * 1000)),
            "faces_detected": faces,
        })

        if fi % 500 == 0:
            print(f"  render frame {fi}/{n_frames}")

    cap.release()
    video_writer.release()
    return frames_out
