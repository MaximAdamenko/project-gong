import os

from detector import cluster_slots, detect_faces, load_model, probe_video
from renderer import render
from writer import reencode_h264, validate_json, write_json


def run_pipeline(
    video_path,
    output_json,
    annotated_path,
    conf=0.30,
    imgsz=1280,
    max_frames=None,
    slot_radius=0.16,
    min_presence=0.08,
    validate=False,
):
    print(f"[1/4] probing {video_path}")
    fps, W, H, total = probe_video(video_path)
    print(f"      {W}x{H} @ {fps:.2f}fps, ~{total} frames")

    print("[2/4] loading YOLOv8-Face model")
    model = load_model()

    print("[3/4] detecting faces (Pass A)")
    per_frame = detect_faces(
        model, video_path, W, H, total,
        conf=conf, imgsz=imgsz, max_frames=max_frames,
    )
    real_slots, _ = cluster_slots(per_frame, slot_radius=slot_radius, min_presence=min_presence)

    print("[4/4] rendering annotated video (Pass B)")
    raw_annotated = annotated_path.replace(".mp4", "_raw.mp4")
    frames_out = render(video_path, per_frame, real_slots, fps, W, H, raw_annotated, slot_radius=slot_radius)

    write_json(output_json, len(real_slots), fps, frames_out)
    reencode_h264(raw_annotated, annotated_path)

    if validate:
        validate_json(output_json)

    return real_slots
