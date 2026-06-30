import json
import os
import subprocess


def write_json(output_path, total_participants, fps, frames_out):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "total_participants_detected": int(total_participants),
        "fps": float(round(fps, 2)),
        "frames": frames_out,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"wrote {output_path}")


def validate_json(output_path):
    data = json.load(open(output_path))
    assert set(data) >= {"total_participants_detected", "fps", "frames"}, "missing top-level keys"
    bad = sum(
        1
        for fr in data["frames"]
        for f in fr["faces_detected"]
        for k in ("x_min", "y_min", "width", "height")
        if not (0.0 <= f["bounding_box"][k] <= 1.0)
    )
    assert bad == 0, f"{bad} out-of-range bounding box values"
    print(f"validate OK — participants: {data['total_participants_detected']}, "
          f"frames: {len(data['frames'])}, out-of-range: {bad}")


def reencode_h264(raw_path, out_path):
    """Re-encode raw mp4v to H.264 for broad player compatibility; removes raw file."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", raw_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", out_path],
        check=True,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"annotated video: {out_path} ({size_mb:.1f} MB)")
    os.remove(raw_path)
