import argparse
import os

from lip_analysis import run_pipeline

HERE           = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO  = os.path.join(HERE, "..", "Gong_Test_VIDEO.mp4")
DEFAULT_JSON   = os.path.join(HERE, "..", "Face_Detection", "output", "faces_output.json")
DEFAULT_MODEL  = os.path.join(HERE, "face_landmarker.task")
DEFAULT_OUTPUT = os.path.join(HERE, "output", "lips_output.json")


def parse_args():
    p = argparse.ArgumentParser(description="Team 4 — Lip movement speaker detection")
    p.add_argument("input",      nargs="?", default=DEFAULT_VIDEO, help="Input video (default: ../Gong_Test_VIDEO.mp4)")
    p.add_argument("faces_json", nargs="?", default=DEFAULT_JSON,  help="faces_output.json from Team 3 (default: ../Face_Detection/output/faces_output.json)")
    p.add_argument("--model",      default=DEFAULT_MODEL,  help="Path to face_landmarker.task model file")
    p.add_argument("--method",     choices=["std", "fft"], default="std", help="Detection method: std (7/7, default) or fft (6/7)")
    p.add_argument("--threshold",  type=float, default=None, help="Override threshold (std default: 0.035 | fft default: 300.0)")
    p.add_argument("--frame-skip", type=int,   default=3,    help="Process every Nth frame (default: 3 — good balance)")
    p.add_argument("--output",     default=DEFAULT_OUTPUT, help="Save lips_output.json here (default: output/lips_output.json)")
    return p.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        video_path=args.input,
        faces_json=args.faces_json,
        model_path=args.model,
        method=args.method,
        threshold=args.threshold,
        frame_skip=args.frame_skip,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
