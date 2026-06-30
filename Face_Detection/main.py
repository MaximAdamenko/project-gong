import argparse
import os

from face_detection import run_pipeline

DEFAULT_VIDEO = os.path.join(os.path.dirname(__file__), "..", "Gong_Test_VIDEO.mp4")
DEFAULT_JSON  = os.path.join(os.path.dirname(__file__), "output", "faces_output.json")


def parse_args():
    p = argparse.ArgumentParser(
        description="Team 3 — Face detection + stable tile tracking → JSON for Team 4"
    )
    p.add_argument("input",       nargs="?", default=DEFAULT_VIDEO, help="Input video (default: ../Gong_Test_VIDEO.mp4)")
    p.add_argument("output_json", nargs="?", default=DEFAULT_JSON,  help="Output JSON path (default: output/faces_output.json)")
    p.add_argument("--annotated",    default=None,  help="Annotated video path (default: output/annotated_output.mp4)")
    p.add_argument("--conf",         type=float, default=0.30,  help="YOLO confidence threshold (default: 0.30)")
    p.add_argument("--imgsz",        type=int,   default=1280,  help="YOLO inference image size  (default: 1280)")
    p.add_argument("--max-frames",   type=int,   default=None,  help="Cap frames processed — useful for quick tests")
    p.add_argument("--slot-radius",  type=float, default=0.16,  help="Tile slot clustering radius 0-1 (default: 0.16)")
    p.add_argument("--min-presence", type=float, default=0.08,  help="Min frame-presence fraction  (default: 0.08)")
    p.add_argument("--validate",     action="store_true",        help="Validate output JSON schema after writing")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    annotated = args.annotated or os.path.join(output_dir, "annotated_output.mp4")

    slots = run_pipeline(
        video_path=args.input,
        output_json=args.output_json,
        annotated_path=annotated,
        conf=args.conf,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
        slot_radius=args.slot_radius,
        min_presence=args.min_presence,
        validate=args.validate,
    )
    print(f"\ndone — {len(slots)} participants")


if __name__ == "__main__":
    main()
