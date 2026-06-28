"""
main.py — Entry point for the Speech Detection pipeline.

Manages the full detection event flow:
  1. LOAD     — prepare canonical audio
  2. DETECT   — run chosen VAD method
  3. WRITE    — serialize segments to file
  4. VALIDATE — verify output contract (optional)
  5. BENCHMARK— compare all methods (optional)

Run:
    python main.py INPUT OUTPUT [--method energy|silero]
                                [--min-duration 0.5]
                                [--min-gap 0.4]
                                [--validate]
                                [--benchmark]
"""

import argparse
import sys

from speech_detection import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="speech_detection",
        description="Voice Activity Detection — produces speech segmentation file.",
    )
    parser.add_argument("INPUT",  help="Input audio/video file (.mp4, .mp3, .wav, …)")
    parser.add_argument("OUTPUT", help="Output segmentation file (.json or .csv)")
    parser.add_argument(
        "--method",
        choices=["energy", "silero"],
        default="silero",
        help="VAD method to use (default: silero)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Drop segments shorter than SEC seconds (default: 0.5)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.4,
        metavar="SEC",
        help="Merge segments with silence gap shorter than SEC seconds (default: 0.4)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output against the downstream contract after writing",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run all VAD methods and print a comparison table",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Directory for all output files: meeting.wav, segments, etc. (default: output/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success = run_pipeline(
        input_path=args.INPUT,
        output_path=args.OUTPUT,
        method=args.method,
        min_duration=args.min_duration,
        min_gap=args.min_gap,
        validate=args.validate,
        benchmark=args.benchmark,
        output_dir=args.output_dir,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
