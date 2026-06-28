"""
audio_io.py — Audio loading and canonical WAV conversion.

Single job: accept any input file, return float32 mono 16 kHz samples.
"""

import os
import subprocess
import sys

import numpy as np
import soundfile as sf


def prepare_audio(input_path: str, output_path: str = "meeting.wav") -> tuple[np.ndarray, int]:
    """
    Load input file and return (float32 mono samples, sample_rate).

    Non-WAV files and WAV files that are not 16 kHz mono are converted
    via ffmpeg to a canonical 16 kHz / mono / 16-bit PCM WAV saved at
    output_path before loading.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    _, ext = os.path.splitext(input_path.lower())

    needs_conversion = ext != ".wav"
    if not needs_conversion:
        probe, probe_sr = sf.read(input_path, dtype="float32")
        if probe_sr != 16000 or probe.ndim > 1:
            needs_conversion = True

    target_file = input_path
    if needs_conversion:
        print(f"Converting '{input_path}' -> '{output_path}' via ffmpeg...")
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-vn", "-sample_fmt", "s16",
            output_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"ffmpeg failed:\n{result.stderr.decode()}")
            sys.exit(1)
        target_file = output_path

    print(f"Loading audio from '{target_file}'...")
    data, samplerate = sf.read(target_file, dtype="float32")
    return data, samplerate
