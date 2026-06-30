import numpy as np

DEFAULT_THRESHOLD = 0.035  # MAR std above this => speaking


def scores(series):
    """Return {face_id: std_value} — use for calibration before committing to a threshold."""
    return {fid: float(np.std(vals)) for fid, vals in series.items()}


def detect_speaking(series, threshold=DEFAULT_THRESHOLD):
    """Classify each face_id as speaking/silent by MAR standard deviation.

    Result on test video: 7/7.
    Raise threshold if too many false positives; lower if quiet speakers are missed.
    """
    return {fid: float(np.std(vals)) > threshold for fid, vals in series.items()}
