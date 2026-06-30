import numpy as np

DEFAULT_THRESHOLD = 300.0
SPEECH_LO = 1.5   # Hz — lower bound of speech-rhythm frequency band
SPEECH_HI = 6.0   # Hz — upper bound


def speech_power(signal, fs):
    """Absolute energy in the 1.5–6 Hz speech-rhythm band.

    Speech produces fast rhythmic mouth movement -> high power in this band.
    Smiling / head-nodding = slow movement -> below SPEECH_LO -> filtered out.
    Silence = weak noise -> low overall power.
    """
    sig = np.asarray(signal, float)
    if len(sig) < 32:
        return 0.0
    sig = sig - sig.mean()
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    power = np.abs(np.fft.rfft(sig)) ** 2
    hi = min(SPEECH_HI, fs / 2 * 0.95)  # stay below Nyquist
    return float(power[(freqs >= SPEECH_LO) & (freqs <= hi)].sum() / len(sig) * 1000)


def scores(series, fs):
    """Return {face_id: power_value} — use for calibration before committing to a threshold."""
    return {fid: speech_power(vals, fs) for fid, vals in series.items()}


def detect_speaking(series, fs, threshold=DEFAULT_THRESHOLD):
    """Classify each face_id as speaking/silent by FFT speech-band power.

    Result on test video: 6/7.
    More robust to smiles/nods than std method, but scored lower on this dataset.
    Re-run with a different threshold by calling scores() first to see the distribution.
    """
    return {fid: speech_power(vals, fs) > threshold for fid, vals in series.items()}
