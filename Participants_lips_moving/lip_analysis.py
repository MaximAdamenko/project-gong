from landmarker import measure_mar
from loader import load_face_boxes
from reporter import report


def run_pipeline(video_path, faces_json, model_path, method="std", threshold=None, frame_skip=3):
    print(f"[1/3] loading face boxes from {faces_json}")
    boxes_by_frame, total_participants, fps = load_face_boxes(faces_json)
    fs = fps / frame_skip  # effective MAR sampling rate

    print(f"[2/3] measuring lip movement  (frame_skip={frame_skip}, sampling={fs:.1f}Hz)")
    series = measure_mar(video_path, boxes_by_frame, model_path=model_path, frame_skip=frame_skip)

    print(f"[3/3] classifying speakers — method: {method}")
    if method == "std":
        from detectors.std_detector import detect_speaking, scores, DEFAULT_THRESHOLD
        thr = threshold if threshold is not None else DEFAULT_THRESHOLD
        decisions  = detect_speaking(series, threshold=thr)
        score_map  = scores(series)
    else:
        from detectors.fft_detector import detect_speaking, scores, DEFAULT_THRESHOLD, SPEECH_LO, SPEECH_HI
        thr = threshold if threshold is not None else DEFAULT_THRESHOLD
        print(f"      speech band: {SPEECH_LO}-{min(SPEECH_HI, fs / 2 * 0.95):.1f}Hz")
        decisions  = detect_speaking(series, fs=fs, threshold=thr)
        score_map  = scores(series, fs=fs)

    return report(series, decisions, score_map, total_participants, method)
