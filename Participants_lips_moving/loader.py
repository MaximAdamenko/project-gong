import json


def load_face_boxes(json_path):
    """Load Team 3 output: frame_index -> [(face_id, x, y, w, h normalized)]."""
    data = json.load(open(json_path))
    by_frame = {
        fr["frame_index"]: [
            (f["face_id"],
             f["bounding_box"]["x_min"], f["bounding_box"]["y_min"],
             f["bounding_box"]["width"], f["bounding_box"]["height"])
            for f in fr["faces_detected"]
        ]
        for fr in data["frames"]
    }
    return by_frame, data.get("total_participants_detected"), data.get("fps", 30)
