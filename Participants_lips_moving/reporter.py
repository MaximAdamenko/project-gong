import json
import os


def report(series, decisions, score_map, total_participants, method, output_path=None):
    """Print per-face results table, summary counts, and optionally write lips_output.json."""
    col = "std" if method == "std" else "speech_power"
    print(f"\n{'face_id':<10}{col:<18}{'frames':<9}speaking?")
    print("-" * 46)
    for fid in sorted(series):
        status = "speaking" if decisions[fid] else "silent"
        print(f"{fid:<10}{round(score_map[fid], 4):<18}{len(series[fid]):<9}{status}")

    speaking_count = sum(decisions.values())
    print(f"\ntotal participants (from Team 3): {total_participants}")
    print(f"speaking (by lip movement):       {speaking_count}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        payload = {
            "method": method,
            "total_participants": total_participants,
            "speaking_count": int(speaking_count),
            "per_face": {
                fid: {"score": round(score_map[fid], 4), "speaking": bool(decisions[fid])}
                for fid in sorted(series)
            },
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"lips output saved -> {output_path}")

    return speaking_count
