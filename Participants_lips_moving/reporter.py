def report(series, decisions, score_map, total_participants, method):
    """Print per-face results table and summary counts."""
    col = "std" if method == "std" else "speech_power"
    print(f"\n{'face_id':<10}{col:<18}{'frames':<9}speaking?")
    print("-" * 46)
    for fid in sorted(series):
        status = "speaking" if decisions[fid] else "silent"
        print(f"{fid:<10}{round(score_map[fid], 4):<18}{len(series[fid]):<9}{status}")

    speaking_count = sum(decisions.values())
    print(f"\ntotal participants (from Team 3): {total_participants}")
    print(f"speaking (by lip movement):       {speaking_count}")
    return speaking_count
