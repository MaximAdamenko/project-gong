"""
Run speaker diarization on YOUR video and open a visual report.
Called by "RUN MY VIDEO.bat" — receives the video path as argv[1].
"""

import os
import sys
import json
import subprocess
import webbrowser
from pathlib import Path
from collections import defaultdict

BASE     = Path(__file__).resolve().parent
PIPELINE = str(BASE / "offline_pipeline.py")
OUT_DIR  = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

COLORS = [
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
    "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fmt_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m} min {s} sec" if m else f"{s} sec"


def speaker_stats(segs: list, total_dur: float) -> list:
    """Return per-speaker total speaking time and percentage."""
    totals = defaultdict(float)
    for seg in segs:
        totals[seg["speaker_id"]] += seg["end"] - seg["start"]
    result = []
    for spk, dur in sorted(totals.items()):
        result.append({
            "id":      spk,
            "dur":     round(dur, 1),
            "pct":     round(dur / total_dur * 100, 1),
        })
    return sorted(result, key=lambda x: -x["dur"])


def build_timeline_svg(segs, total_dur, color_map, height=40):
    W = 860
    rects, labels = [], []
    for seg in segs:
        x = seg["start"] / total_dur * W
        w = max((seg["end"] - seg["start"]) / total_dur * W, 2)
        c = color_map.get(seg["speaker_id"], "#ccc")
        tip = f'{seg["speaker_id"]}  {fmt_time(seg["start"])} – {fmt_time(seg["end"])}  ({fmt_duration(seg["end"]-seg["start"])})'
        rects.append(
            f'<rect x="{x:.1f}" y="2" width="{w:.1f}" height="{height-4}" '
            f'rx="4" fill="{c}" opacity="0.92"><title>{tip}</title></rect>'
        )
        if w > 55:
            labels.append(
                f'<text x="{x+w/2:.1f}" y="{height/2+5:.0f}" '
                f'text-anchor="middle" font-size="11" fill="white" '
                f'font-family="Segoe UI,Arial" font-weight="600">'
                f'{seg["speaker_id"]}</text>'
            )

    # Time ticks every 60s (or 30s for short videos)
    tick_step = 30 if total_dur < 300 else 60
    ticks = []
    t = 0
    while t <= total_dur:
        x = t / total_dur * W
        ticks.append(f'<line x1="{x:.0f}" y1="0" x2="{x:.0f}" y2="{height}" '
                     f'stroke="#e0e0e0" stroke-width="1"/>')
        ticks.append(f'<text x="{x:.0f}" y="{height+13}" text-anchor="middle" '
                     f'font-size="10" fill="#aaa" font-family="Segoe UI,Arial">'
                     f'{fmt_time(t)}</text>')
        t += tick_step

    return (f'<svg width="{W}" height="{height+18}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(ticks) + "".join(rects) + "".join(labels) + "</svg>")


def build_html(video_name, segs, total_dur, color_map, stats):
    # Speaker cards
    stat_cards = ""
    for s in stats:
        color = color_map.get(s["id"], "#ccc")
        bar_w = s["pct"]
        stat_cards += f"""
        <div class="stat-card">
          <div class="stat-header">
            <span class="dot" style="background:{color}"></span>
            <strong>{s['id']}</strong>
          </div>
          <div class="stat-bar-bg">
            <div class="stat-bar" style="width:{bar_w}%;background:{color}"></div>
          </div>
          <div class="stat-nums">{fmt_duration(s['dur'])} &nbsp;({s['pct']}%)</div>
        </div>"""

    svg = build_timeline_svg(segs, total_dur, color_map)
    n_speakers = len(set(s["speaker_id"] for s in segs))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Who Spoke in {video_name}?</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    font-family:"Segoe UI",Arial,sans-serif;
    background:#f4f6f9; color:#333; padding:36px 28px;
  }}
  h1 {{ font-size:26px; font-weight:700; color:#1a1a2e; margin-bottom:4px; }}
  .subtitle {{ font-size:14px; color:#888; margin-bottom:28px; }}
  .card {{
    background:white; border-radius:14px;
    box-shadow:0 2px 14px rgba(0,0,0,0.08);
    padding:28px 32px; margin-bottom:24px;
  }}
  .card-title {{ font-size:16px; font-weight:700; color:#1a1a2e; margin-bottom:6px; }}
  .meta {{ font-size:13px; color:#999; margin-bottom:20px; }}
  .tl-wrap {{ overflow-x:auto; }}
  .stats-grid {{
    display:grid;
    grid-template-columns:repeat(auto-fill, minmax(200px, 1fr));
    gap:14px; margin-top:4px;
  }}
  .stat-card {{
    border:1px solid #eee; border-radius:10px;
    padding:14px 16px; background:#fafafa;
  }}
  .stat-header {{ display:flex; align-items:center; gap:8px; margin-bottom:10px; }}
  .dot {{
    display:inline-block; width:13px; height:13px;
    border-radius:50%; flex-shrink:0;
  }}
  .stat-bar-bg {{
    background:#eee; border-radius:4px; height:8px; margin-bottom:6px;
  }}
  .stat-bar {{ height:8px; border-radius:4px; }}
  .stat-nums {{ font-size:12px; color:#777; }}
  .tip-box {{
    background:#eef6ff; border-left:4px solid #4E79A7;
    border-radius:8px; padding:14px 18px;
    font-size:13px; color:#555; line-height:1.8;
    margin-bottom:28px;
  }}
  .tip-box strong {{ color:#1a1a2e; }}
  .seg-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .seg-table th {{
    text-align:left; padding:8px 12px;
    background:#f0f2f5; color:#555;
    font-weight:600; border-bottom:2px solid #e0e0e0;
  }}
  .seg-table td {{ padding:7px 12px; border-bottom:1px solid #f0f0f0; }}
  .seg-table tr:last-child td {{ border-bottom:none; }}
  .spk-badge {{
    display:inline-block; padding:2px 10px; border-radius:12px;
    color:white; font-size:12px; font-weight:600;
  }}
</style>
</head>
<body>

<h1>&#127908; Who Spoke in "{video_name}"?</h1>
<p class="subtitle">Speaker diarization — automatic detection of who spoke and when</p>

<div class="tip-box">
  <strong>How to read this:</strong><br>
  Each color = a different person's voice. Hover any bar to see the exact time range.<br>
  The system does NOT know names — it labels speakers as SPEAKER_00, SPEAKER_01, etc.<br>
  <strong>You</strong> can verify: does the same color always match the same person speaking?
</div>

<!-- Timeline -->
<div class="card">
  <div class="card-title">Speaker Timeline</div>
  <p class="meta">{n_speakers} speakers detected &nbsp;|&nbsp; Total duration: {fmt_duration(total_dur)}</p>
  <div class="tl-wrap">{svg}</div>
</div>

<!-- Speaking time breakdown -->
<div class="card">
  <div class="card-title">Speaking Time per Person</div>
  <p class="meta">How much of the audio each speaker occupied</p>
  <div class="stats-grid">{stat_cards}</div>
</div>

<!-- Segment table -->
<div class="card">
  <div class="card-title">Full Segment List</div>
  <p class="meta">Every speech segment detected, in order</p>
  <table class="seg-table">
    <thead>
      <tr>
        <th>#</th><th>Speaker</th><th>Start</th><th>End</th><th>Duration</th>
      </tr>
    </thead>
    <tbody>
      {''.join(
          f'<tr>'
          f'<td style="color:#bbb">{i+1}</td>'
          f'<td><span class="spk-badge" style="background:{color_map.get(s["speaker_id"],"#ccc")}">'
          f'{s["speaker_id"]}</span></td>'
          f'<td>{fmt_time(s["start"])}</td>'
          f'<td>{fmt_time(s["end"])}</td>'
          f'<td>{fmt_duration(s["end"]-s["start"])}</td>'
          f'</tr>'
          for i, s in enumerate(segs)
      )}
    </tbody>
  </table>
</div>

</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Get video path from argument or ask
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1].strip('"'))
    else:
        print("\n  Drag your video file onto 'RUN MY VIDEO.bat'")
        print("  OR paste the full path here:")
        raw = input("  > ").strip().strip('"')
        if not raw:
            print("No path provided.")
            input("Press Enter to exit...")
            return
        video_path = Path(raw)

    if not video_path.exists():
        print(f"\nERROR: File not found: {video_path}")
        input("Press Enter to exit...")
        return

    print(f"\n  Video : {video_path.name}")
    print(f"  Size  : {video_path.stat().st_size / 1e6:.1f} MB\n")

    # Output paths
    safe_name = video_path.stem.replace(" ", "_")
    out_json  = OUT_DIR / f"{safe_name}_predicted.json"
    out_html  = OUT_DIR / f"{safe_name}_results.html"

    # Run diarization
    print("  Running speaker diarization (this may take 1-2 minutes)...")
    r = subprocess.run(
        [sys.executable, PIPELINE, str(video_path), str(out_json)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print("\nERROR during diarization:")
        print(r.stderr[-500:])
        input("\nPress Enter to exit...")
        return

    # Load results
    with open(out_json) as f:
        segs = json.load(f)

    if not segs:
        print("No speech segments detected. Check if the video has audio.")
        input("Press Enter to exit...")
        return

    total_dur  = max(s["end"] for s in segs)
    spk_ids    = sorted(set(s["speaker_id"] for s in segs))
    color_map  = {sid: COLORS[i % len(COLORS)] for i, sid in enumerate(spk_ids)}
    stats      = speaker_stats(segs, total_dur)

    # Print quick summary to terminal
    print(f"\n  Detected {len(spk_ids)} speaker(s) in {fmt_duration(total_dur)}\n")
    for s in stats:
        bar = "#" * int(s["pct"] / 3)
        print(f"  {s['id']:12s}  {bar:30s}  {fmt_duration(s['dur'])} ({s['pct']}%)")

    # Generate HTML
    html = build_html(video_path.name, segs, total_dur, color_map, stats)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  Report saved -> {out_html}")
    print("  Opening in browser...")
    webbrowser.open(out_html.as_uri())


if __name__ == "__main__":
    main()
