"""Build a self-contained contact sheet (base64-embedded) from shots/*.png."""
import base64
from pathlib import Path

HERE = Path(__file__).parent
TITLES = {
    "01_authentication": "1. Authentication — card centered in window",
    "02_mission_control": "2. Mission Control — controls | radar | camera side-by-side, 4×4 telemetry",
    "03_test_lab": "3. Test Lab",
    "04_twin_full_scene": "4. Digital Twin — full scene (toolbar rows + read-only motion + mission strip)",
    "05_twin_beam_impact": "5. Beam Impact view — live impact at receiver aperture",
    "06_twin_true_fullscreen": "6. Twin TRUE full screen — canvas owns the window (ESC exits)",
    "07_handoff_window_dialog": "7. HANDOFF WINDOW dialog — real telemetry",
    "08_why_locked_dialog": "8. WHY LOCKED? (READY case)",
    "09_why_not_locked_dialog": "9. WHY NOT LOCKED? (NOT-READY case, forced loss)",
    "10_replay_reports": "10. Replay & Reports",
}

cards = []
for png in sorted(HERE.glob("shots/*.png")):
    key = png.stem
    title = TITLES.get(key, key)
    data = base64.b64encode(png.read_bytes()).decode()
    cards.append(f'''
    <figure>
      <figcaption>{title}</figcaption>
      <img src="data:image/png;base64,{data}" alt="{title}">
    </figure>''')

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FSOC — Stage 20 Visual Inspection</title>
<style>
  body {{ background: #0b0f14; color: #d7e6ef; font-family: "Segoe UI", sans-serif; margin: 24px; }}
  h1 {{ font-size: 20px; letter-spacing: 1px; }}
  p.meta {{ color: #8ba3b4; font-size: 13px; }}
  figure {{ margin: 0 0 28px 0; }}
  figcaption {{ font-size: 14px; font-weight: 700; color: #9fd8ff; margin-bottom: 6px; }}
  img {{ max-width: 100%; border: 1px solid #24313d; border-radius: 6px; display: block; }}
</style>
</head>
<body>
<h1>FSOC — FINAL VISUAL INSPECTION (Stage 20)</h1>
<p class="meta">Rendered offscreen at 1845×917 from the committed build. All images are real Qt renders, not mock-ups.</p>
{''.join(cards)}
</body>
</html>
'''

out = HERE / "ui_contact_sheet.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(cards)} screens)")
