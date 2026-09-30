"""Final real-GUI verification: telemetry values + HANDOFF READY under disturbance.

Run:  python _final_verify.py
"""
import os
import sys
import time
import importlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_real_exit = sys.exit
sys.exit = lambda *a, **k: None
from PySide6.QtWidgets import QApplication as _QApplication  # noqa: E402

_real_exec = _QApplication.exec
_QApplication.exec = lambda self, *a, **k: 0
gui = importlib.import_module("gui")
_QApplication.exec = _real_exec
sys.exit = _real_exit

app = gui.app
window = gui.window

METRICS = ["Processing FPS", "Tracking Error", "Center Error", "Acquisition",
           "Re-acquisition", "Target Loss", "Avg Tracking Error", "Max Deviation",
           "RMSE", "Lock Retention", "State", "Pan", "Tilt", "Velocity",
           "Filter Confidence", "AI Confidence"]
LABELS = {
    "Processing FPS": window.t_fps, "Tracking Error": window.t_error,
    "Center Error": window.t_centroid, "Acquisition": window.t_acq,
    "Re-acquisition": window.t_reacq, "Target Loss": window.t_loss,
    "Avg Tracking Error": window.t_avg, "Max Deviation": window.t_max,
    "RMSE": window.t_rmse, "Lock Retention": window.t_lock,
    "State": window.t_state, "Pan": window.t_pan, "Tilt": window.t_tilt,
    "Velocity": window.t_velocity, "Filter Confidence": window.t_confidence,
    "AI Confidence": window.t_ai,
}

window.resize(1600, 980)
window.show()
window.auth_input.setText("isro@1969")
window._authenticate()

# ~8 s clear tracking, then apply Full Combined mid-flight via the real signal path.
t_end = time.time() + 8.0
while time.time() < t_end:
    app.processEvents()
    time.sleep(0.005)

print(f"before disturbance: frames={window.frames_processed} state={window.tracker.state} "
      f"streak={window.link_ready_streak}")
window.preset_selector.setCurrentText("Full Combined")

# ~10 s under disturbance
t_end = time.time() + 10.0
while time.time() < t_end:
    app.processEvents()
    time.sleep(0.005)

print(f"under disturbance: frames={window.frames_processed} state={window.tracker.state} "
      f"streak={window.link_ready_streak}/{window.link_required_frames}")
print("readiness label:", window.link_readiness_label.text())
print("caption:", window.link_conditions_label.text())
missing = [m for m in METRICS if not LABELS[m].text()]
print("empty telemetry values:", missing if missing else "none")
for m in METRICS:
    print(f"  {m:<20} = {LABELS[m].text()}")

pix = window.grab()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry_ready_disturbance.png")
pix.save(out)
print("screenshot:", out)
