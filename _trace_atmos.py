"""Trace the failing mid-flight Atmosphere case frame by frame.

Run:  python _trace_atmos.py
"""
import os
import sys
import importlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_real_exit = sys.exit
sys.exit = lambda *a, **k: None
from PySide6.QtWidgets import QApplication as _QApplication  # noqa: E402

_real_exec = _QApplication.exec
_QApplication.exec = lambda self, *a, **k: 0
try:
    gui = importlib.import_module("gui")
except (SystemExit, RuntimeError):
    gui = sys.modules.get("gui")
finally:
    _QApplication.exec = _real_exec
    sys.exit = _real_exit

import math  # noqa: E402
import config  # noqa: E402

window = gui.window
app = gui.app
window.auth_input.setText("isro@1969")
window._authenticate()
window.timer.stop()

window.reset_simulation()
window.timer.stop()
for _ in range(240):
    window.update_simulation()

window.preset_selector.setCurrentText("Atmosphere")

for i in range(300):
    frame = window.simulator.get_frame()
    gt = window.simulator.get_ground_truth_image_position()
    det, info = __import__("detector").detect_beacon(
        frame, return_details=True,
        expected_position=None,
        preferred_signature=window.simulator.get_designated_target_signature(),
    )
    raw = det
    window.update_simulation()
    tr = window.tracker
    cx, cy = config.FRAME_WIDTH // 2, config.FRAME_HEIGHT // 2
    err = math.hypot(tr.x - cx, tr.y - cy) if tr.x is not None else None
    if i < 25 or i % 25 == 0:
        print(f"f{i:>3} rawDet={'Y' if raw else 'n'} mode={info.get('mode','?'):<18} "
              f"idRej={str(info.get('identity_rejected','')):<5} cands={info.get('candidate_count',0):>2} "
              f"| closedState={tr.state:<10} lost={tr.lost_frames:>2} err={err and round(err,1)} "
              f"| gt={gt and (round(gt[0]), round(gt[1]))} camPan={window.simulator.camera_pan:.2f}")
