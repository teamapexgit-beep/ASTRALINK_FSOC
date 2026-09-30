"""Condition-response check for the live BENCHMARK PERFORMANCE GRAPHS.

Runs the REAL simulation twice under different conditions and proves the graph
panel is fed genuine per-frame measurements (never static/fake values):

  RUN A  normal   : low noise, no turbulence/jitter/platform, beacon speed 4
  RUN B  disturbed: max noise, turbulence+jitter+platform, beacon speed 10

For each run it reads the panel's own stored series and checks finite values,
a fresh start (frame 1 first) and a real change between runs.

Run:  python _graph_conditions_test.py
"""
import math
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROBLEMS = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        PROBLEMS.append(f"{label} {detail}")


# gui.py builds the app/window at import time and, like every Qt entry
# point, calls sys.exit() to hand control to the event loop. Headless tests
# must neuter both so the module import returns instead of terminating.
_real_exit = sys.exit
sys.exit = lambda *a, **k: None
try:
    from PySide6.QtWidgets import QApplication as _QApp
    _real_exec = _QApp.exec
    _QApp.exec = lambda self, *a, **k: 0
    import gui as gui_mod
finally:
    _QApp.exec = _real_exec
    sys.exit = _real_exit

app = gui_mod.app
w = gui_mod.window


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def mean(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def maxi(values):
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def snapshot(panel):
    return {
        "n": len(panel.timestamps),
        "frames": list(panel.frames[:4]),
        "mean_center": mean(panel.center_errors),
        "max_center": maxi(panel.center_errors),
        "mean_fps": mean(panel.fps_values),
        "mean_det": mean(panel.detector_confidence),
        "mean_ai": mean(panel.ai_confidence),
        "mean_pan": mean(panel.pan_values),
    }


# ---- auth -----------------------------------------------------------------
w.auth_input.setText("isro@1969")
w._authenticate()
pump(0.8)

# ---- RUN A: normal conditions --------------------------------------------
w.noise_level_slider.setValue(0)
w.turbulence_check.setChecked(False)
w.jitter_check.setChecked(False)
w.platform_check.setChecked(False)
w.speed_slider.setValue(40)          # 4.0 px/frame
w._apply_runtime_settings()
pump(0.2)

w.reset_simulation()                 # fresh run -> panel must restart empty
pump(0.1)
panel = w.graph_panel
check(list(panel.frames[:1]) in ([], [1]),
      "RUN A: a new run starts the graphs from frame 1", str(panel.frames[:2]))
pump(3.0)
normal = snapshot(panel)
print(f"        A normal: {normal}")
check(normal["n"] > 10, "RUN A: graphs received real samples", str(normal["n"]))

# ---- RUN B: high disturbance + fast target -------------------------------
w.noise_level_slider.setValue(20)
w.noise_selector.setCurrentText(w.noise_selector.itemText(
    min(1, w.noise_selector.count() - 1)))  # a non-default noise type if any
w.turbulence_check.setChecked(True)
w.turbulence_slider.setValue(20)
w.jitter_check.setChecked(True)
w.jitter_slider.setValue(20)
w.platform_check.setChecked(True)
w.platform_slider.setValue(20)
w.speed_slider.setValue(100)         # 10.0 px/frame (fast target)
w._apply_runtime_settings()
pump(0.2)

w.reset_simulation()
pump(0.1)
check(list(panel.frames[:1]) in ([], [1]),
      "RUN B: the panel was cleared before the new run", str(panel.frames[:2]))
pump(3.0)
disturbed = snapshot(panel)
print(f"        B disturb: {disturbed}")
check(disturbed["n"] > 10, "RUN B: graphs received real samples", str(disturbed["n"]))

# ---- every plotted value is a real finite number -------------------------
bad = []
for name, arr in (("center", panel.center_errors), ("fps", panel.fps_values),
                  ("pan", panel.pan_values), ("tilt", panel.tilt_values),
                  ("det", panel.detector_confidence), ("ai", panel.ai_confidence),
                  ("target_x", panel.target_x), ("pred_x", panel.predicted_x)):
    for v in arr:
        if v is not None and not math.isfinite(v):
            bad.append(name)
check(not bad, "no NaN/Infinity reached any live graph series", str(set(bad)))

# ---- the readings really moved with the conditions -----------------------
diff = {k: (normal[k], disturbed[k]) for k in normal
        if k not in ("n", "frames") and normal[k] != disturbed[k]}
check(len(diff) >= 3, "the readings changed with the conditions",
      f"{len(diff)}/6 metrics differ: {diff}")
check(len(panel.timestamps) > 0 and panel.timestamps[0] < 5.0,
      "the current run's series is the one on screen (no stale frames)",
      str(panel.timestamps[:2]))

print("\n" + "=" * 60)
if PROBLEMS:
    print(f"RESULT: {len(PROBLEMS)} PROBLEM(S)")
    for p in PROBLEMS:
        print("  -", p)
    sys.exit(1)
print("RESULT: ALL CONDITION CHECKS PASSED")
print("=" * 60)
