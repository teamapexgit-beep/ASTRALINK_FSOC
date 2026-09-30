"""Probe 2: faithful user flows.
A) Start mission under disturbance via REAL widget signals (no reset):
   exactly what happens when a user picks a preset while running.
B) Track clear first, then switch disturbance MID-MISSION (no reset):
   does HANDOFF READY (re)appear?

Run:  python _readiness_probe2.py
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

window = gui.window
app = gui.app
window.auth_input.setText("isro@1969")
window._authenticate()
window.timer.stop()


def run_frames(n):
    max_streak = 0
    states = {}
    for _ in range(n):
        window.update_simulation()
        max_streak = max(max_streak, window.link_ready_streak)
        st = window.tracker.state
        states[st] = states.get(st, 0) + 1
    return max_streak, states


def report(title, pre_streak, post_streak, post_states, caption):
    ready = post_streak >= window.link_required_frames
    print(f"{title:<34} streakPre={pre_streak:>4} streakPost={post_streak:>4} "
          f"{('READY' if ready else 'NO'):<6} {post_states}")
    print(f"    {caption}")


PRESETS = ["Sensor Noise", "Vibration", "Atmosphere", "Full Combined", "Extreme PAT"]

print("=== A) preset applied mid-mission after clear lock (real user flow) ===")
for preset in PRESETS:
    window.reset_simulation()
    window.timer.stop()
    for _ in range(240):  # 8 s clear tracking
        window.update_simulation()
    pre, _ = run_frames(0)
    window.preset_selector.setCurrentText(preset)  # REAL signal path
    post, states = run_frames(300)
    report(f"mid-flight -> {preset}", pre, post, states, window.link_conditions_label.text())

print()
print("=== B) individual disturbances via real widget signals, mid-mission ===")
INDIVIDUAL = [
    ("Salt & Pepper 10%", lambda: (window.noise_selector.setCurrentText("Salt & Pepper"),
                                   window.noise_level_slider.setValue(10))),
    ("Gaussian 20", lambda: (window.noise_selector.setCurrentText("Gaussian"),
                             window.noise_level_slider.setValue(20))),
    ("Poisson 15", lambda: (window.noise_selector.setCurrentText("Poisson"),
                            window.noise_level_slider.setValue(15))),
    ("Turbulence 10px", lambda: (window.turbulence_check.setChecked(True),
                                 window.turbulence_slider.setValue(10))),
    ("Platform Motion 10px", lambda: (window.platform_check.setChecked(True),
                                      window.platform_slider.setValue(10))),
    ("Camera Jitter 15px", lambda: (window.jitter_check.setChecked(True),
                                    window.jitter_slider.setValue(15))),
    ("Fog 70%", lambda: (window.atmosphere_selector.setCurrentText("Fog"),
                         window.atmosphere_slider.setValue(70))),
    ("Haze 35%", lambda: (window.atmosphere_selector.setCurrentText("Haze"),
                          window.atmosphere_slider.setValue(35))),
]
for label, apply_fn in INDIVIDUAL:
    window.reset_simulation()
    window.timer.stop()
    for _ in range(240):
        window.update_simulation()
    pre, _ = run_frames(0)
    apply_fn()  # real signal handlers fire (sliders/combos connected in __init__)
    window._apply_disturbances()  # checkbox/slider handlers already did this; harmless
    post, states = run_frames(300)
    report(f"mid-flight -> {label}", pre, post, states, window.link_conditions_label.text())
