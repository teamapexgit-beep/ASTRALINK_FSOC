"""Offscreen disturbance harness: for each disturbance case, reset the mission
(like the real GUI reset flow), apply the disturbance, run the real 30 Hz loop
and report gate outcomes.

Run:  python _readiness_probe.py
"""
import os
import sys
import time
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

import config  # noqa: E402

window = gui.window
app = gui.app
window.auth_input.setText("isro@1969")
window._authenticate()
window.timer.stop()

WARM = 60
RUN = 300  # 10 s at 30 Hz


def run_case(setup, label):
    window.reset_simulation()
    window.timer.stop()
    setup()
    for _ in range(WARM):
        window.update_simulation()
    max_streak = 0
    fps_min = 1e9
    states = {}
    for _ in range(RUN):
        window.update_simulation()
        max_streak = max(max_streak, window.link_ready_streak)
        fps_min = min(fps_min, window.fps)
        st = window.tracker.state
        states[st] = states.get(st, 0) + 1
    return {
        "label": label,
        "max_streak": max_streak,
        "fps_min": fps_min,
        "states": states,
        "caption": window.link_conditions_label.text(),
    }


def preset(name):
    return lambda: window._apply_disturbance_preset(name)


def combo(noise=None, noise_level=None, turbulence=None, tpx=None,
          jitter=None, jpx=None, platform=None, ppx=None, atmosphere=None, alevel=None):
    def _apply():
        window.noise_selector.blockSignals(True)
        if noise is not None:
            window.noise_selector.setCurrentText(noise)
        window.noise_selector.blockSignals(False)
        window.noise_level_slider.blockSignals(True)
        if noise_level is not None:
            window.noise_level_slider.setValue(noise_level)
        window.noise_level_slider.blockSignals(False)
        for check, slider, val, enabled in [
            (window.turbulence_check, window.turbulence_slider, tpx, turbulence),
            (window.jitter_check, window.jitter_slider, jpx, jitter),
            (window.platform_check, window.platform_slider, ppx, platform),
        ]:
            check.blockSignals(True)
            if enabled is not None:
                check.setChecked(enabled)
            check.blockSignals(False)
            slider.blockSignals(True)
            if val is not None:
                slider.setValue(val)
            slider.blockSignals(False)
        window.atmosphere_selector.blockSignals(True)
        if atmosphere is not None:
            window.atmosphere_selector.setCurrentText(atmosphere)
        window.atmosphere_selector.blockSignals(False)
        window.atmosphere_slider.blockSignals(True)
        if alevel is not None:
            window.atmosphere_slider.setValue(alevel)
        window.atmosphere_slider.blockSignals(False)
        window._apply_disturbances()
    return _apply


CASES = [
    (preset("Sensor Noise"), "preset: Sensor Noise"),
    (preset("Vibration"), "preset: Vibration"),
    (preset("Atmosphere"), "preset: Atmosphere"),
    (preset("Full Combined"), "preset: Full Combined"),
    (preset("Extreme PAT"), "preset: Extreme PAT"),
    (combo(noise="Salt & Pepper", noise_level=10), "Salt & Pepper 10%"),
    (combo(noise="Gaussian", noise_level=20), "Gaussian 20"),
    (combo(turbulence=True, tpx=10), "Turbulence 10px"),
    (combo(platform=True, ppx=10), "Platform Motion 10px"),
    (combo(jitter=True, jpx=15), "Camera Jitter 15px"),
]

print(f"{'case':<24} {'maxStreak':>9} {'ready?':<8} {'fpsMin':>7}  states")
for setup, label in CASES:
    r = run_case(setup, label)
    ready = r["max_streak"] >= window.link_required_frames
    print(f"{label:<24} {r['max_streak']:>9} {('READY' if ready else 'NO'):<8} {r['fps_min']:>7.1f}  {r['states']}")
    print(f"    {r['caption']}")
