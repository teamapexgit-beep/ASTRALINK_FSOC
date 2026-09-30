"""Full GUI-path benchmark test: real BenchmarkScreen + real worker thread,
from video selection-equivalent state through results population.

Run:  python _benchmark_gui_test.py
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

app = gui.app
window = gui.window

VIDEO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Benchmark vid 1.mp4"))

# Reproduce benchmark_video()'s body minus the QFileDialogs (headless):
window.pause_simulation()
window._benchmark_screen = gui.BenchmarkScreen(window)
window._benchmark_screen.show()
window._benchmark_screen.raise_()
ok_preview = window._benchmark_screen.set_video_preview(VIDEO)
window._benchmark_screen.info_value.setText(f"{os.path.basename(VIDEO)}\nGround truth: NOT PROVIDED")
window._benchmark_screen.set_progress(0, "Starting standardized benchmark…")
window._benchmark_screen.run_button.setEnabled(False)
window._benchmark_screen.cancel_button.setEnabled(True)

print("preview_ok =", ok_preview)

window._benchmark_worker = gui.VideoBenchmarkWorker(
    VIDEO,
    ground_truth_path=None,
    preferred_signature=None,  # AUTO, matching the fixed benchmark_video flow
    supplied_status="NOT_PROVIDED",
)
window._benchmark_worker.finished.connect(window._on_benchmark_screen_finished)
window._benchmark_worker.failed.connect(window._on_benchmark_screen_failed)
window._benchmark_worker.progress.connect(window._on_benchmark_screen_progress)
window._benchmark_worker.start()

# Pump the real event loop until the worker finishes (timeout 120 s)
start = time.time()
progress_seen = []
while window._benchmark_worker.isRunning() and time.time() - start < 120:
    app.processEvents()
    lb = window._benchmark_screen.progress_label.text()
    if not progress_seen or progress_seen[-1] != lb:
        progress_seen.append(lb)
    time.sleep(0.01)
for _ in range(20):
    app.processEvents()
    time.sleep(0.01)

screen = window._benchmark_screen
print("progress samples:", progress_seen[:3], "...", progress_seen[-2:])
print("progress bar:", screen.progress_bar.value(), "| label:", screen.progress_label.text())
print("results:")
for name, val in screen.result_labels.items():
    print(f"  {name:<15} = {val.text()}")
assert screen.result_labels["FPS"].text() not in ("", "—"), "FPS not populated"
print("GUI-PATH BENCHMARK: OK")
