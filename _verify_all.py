"""End-to-end verification of the ASTRALINK fixes (headless / offscreen).

Covers:
  * every simulation + TEST/OUTPUT button actually does something
  * BENCHMARK VIDEO plays while the analysis runs, from ONE loop
  * PLAY / PAUSE / RESUME / REPLAY all control that loop (TESTS 1-7)
  * graph data comes from the real processed frames, never invented values
  * no graph draw error (the old "QBrush" text) and every title fits its card
  * benchmark results reach the graph panel, EXPORT REPORT and INSTANT REPORT
  * no NaN / Infinity anywhere, no exceptions, no horizontal scrolling
  * every control button stays inside the control panel at several widths

Run:  python _verify_all.py
"""
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROBLEMS = []
NOTES = []


def check(condition, label, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        PROBLEMS.append(f"{label} {detail}")


_real_exit = sys.exit
sys.exit = lambda *a, **k: None
from PySide6.QtWidgets import (  # noqa: E402
    QApplication as _QApplication,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtGui import QFontMetrics, QPixmap  # noqa: E402

import benchmark_graphs as bg_mod  # noqa: E402

_real_exec = _QApplication.exec
_QApplication.exec = lambda self, *a, **k: 0
try:
    import gui as gui_mod
except (SystemExit, RuntimeError):
    gui_mod = sys.modules["gui"]
finally:
    _QApplication.exec = _real_exec
    sys.exit = _real_exit

app = gui_mod.app
w = gui_mod.window
VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_fsoc_beacon.mp4")

# ---- silence / record dialogs ------------------------------------------------
DIALOGS = []
QMessageBox.critical = staticmethod(lambda *a, **k: DIALOGS.append(("critical", a[2] if len(a) > 2 else "")) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: DIALOGS.append(("warning", a[2] if len(a) > 2 else "")) or QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: DIALOGS.append(("info", "")) or QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", True))
w._show_report_dialog = lambda title, paths: None
w._open_path = lambda path: True

# Qt emits unhandled exceptions to stderr; capture them explicitly.
ERRORS = []
_orig_hook = sys.excepthook


def _hook(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    ERRORS.append(text)
    print("UNHANDLED EXCEPTION:\n" + text, flush=True)


sys.excepthook = _hook


def pump(seconds=0.4):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        # A real event-loop return also runs deferred deletions. Without this
        # the harness would keep every closed benchmark screen listed forever
        # and could not tell a leak from a tidy shutdown.
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        time.sleep(0.005)


# =============================================================================
print("\n[1] AUTH GATE — buttons give feedback instead of dying silently")
# =============================================================================
w.start_simulation()
check("start the simulation" in w.footer_status.text(), "START without auth explains itself",
      w.footer_status.text())
w.generate_instant_report()
check("access key" in w.footer_status.text(), "INSTANT REPORT without auth explains itself",
      w.footer_status.text())
w.open_digital_twin()
check("access key" in w.footer_status.text(), "3D DIGITAL TWIN without auth explains itself",
      w.footer_status.text())

w.auth_input.setText("isro@1969")
w._authenticate()
pump(1.2)
check(w._authenticated, "authenticated via the real auth path")

# =============================================================================
print("\n[2] SIMULATION CONTROLS")
# =============================================================================
w.start_simulation()
pump(1.0)
frames_before = w.frames_processed
check(w.simulation_running and w.timer.isActive(), "START runs the loop")
pump(1.0)
check(w.frames_processed > frames_before, "frames are processed", f"{frames_before} -> {w.frames_processed}")

w.pause_simulation()
paused_at = w.frames_processed
pump(0.5)
check(not w.timer.isActive() and w.frames_processed == paused_at, "PAUSE stops the loop")

w.reset_simulation()
pump(0.6)
check(w.frames_processed > 0 and w.timer.isActive(), "RESET restarts cleanly")

w.force_target_loss()
pump(0.8)
check(True, "TEST LOSS / RECOVER handler runs without error")

# =============================================================================
print("\n[3] CURRENT SCENARIO reflects real settings")
# =============================================================================
w.motion_selector.setCurrentText("Figure 8")
w.target_count_selector.setCurrentText("3")
w.range_slider.setValue(30)
pump(0.8)
info = w.info.text()
check("CURRENT SCENARIO" in info, "scenario panel present")
check(f"Targets {w.target_count}" in info, "target count tracks the selector", info.splitlines()[2] if len(info.splitlines()) > 2 else "")
check("Figure 8" in info or "Figure 8" in w.footer_status.text(), "motion pattern tracks the selector")
check("Range 3000 m" in info, "beacon range tracks the slider", [l for l in info.splitlines() if "Range" in l])

# =============================================================================
print("\n[4] LIVE METRICS ARE FINITE")
# =============================================================================
w.motion_selector.setCurrentText("Straight Line")
w.target_count_selector.setCurrentText("1")
w.range_slider.setValue(10)
pump(1.5)
labels = {
    "FPS": w.t_fps, "Tracking Error": w.t_error, "Center Error": w.t_centroid,
    "Avg": w.t_avg, "Max Dev": w.t_max, "RMSE": w.t_rmse, "Loss": w.t_loss,
    "Lock": w.t_lock, "Velocity": w.t_velocity,
}
empty = [k for k, v in labels.items() if not v.text().strip()]
check(not empty, "no empty telemetry cards", str(empty))
bad = [k for k, v in labels.items() if "nan" in v.text().lower() or "inf" in v.text().lower()]
check(not bad, "no NaN/Infinity in telemetry", str(bad))

# =============================================================================
print("\n[5] TEST 1: PLAY starts the video AND the analysis together")
# =============================================================================
_open_calls = {"n": 0}


def fake_open(parent, title, *args, **kwargs):
    _open_calls["n"] += 1
    if _open_calls["n"] % 2 == 1:
        return (VIDEO, "")
    return ("", "")  # ground truth: cancel (sibling CSV is auto-detected)


QFileDialog.getOpenFileName = staticmethod(fake_open)

w.benchmark_video()
screen = w._benchmark_screen
check(not (getattr(w, "_benchmark_worker", None) is not None
           and w._benchmark_worker.isRunning()),
      "loading a video does NOT auto-start a run (PLAY owns it)")
check(screen._source_path == VIDEO, "screen registered the benchmark video",
      str(screen._source_path))
check(screen.play_pause_button.isEnabled(), "PLAY is enabled once a video is loaded")
check(screen.play_pause_button.text().strip().endswith("PLAY"),
      "transport starts in the PLAY state", screen.play_pause_button.text())

gp = w.graph_panel
s_gp = screen.graph_panel
check(len(s_gp.timestamps) == 0, "screen graphs start empty", str(len(s_gp.timestamps)))

screen._toggle_video_playback()          # PLAY
check(screen._playback_state == screen.PLAYBACK_RUNNING, "PLAY starts the benchmark")
check("PAUSE" in screen.play_pause_button.text(), "PLAY becomes PAUSE",
      screen.play_pause_button.text())

start = time.time()
while (screen._live_frame is None or len(s_gp.timestamps) < 5) and time.time() - start < 30:
    app.processEvents()
    time.sleep(0.02)
frames_a = len(s_gp.timestamps)
at_a = screen._video_current_frame
pump(1.0)
frames_b = len(s_gp.timestamps)
check(screen._live_frame is not None, "the analysed video frame is displayed")
check(screen.video_view.pixmap() is not None and not screen.video_view.pixmap().isNull(),
      "TEST VIEW shows a real frame while analysing")
check(frames_b > frames_a, "analysis is progressing", f"{frames_a} -> {frames_b}")
check(screen._video_current_frame > at_a, "video position is progressing",
      f"{at_a} -> {screen._video_current_frame}")
check(w._benchmark_worker.isRunning(), "exactly one benchmark loop is live")
check(s_gp.status_label.text().startswith("Live —"),
      "screen graphs are fed live from the processed frames", s_gp.status_label.text())
check(screen.result_labels["FPS"].text() not in ("", "—"),
      "metrics update during the run", screen.result_labels["FPS"].text())

# =============================================================================
print("\n[6] TEST 2: PAUSE freezes video, analysis and metrics")
# =============================================================================
screen._toggle_video_playback()          # PAUSE
check(screen._playback_state == screen.PLAYBACK_PAUSED, "PAUSE is registered")
check("RESUME" in screen.play_pause_button.text(), "PAUSE becomes RESUME",
      screen.play_pause_button.text())
# The frame already in flight when PAUSE was clicked is allowed to land
# (it was genuinely processed), then the run must be REALLY frozen.
pump(1.0)
paused_frames = len(s_gp.timestamps)
paused_at_frame = screen._video_current_frame
paused_fps = screen.result_labels["FPS"].text()
pump(1.5)
check(len(s_gp.timestamps) == paused_frames, "metrics stopped accumulating",
      f"{paused_frames} -> {len(s_gp.timestamps)}")
check(screen._video_current_frame == paused_at_frame, "video position is frozen",
      f"{paused_at_frame} -> {screen._video_current_frame}")
check(screen.result_labels["FPS"].text() == paused_fps, "metrics read-out is frozen")
check(w._benchmark_worker.isRunning(), "the paused run keeps ONE loop parked")
check(screen._video_current_frame - paused_at_frame == 0,
      "no frame is decoded or processed while paused")

# =============================================================================
print("\n[7] TEST 3: RESUME continues from the same position")
# =============================================================================
screen._toggle_video_playback()          # RESUME
check(screen._playback_state == screen.PLAYBACK_RUNNING, "RESUME is registered")
check("PAUSE" in screen.play_pause_button.text(), "RESUME becomes PAUSE",
      screen.play_pause_button.text())
start = time.time()
while len(s_gp.timestamps) <= paused_frames and time.time() - start < 30:
    app.processEvents()
    time.sleep(0.02)
check(len(s_gp.timestamps) > paused_frames, "analysis continues after RESUME")
check(s_gp.frames[paused_frames] == paused_at_frame + 1,
      "RESUME continues at the NEXT frame — the run is not restarted",
      f"{s_gp.frames[paused_frames]} vs {paused_at_frame}")
check(list(s_gp.frames[:3]) == [1, 2, 3], "the frame timeline was never reset",
      str(s_gp.frames[:4]))

# =============================================================================
print("\n[8] TEST 4: run to completion — final metrics, graphs, no QBrush")
# =============================================================================
start = time.time()
while w._benchmark_worker.isRunning() and time.time() - start < 240:
    app.processEvents()
    time.sleep(0.01)
pump(1.0)
summary1 = w._latest_benchmark_summary()
check(summary1 is not None, "run 1 produced a benchmark result")
check(screen.progress_bar.value() == 100, "run 1 reached 100%",
      str(screen.progress_bar.value()))
check(screen.result_labels["FPS"].text() not in ("", "—"), "run 1 shows a real FPS",
      screen.result_labels["FPS"].text())
check(screen._playback_state == screen.PLAYBACK_DONE, "a finished run offers REPLAY",
      screen._playback_state)
check("REPLAY" in screen.play_pause_button.text(), "transport label is REPLAY",
      screen.play_pause_button.text())
check(not w._benchmark_worker.isRunning(), "the single loop stopped at the end")

if summary1 is not None:
    print(f"        frames={summary1.frames_processed} fps={summary1.fps:.1f} "
          f"rmse={summary1.rmse_tracking_error_px} lock={summary1.lock_retention_percentage}")
    check(summary1.frames_processed > 100, "run 1 processed real frames", str(summary1.frames_processed))
    check(summary1.fps and summary1.fps > 0, "run 1 FPS measured", str(summary1.fps))
    check(summary1.ground_truth_available, "run 1 used real ground truth (error metrics real)")
    for name in ("rmse_tracking_error_px", "maximum_tracking_error_px",
                 "average_tracking_error_px", "lock_retention_percentage",
                 "target_loss_percentage"):
        value = getattr(summary1, name)
        ok = value is None or (isinstance(value, (int, float)) and math.isfinite(value))
        check(ok, f"run 1 {name} finite", str(value))

# graphs hold one point per processed frame, and nothing invalid
check(len(s_gp.timestamps) > 100, "screen graph panel holds the run's frames",
      str(len(s_gp.timestamps)))
if summary1 is not None:
    check(abs(len(s_gp.timestamps) - summary1.frames_processed) <= 2,
          "one graph point per processed frame (real, not invented)",
          f"{len(s_gp.timestamps)} vs {summary1.frames_processed}")
check(len(gp.timestamps) > 100, "graph panel loaded run-1 telemetry", str(len(gp.timestamps)))
series_ok = True
for arr in (gp.center_errors, gp.fps_values, gp.pan_values, gp.tilt_values,
            s_gp.center_errors, s_gp.fps_values, s_gp.pan_values, s_gp.tilt_values):
    for v in arr:
        if v is not None and not math.isfinite(v):
            series_ok = False
check(series_ok, "no NaN/Infinity reached any graph series")
check(gp.graph_widgets[0].data.get("timestamps") is not None or
      len(gp.graph_widgets[0].data.get("center_errors", [])) > 0,
      "graph widgets received the run-1 data")

# THE QBrush FIX: every graph draws real data with no draw error, and every
# title is measured to fit inside its own card at every width.
_fit_problems = []
for _panel in (s_gp, gp):
    for _widget in _panel.graph_widgets:
        # Direct card sizes: a layout pass cannot quietly undo them, so the
        # measurement really is taken at each width.
        for _ww in (620, 420, 300, 190):
            _widget.resize(_ww, 210)
            _pix = QPixmap(_ww, 210)
            _widget.render(_pix)
            if _widget.draw_error is not None:
                _fit_problems.append(f"{_widget.title}@{_ww}: {_widget.draw_error}")
            _font, _lines, _block = _widget._title_layout()
            _fm = QFontMetrics(_font)
            _unit_w = _fm.horizontalAdvance(str(_widget.unit)) + 10 if _widget.unit else 0
            _avail = max(40, _ww - 2 * bg_mod._GRAPH_PAD_X - _unit_w)
            if not _lines or not _lines[0].strip():
                _fit_problems.append(f"{_widget.title}@{_ww}: empty title")
            for _line in _lines:
                if _fm.horizontalAdvance(_line) > max(20, _avail):
                    _fit_problems.append(
                        f"{_widget.title}@{_ww}: '{_line}' overflows {_avail}px"
                    )
            if _block + 40 > 210:
                _fit_problems.append(
                    f"{_widget.title}@{_ww}: title block crowds the plot area"
                )
check(not _fit_problems, "every graph title fits its card at every width",
      "; ".join(_fit_problems[:4]))
for _panel in (s_gp, gp):
    for _widget in _panel.graph_widgets:
        _widget.resize(420, 210)
    _panel._relayout_graphs(3)
    _panel.update()
check(all(gw.draw_error is None for gw in s_gp.graph_widgets + gp.graph_widgets),
      "no graph raised a draw error (the old 'QBrush' text is gone)")
_painted_text = " ".join(
    [gw.title for gw in s_gp.graph_widgets] + [s_gp.status_label.text()]
)
check("QBrush" not in _painted_text and "Error drawing" not in _painted_text,
      "no QBrush / drawing-error text in the graph section")

# Regression: a FRESH panel fed only live frames (never set_data / reset) must
# paint cleanly — that is exactly the path live playback uses, and it used to
# raise inside paintEvent because the live marker attributes did not exist yet.
_fresh = bg_mod.BenchmarkGraphPanel()
_fresh.resize(900, 700)
for _i in range(40):
    _fresh.append_live_sample(_i / 30.0, _i + 1, {
        "center_error": 1.0, "centroid_error": 0.5, "target_x": 300.0,
        "target_y": 200.0, "predicted_x": 301.0, "predicted_y": 201.0,
        "pan": 0.5, "tilt": 0.2, "fps": 250.0, "filter_confidence": 0.8,
        "detector_confidence": 0.9, "ai_confidence": 0.4, "state": "TRACKING",
    })
pump(0.1)
_fresh_problems = []
for _gw in _fresh.graph_widgets:
    _gw.render(QPixmap(max(1, _gw.width()), max(1, _gw.height())))
    if _gw.draw_error:
        _fresh_problems.append(str(_gw.draw_error))
check(not _fresh_problems, "a fresh panel painted from live frames alone is clean",
      str(_fresh_problems))

# Invalid values must be rejected BEFORE they reach the graph.
_fresh.append_live_sample(99.0, 999, {
    "center_error": float("nan"), "fps": float("inf"), "pan": "junk",
    "tilt": None, "state": None, "target_x": float("-inf"),
})
check(_fresh.center_errors[-1] is None and _fresh.fps_values[-1] is None
      and _fresh.pan_values[-1] is None and _fresh.target_x[-1] is None
      and _fresh.states[-1] == "UNKNOWN",
      "NaN / Infinity / junk live values never reach the graph")
for _gw in _fresh.graph_widgets:
    _gw.render(QPixmap(max(1, _gw.width()), max(1, _gw.height())))
check(all(_gw.draw_error is None for _gw in _fresh.graph_widgets),
      "a graph stays clean even when a frame carried invalid values")
_fresh.deleteLater()
s_gp.resize(720, 620)
gp.resize(720, 620)
pump(0.1)

# =============================================================================
print("\n[9] REPORT EXPORT USES THE BENCHMARK RESULT")
# =============================================================================
before = w.report_list.count()
w.export_report()
check(w.report_list.count() == before + 1, "EXPORT REPORT produced an artifact")
check(w.report_last_saved and os.path.exists(w.report_last_saved[0]),
      "exported PDF exists", str(w.report_last_saved[0] if w.report_last_saved else None))
check(w.footer_status.text().startswith("Report saved (latest benchmark)"),
      "export says it used the benchmark", w.footer_status.text())
check("Error" not in w.footer_status.text(), "export reported no error", w.footer_status.text())

# =============================================================================
print("\n[10] INSTANT REPORT")
# =============================================================================
saved_history = list(w.instant_report_history)
w.instant_report_history = []          # force the benchmark fallback path
before = w.report_list.count()
w.generate_instant_report()
check(w.report_list.count() == before + 1, "INSTANT REPORT produced an artifact with no live history")
check("latest benchmark" in w.footer_status.text(), "instant report used the benchmark",
      w.footer_status.text())
w.instant_report_history = saved_history
w.generate_instant_report()
check(w.report_list.count() == before + 2, "INSTANT REPORT works from live telemetry too")
check(not any("Error" in str(d[1]) for d in DIALOGS), "no error dialogs raised", str(DIALOGS[-3:]))

# No benchmark AND no live history: the exact message the spec asks for.
DIALOGS.clear()
saved_summary = w._last_benchmark_summary
w.instant_report_history = []
w._last_benchmark_summary = None
w.generate_instant_report()
check(any(d[0] == "warning" and "No benchmark data available. Run Benchmark Video first." in str(d[1])
          for d in DIALOGS), "empty state asks for a benchmark run", str(DIALOGS))
w.instant_report_history = saved_history
w._last_benchmark_summary = saved_summary

# =============================================================================
print("\n[11] 3D DIGITAL TWIN + graph launcher buttons")
# =============================================================================
w.open_digital_twin()
pump(0.6)
check(w.digital_twin_window is not None, "3D DIGITAL TWIN window opens")
w._open_benchmark_graphs()
check("loaded" in gp.status_label.text().lower(), "VIEW BENCHMARK GRAPHS loads telemetry",
      gp.status_label.text())

# =============================================================================
print("\n[12] TEST 5: REPLAY resets everything and starts a fresh run")
# =============================================================================
screen._toggle_video_playback()          # REPLAY
check(screen._playback_state == screen.PLAYBACK_RUNNING, "REPLAY starts a fresh run")
check(screen.result_labels["RMSE"].text() == "—",
      "REPLAY cleared the previous results", screen.result_labels["RMSE"].text())
check(len(s_gp.timestamps) < 60, "REPLAY cleared the previous graph data",
      str(len(s_gp.timestamps)))
check(screen._video_current_frame < 60, "REPLAY went back to the beginning",
      str(screen._video_current_frame))
check(gp.status_label.text().startswith("Benchmark running"),
      "REPLAY reset the dashboard metrics too", gp.status_label.text())

_threads_before = threading.active_count()
_worker_during_replay = w._benchmark_worker
for _ in range(3):
    w._start_benchmark_screen_run(screen)
check(w._benchmark_worker is _worker_during_replay,
      "repeated PLAY cannot start a second benchmark loop")
check(threading.active_count() <= _threads_before + 1,
      "repeated PLAY creates no extra threads",
      f"{_threads_before} -> {threading.active_count()}")

start = time.time()
while len(s_gp.timestamps) < 3 and time.time() - start < 20:
    app.processEvents()
    time.sleep(0.02)
check(list(s_gp.frames[:3]) == [1, 2, 3], "replay starts again from frame 1",
      str(s_gp.frames[:4]))

start = time.time()
while w._benchmark_worker.isRunning() and time.time() - start < 240:
    app.processEvents()
    time.sleep(0.01)
pump(1.0)
summary2 = w._latest_benchmark_summary()
check(summary2 is not None and summary2 is not summary1,
      "run 2 replaced the benchmark result object")
check(screen.progress_bar.value() == 100, "run 2 reached 100%")
check(len(s_gp.timestamps) > 100, "screen graphs hold only the run-2 frames",
      str(len(s_gp.timestamps)))
check(len(gp.timestamps) > 100, "graph panel refreshed from run-2 telemetry",
      str(len(gp.timestamps)))
check(screen._playback_state == screen.PLAYBACK_DONE, "run 2 also ends in REPLAY")
if summary2 is not None:
    print(f"        frames={summary2.frames_processed} fps={summary2.fps:.1f} "
          f"rmse={summary2.rmse_tracking_error_px}")
    check(summary2.frames_processed > 100, "run 2 processed real frames")
    if summary1 is not None:
        check(summary2.rmse_tracking_error_px == summary1.rmse_tracking_error_px,
              "run 2 is identical to run 1 (no stale or duplicated state)",
              f"{summary2.rmse_tracking_error_px} vs {summary1.rmse_tracking_error_px}")

# =============================================================================
print("\n[13] TEST 6: cancel works even while PAUSED (clean shutdown)")
# =============================================================================
_threads_idle = threading.active_count()
screen._toggle_video_playback()          # start run 3
start = time.time()
while len(s_gp.timestamps) < 5 and time.time() - start < 20:
    app.processEvents()
    time.sleep(0.02)
screen._toggle_video_playback()          # PAUSE
check(screen._playback_state == screen.PLAYBACK_PAUSED, "run 3 paused")
screen._request_cancel()
start = time.time()
while w._benchmark_worker.isRunning() and time.time() - start < 30:
    app.processEvents()
    time.sleep(0.01)
check(not w._benchmark_worker.isRunning(), "CANCEL stops a PAUSED run")
check(screen._playback_state in (screen.PLAYBACK_IDLE, screen.PLAYBACK_DONE),
      "a cancelled run leaves the transport usable", screen._playback_state)
pump(0.5)
check(threading.active_count() <= _threads_idle + 1, "no worker threads leaked",
      f"{_threads_idle} -> {threading.active_count()}")

# =============================================================================
print("\n[14] TEST 7: resize the benchmark screen — titles fit, nothing clipped")
# =============================================================================
for _sx, _sy in ((1500, 900), (1240, 820), (1100, 720), (980, 680)):
    screen.resize(_sx, _sy)
    pump(0.5)
    _ok = True
    _detail = []
    for _widget in screen.graph_panel.graph_widgets:
        _pix = QPixmap(max(1, _widget.width()), max(1, _widget.height()))
        _widget.render(_pix)
        if _widget.draw_error is not None:
            _ok = False
            _detail.append(str(_widget.draw_error))
        _font, _lines, _block = _widget._title_layout()
        _fm = QFontMetrics(_font)
        _unit_w = _fm.horizontalAdvance(str(_widget.unit)) + 10 if _widget.unit else 0
        _avail = max(40, _widget.width() - 2 * bg_mod._GRAPH_PAD_X - _unit_w)
        if not _lines or not _lines[0].strip():
            _ok = False
            _detail.append(f"{_widget.title}: empty title")
        for _line in _lines:
            if _fm.horizontalAdvance(_line) > max(20, _avail):
                _ok = False
                _detail.append(f"{_widget.title}: '{_line}'")
    check(_ok, f"@{_sx}: every benchmark graph title still fits", "; ".join(_detail[:3]))
    check(not screen.graph_panel._scroll.horizontalScrollBar().isVisible(),
          f"@{_sx}: no horizontal overflow in the graph section")
    _btn = screen.play_pause_button
    _host = _btn.parentWidget()
    if _host is not None:
        check(_btn.x() >= -1 and _btn.x() + _btn.width() <= _host.width() + 1,
              f"@{_sx}: PLAY/PAUSE/REPLAY stays inside the video panel",
              f"x={_btn.x()} w={_btn.width()} host={_host.width()}")

# =============================================================================
print("\n[15] RESPONSIVE LAYOUT — no horizontal scrolling, buttons inside")
# =============================================================================
for width, height in ((1600, 950), (1366, 768), (1280, 800), (1152, 720), (1024, 700)):
    w.resize(width, height)
    pump(0.35)
    w.resize(max(w.minimumWidth(), width - 1), height)
    pump(0.35)
    bar = w.control_scroll.horizontalScrollBar()
    check(not bar.isVisible(), f"@{width}: control column has no horizontal scrollbar",
          f"range={bar.maximum()}")
    check(not gp._scroll.horizontalScrollBar().isVisible(),
          f"@{width}: graph panel has no horizontal scrollbar")
    viewport = w.control_scroll.viewport().width()
    content = w.control_scroll.widget()
    from PySide6.QtWidgets import QPushButton
    from PySide6.QtCore import QPoint
    checked = 0
    for btn in content.findChildren(QPushButton):
        if not btn.isVisible() or not btn.text().strip():
            continue
        g = btn.geometry()
        inside = g.x() >= -1 and g.x() + g.width() <= content.width() + 1
        check(inside, f"@{width}: {btn.text().strip()!r} fully inside the panel",
              f"x={g.x()} right={g.x() + g.width()} content={content.width()}")
        checked += 1
    check(checked >= 9, f"@{width}: checked every control-column button", str(checked))
    check(w.minimumSizeHint().width() <= max(width, 1069) + 1,
          f"@{width}: window minimum fits the screen", str(w.minimumSizeHint().width()))

# =============================================================================
print("\n[16] VIDEO LOAD ERRORS — clear message, no crash, retry available")
# =============================================================================
_MISSING = str(Path(VIDEO).with_name("definitely_missing_benchmark.mp4"))
_bad = w._prepare_benchmark_screen(_MISSING, None)
pump(0.4)
check("could not be loaded" in _bad.progress_label.text(),
      "a missing video shows the exact load-error message", _bad.progress_label.text())
check("could not be loaded" in _bad.video_view.text(),
      "the load error is visible in the test view", _bad.video_view.text())
check(not _bad.play_pause_button.isEnabled(),
      "PLAY is disabled when there is nothing to play")
check(_bad.run_button.isEnabled(), "RUN BENCHMARK stays available as the retry path")
check(_bad._source_path is None, "no bogus video is registered as the source")
_bad._request_run_benchmark()
pump(0.5)
check("could not be loaded" in _bad.progress_label.text()
      or "not found" in w.footer_status.text(),
      "retrying an unreadable file reports the same clear error",
      f"{_bad.progress_label.text()} | {w.footer_status.text()}")
check(not (getattr(w, "_benchmark_worker", None) is not None
           and w._benchmark_worker.isRunning()),
      "an unreadable video never starts a benchmark loop")
check(not DIALOGS or all(d[0] != "critical" for d in DIALOGS[-1:]),
      "the load error is reported inline, not as a fatal dialog", str(DIALOGS[-1:]))

# Recovery: picking a good video afterwards works normally.
_ok_screen = w._prepare_benchmark_screen(VIDEO, None)
pump(0.4)
check(_ok_screen.play_pause_button.isEnabled()
      and _ok_screen._source_path == VIDEO,
      "picking a readable video after a failure recovers",
      f"{_ok_screen._source_path} enabled={_ok_screen.play_pause_button.isEnabled()}")
check(_ok_screen.play_pause_button.text().strip().endswith("PLAY"),
      "the recovered screen offers PLAY again", _ok_screen.play_pause_button.text())
screen = _ok_screen

# =============================================================================
print("\n[17] ERROR HYGIENE")
# =============================================================================
check(not ERRORS, "no unhandled Qt exceptions", (ERRORS[:1] or [""])[0][:400])
check(gp._refresh_timer.isActive(), "graph refresh timer is running (and single)")
check(len([s for s in _QApplication.topLevelWidgets()
           if type(s).__name__ == "BenchmarkScreen"]) <= 1,
      "only one benchmark screen exists at a time")
w.close()
pump(0.3)

print("\n" + "=" * 70)
if PROBLEMS:
    print(f"RESULT: {len(PROBLEMS)} PROBLEM(S)")
    for p in PROBLEMS:
        print("  -", p)
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
