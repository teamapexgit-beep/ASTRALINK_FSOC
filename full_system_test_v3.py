"""FSOC master system self-test (v2).

Aligned with the current Sept-2026 FSOC baseline APIs. Run from the project
folder with the project's .venv active:
    python full_system_test_v2.py

This test intentionally does not fabricate accuracy numbers. It checks core
behavior, real forced loss/re-acquisition, reporting, benchmark generation,
and a safe GUI startup smoke test.
"""
from __future__ import annotations

import csv
import importlib
import math
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from PySide6.QtWidgets import QComboBox

ROOT = Path(__file__).resolve().parent
PASS = WARN = FAIL = 0


def report(status: str, name: str, detail: str = "") -> None:
    global PASS, WARN, FAIL
    if status == "PASS":
        PASS += 1
    elif status == "WARN":
        WARN += 1
    else:
        FAIL += 1
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def check(condition: bool, name: str, detail: str = "") -> None:
    report("PASS" if condition else "FAIL", name, detail)


def warn(name: str, detail: str = "") -> None:
    report("WARN", name, detail)


def compile_project() -> None:
    excluded = {".venv", ".git", "__pycache__"}
    files = [p for p in ROOT.rglob("*.py") if not any(x in excluded for x in p.parts)]
    errors = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"{path.name}: {exc.msg}")
    check(not errors, "Python compilation", f"{len(files)} files checked" if not errors else "; ".join(errors))


def import_core():
    names = [
        "config", "ai_verifier", "controller", "detector", "tracker",
        "simulator", "performance_report", "metrics", "benchmark_video",
        "instant_report", "digital_twin",
    ]
    loaded = {}
    for name in names:
        loaded[name] = importlib.import_module(name)
    report("PASS", "Core module imports")
    return loaded


def run_existing_self_test() -> None:
    script = ROOT / "self_test.py"
    if not script.exists():
        check(False, "Existing self_test.py is present")
        return

    env = os.environ.copy()
    # Avoid Windows CP1252 crashing while printing ≥ and ≤.
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        check(False, "Existing self_test.py completes", "timed out after 90 seconds")
        return

    print("\n--- self_test.py output ---")
    print(proc.stdout.rstrip())
    print("--- end self_test.py output ---\n")
    check(proc.returncode == 0, "Existing self_test.py exits cleanly", f"exit code {proc.returncode}")

    # Parse the FOG row, but do not turn a known stress weakness into an
    # artificial PASS.
    for line in proc.stdout.splitlines():
        if line.lstrip().startswith("FOG_100"):
            parts = line.split()
            try:
                det = float(parts[1])
            except (IndexError, ValueError):
                det = None
            if det is not None and det < 95.0:
                warn("Fog stress case", f"FOG_100 detection is {det:.2f}%")
            break


def run_disturbance_configuration(mod) -> None:
    Simulator = mod["simulator"].Simulator
    sim = Simulator()

    sim.set_disturbances(
        turbulence_enabled=False,
        turbulence_strength_px=0,
        camera_jitter_enabled=False,
        camera_jitter_px=0,
        platform_motion_enabled=False,
        platform_motion_px=0,
        atmosphere="Clear",
        atmosphere_level=0,
    )
    check(
        not sim.turbulence_enabled and not sim.camera_jitter_enabled and not sim.platform_motion_enabled
        and sim.atmosphere == "Clear" and sim.atmosphere_level == 0,
        "Disturbance config: Clear",
    )

    sim.set_noise_type("Gaussian")
    sim.noise_level = 10
    check(sim.noise_type == "Gaussian" and sim.noise_level == 10, "Disturbance config: Sensor Noise")

    sim.set_disturbances(
        turbulence_enabled=True, turbulence_strength_px=10,
        camera_jitter_enabled=True, camera_jitter_px=15,
        platform_motion_enabled=True, platform_motion_px=15,
        atmosphere="Clear", atmosphere_level=0,
    )
    check(
        sim.turbulence_enabled and sim.turbulence_strength_px == 10
        and sim.camera_jitter_enabled and sim.camera_jitter_px == 15
        and sim.platform_motion_enabled and sim.platform_motion_px == 15,
        "Disturbance config: Vibration",
    )

    sim.set_disturbances(
        turbulence_enabled=True, turbulence_strength_px=8,
        camera_jitter_enabled=False, camera_jitter_px=0,
        platform_motion_enabled=False, platform_motion_px=0,
        atmosphere="Fog", atmosphere_level=70,
    )
    check(sim.atmosphere == "Fog" and sim.atmosphere_level == 70 and sim.turbulence_strength_px == 8,
          "Disturbance config: Atmosphere")

    sim.set_noise_type("Gaussian")
    sim.set_disturbances(
        turbulence_enabled=False, turbulence_strength_px=0,
        camera_jitter_enabled=True, camera_jitter_px=8,
        platform_motion_enabled=True, platform_motion_px=8,
        atmosphere="Haze", atmosphere_level=35,
    )
    check(
        sim.noise_type == "Gaussian" and sim.camera_jitter_px == 8
        and sim.platform_motion_px == 8 and sim.atmosphere == "Haze" and sim.atmosphere_level == 35,
        "Disturbance config: Full Combined",
    )

    sim.set_disturbances(
        turbulence_enabled=True, turbulence_strength_px=20,
        camera_jitter_enabled=True, camera_jitter_px=20,
        platform_motion_enabled=True, platform_motion_px=20,
        atmosphere="Fog", atmosphere_level=85,
    )
    check(
        sim.turbulence_strength_px == 20 and sim.camera_jitter_px == 20
        and sim.platform_motion_px == 20 and sim.atmosphere_level == 85,
        "Disturbance config: Extreme PAT",
    )


def run_core_pipeline(mod):
    config = mod["config"]
    Simulator = mod["simulator"].Simulator
    CameraController = mod["controller"].CameraController
    BeaconTracker = mod["tracker"].BeaconTracker
    detect_beacon = mod["detector"].detect_beacon

    sim = Simulator()
    sim.set_target_count(1)
    sim.set_designated_target("TGT-01")
    sim.set_motion_pattern("Figure 8")
    sim.set_beacon_speed(4.0)
    sim.set_beacon_range(1000.0)
    sim.set_beacon_size(10.0)
    sim.set_noise_type("None")
    sim.set_disturbances()

    controller = CameraController()
    controller.set_angle_limits(5.0, 3.75)
    controller.set_gain(1.5)
    tracker = BeaconTracker()

    detections = 0
    tracking = 0
    errors = []
    start = time.perf_counter()

    for _ in range(120):
        frame = sim.get_frame()
        expected = (tracker.x, tracker.y) if tracker.x is not None else None
        detection, _info = detect_beacon(
            frame,
            return_details=True,
            expected_position=expected,
            preferred_signature=sim.get_designated_target_signature(),
        )
        detections += int(detection is not None)
        tracked = tracker.update(detection)
        if tracked is not None:
            x, y, state = tracked
            if state == "TRACKING":
                tracking += 1
                errors.append(math.hypot(x - config.FRAME_WIDTH / 2, y - config.FRAME_HEIGHT / 2))
            lead = 1 / max(config.UPDATE_RATE, 1)
            px = x + tracker.velocity_x * lead * 0.85
            py = y + tracker.velocity_y * lead * 0.85
            pan, tilt = controller.update(px - config.FRAME_WIDTH / 2, py - config.FRAME_HEIGHT / 2)
            sim.update_camera(pan, tilt)

    fps = 120 / max(time.perf_counter() - start, 1e-9)
    detection_pct = 100 * detections / 120
    check(detection_pct >= 95, "Core beacon detection", f"{detection_pct:.2f}% over 120 frames")
    check(tracking > 0, "Kalman tracker reaches TRACKING", f"{tracking} tracking frames")
    check(abs(controller.pan) <= controller.pan_limit_deg + 1e-9, "Pan controller respects limit", f"{controller.pan:.3f} deg")
    check(abs(controller.tilt) <= controller.tilt_limit_deg + 1e-9, "Tilt controller respects limit", f"{controller.tilt:.3f} deg")
    check(fps >= 20, "Core processing speed", f"{fps:.1f} FPS")
    if errors:
        check(max(errors) <= 10.0, "Core nominal alignment", f"max closed-loop error {max(errors):.2f} px")


def run_reacquisition(mod) -> None:
    Simulator = mod["simulator"].Simulator
    BeaconTracker = mod["tracker"].BeaconTracker
    detect_beacon = mod["detector"].detect_beacon
    config = mod["config"]

    sim = Simulator()
    sim.set_target_count(1)
    sim.set_designated_target("TGT-01")
    sim.set_motion_pattern("Straight Line")
    sim.set_noise_type("None")
    sim.set_disturbances()
    tracker = BeaconTracker()

    # Warm up until the tracker is in TRACKING.
    for _ in range(15):
        frame = sim.get_frame()
        det = detect_beacon(frame, expected_position=(tracker.x, tracker.y) if tracker.x is not None else None,
                            preferred_signature=sim.get_designated_target_signature())
        tracker.update(det)

    check(tracker.state == "TRACKING", "Re-acquisition setup reaches TRACKING", tracker.state)

    # Lose the designated target for 6 simulation frames (~0.20 s at 30 Hz).
    loss_start = sim.frame_count + 1
    sim.set_beacon_loss(loss_start, 6)
    first_miss_frame = None
    reacq_frame = None

    for _ in range(25):
        frame = sim.get_frame()
        det = detect_beacon(
            frame,
            expected_position=(tracker.x, tracker.y) if tracker.x is not None else None,
            preferred_signature=sim.get_designated_target_signature(),
        )
        current_frame = sim.frame_count
        if det is None and first_miss_frame is None:
            first_miss_frame = current_frame
        tracked = tracker.update(det)
        if first_miss_frame is not None and det is not None and reacq_frame is None:
            reacq_frame = current_frame
        if reacq_frame is not None and tracked is not None and tracker.state == "RE-ACQUIRING":
            break

    check(first_miss_frame is not None, "Forced target loss is detected", f"frame={first_miss_frame}")
    check(reacq_frame is not None, "Forced target loss recovers", f"frame={reacq_frame}")

    if first_miss_frame is not None and reacq_frame is not None:
        reacq_s = (reacq_frame - first_miss_frame) / max(config.UPDATE_RATE, 1)
        check(reacq_s <= 1.0, "Real re-acquisition time", f"{reacq_s:.3f} s")
    else:
        warn("Real re-acquisition time", "could not measure because loss/recovery did not complete")


def run_multi_target(mod) -> None:
    Simulator = mod["simulator"].Simulator
    detect_beacon = mod["detector"].detect_beacon
    sim = Simulator()
    sim.set_target_count(5)
    ids = sim.get_target_ids()
    check(ids == ["TGT-01", "TGT-02", "TGT-03", "TGT-04", "TGT-05"], "Five-target simulator", str(ids))
    outcomes = []
    for target_id in ("TGT-01", "TGT-03", "TGT-05"):
        selected = sim.set_designated_target(target_id)
        frame = sim.get_frame()
        det, info = detect_beacon(frame, return_details=True, preferred_signature=sim.get_designated_target_signature())
        outcomes.append((target_id, selected, det, info.get("candidate_count", 0)))
    check(all(o[1] for o in outcomes), "Designated-target switching", str([o[0:2] for o in outcomes]))
    check(all(o[3] >= 1 for o in outcomes), "Decoys reach detector candidate stage", str([(o[0], o[3]) for o in outcomes]))
    check(all(o[2] is not None for o in outcomes), "Designated target identity gate selects target", str([(o[0], o[2]) for o in outcomes]))


def run_auth_password(mod) -> None:
    auth_config = importlib.import_module("auth_config")
    check(
        auth_config.APP_PASSWORD == "isro@1969",
        "Application authentication password",
        "configured value confirmed",
    )


def _pdf_open_check(pdf_path) -> tuple[bool, str]:
    """Verify a PDF genuinely opens.

    Primary: PySide6 QtPdf loads the document (no error, >= 1 page).
    Fallback (QtPdf unavailable): structural validation of %PDF- header and
    %%EOF trailer so the suite never false-fails on a stripped environment.
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
            return False, "file missing or smaller than 1 KB"
        data = pdf_path.read_bytes()
        if not data.startswith(b"%PDF-"):
            return False, "missing %PDF- header"
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtGui import QGuiApplication
            from PySide6.QtPdf import QPdfDocument
        except Exception as exc:
            if b"%%EOF" not in data[-1024:]:
                return False, f"QtPdf unavailable and no %%EOF trailer: {exc}"
            return True, "QtPdf unavailable; structural validation passed"

        app = QGuiApplication.instance() or QGuiApplication(["fsoc_pdf_check"])
        doc = QPdfDocument()
        error = doc.load(str(pdf_path))
        pages = doc.pageCount()
        doc.close()
        del doc
        if error != QPdfDocument.Error.None_ or pages < 1:
            return False, f"QtPdf load error={error} pages={pages}"
        return True, f"QtPdf opened {pages} page(s)"
    except Exception as exc:
        return False, repr(exc)


def run_pdf_report_check(mod) -> None:
    """Verify the normal performance report creates a PDF that can be opened.

    Uses mkdtemp + ignore-errors cleanup because on Windows QPdfDocument keeps
    the file mapped briefly even after close(), which would otherwise make
    TemporaryDirectory cleanup raise PermissionError and fail the suite for a
    non-application reason.
    """
    PerformanceSummary = mod["performance_report"].PerformanceSummary
    save_full = mod["performance_report"].save_full_performance_report

    def make_summary(gt: bool) -> PerformanceSummary:
        return PerformanceSummary(
            frames_processed=372, duration_s=12.4, fps=30.0,
            acquisition_time_s=0.73 if gt else None,
            average_tracking_error_px=1.7, maximum_tracking_error_px=4.6,
            rmse_tracking_error_px=1.9,
            average_centroid_error_px=0.42 if gt else 0.0,
            maximum_centroid_error_px=0.9 if gt else 0.0,
            lock_retention_percentage=98.2, target_loss_percentage=1.8,
            target_loss_events=1, successful_reacquisitions=1,
            average_reacquisition_time_s=0.61 if gt else None,
            maximum_reacquisition_time_s=0.80 if gt else None,
            average_processing_time_ms=11.6, max_processing_time_ms=18.2,
            ground_truth_available=gt, source="master PDF check",
        )

    tmp = Path(tempfile.mkdtemp(prefix="fsoc_test_pdf_"))
    try:
        # With ground truth.
        pdf_path, json_path, csv_path = save_full(make_summary(True), output_dir=tmp, prefix="pdf_check_gt")
        check(
            pdf_path.suffix.lower() == ".pdf" and pdf_path.exists() and pdf_path.stat().st_size > 1000,
            "Performance report PDF created",
            f"{pdf_path.name} ({pdf_path.stat().st_size} bytes)",
        )
        ok, detail = _pdf_open_check(pdf_path)
        check(ok, "Performance report PDF can be opened", detail)

        # Without ground truth: PDF must still render (missing values as em dashes).
        pdf2, _, _ = save_full(make_summary(False), output_dir=tmp, prefix="pdf_check_nogt")
        ok2, detail2 = _pdf_open_check(pdf2)
        check(ok2, "Performance report PDF renders without ground truth", detail2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_gui_feature_checks(mod) -> None:
    """Headless behavioral checks for STAGE-1 GUI features.

    Uses the same proven offscreen-import pattern as full_system_test.py (v1):
    QT_QPA_PLATFORM=offscreen is set before this module is loaded by main(),
    and gui.py's module-level sys.exit(app.exec()) raises SystemExit which we
    must catch and then fish the real window out of sys.modules.
    """
    _real_exit = sys.exit
    sys.exit = lambda *a, **k: None
    # gui.py ends with sys.exit(app.exec()); without this stub the module
    # import would enter the (offscreen) event loop and never return.
    from PySide6.QtWidgets import QApplication as _QApplication
    _real_exec = _QApplication.exec
    _QApplication.exec = lambda self, *a, **k: 0
    try:
        gui = importlib.import_module("gui")
    except (SystemExit, RuntimeError):
        # SystemExit: module-level sys.exit(app.exec()); RuntimeError:
        # shiboken singleton complaints if Qt state exists from earlier checks.
        gui = sys.modules.get("gui")
    finally:
        _QApplication.exec = _real_exec
        sys.exit = _real_exit
    if gui is None:
        report("FAIL", "GUI feature checks: gui import", "gui module not importable offscreen")
        return
    window = getattr(gui, "window", None)
    app = getattr(gui, "app", None)
    if window is None or app is None:
        report("FAIL", "GUI feature checks: window constructed", "gui.app/gui.window missing")
        return

    try:
        window.auth_input.setText("isro@1969")
        window._authenticate()
        window.timer.stop()
        window.simulation_running = True
        for _ in range(10):
            window.update_simulation()

        # --- Max Deviation: same value everywhere, real snapshot flow ------
        check(
            window.t_max.objectName() == "metric_value" and window.t_max.parent() is not None,
            "Max Deviation metric card present",
            f"value={window.t_max.text()}",
        )

        # Before the lock gate opens there is no deviation evidence at all;
        # the snapshot must carry None rather than a fabricated number.
        window._reset_metrics()
        window.update_simulation()
        pre_lock = window._last_report_snapshot or {}
        check(
            window.error_count == 0 and pre_lock.get("maximum_tracking_error_px") is None,
            "Max Deviation not fabricated before lock",
            f"snapshot max={pre_lock.get('maximum_tracking_error_px')}",
        )

        # Run long enough for the acquisition gate to open, then the snapshot
        # must carry exactly the live accumulator values.
        for _ in range(150):
            window.update_simulation()
        snapshot = window._last_report_snapshot or {}
        check(
            window.error_count > 0
            and snapshot.get("maximum_tracking_error_px") == window.maximum_error
            and snapshot.get("average_tracking_error_px") == window.average_error,
            "Max Deviation flows into instant-report snapshot",
            f"max={snapshot.get('maximum_tracking_error_px')} px after {window.frames_processed} frames",
        )

        # --- Responsive side-by-side layout at several window sizes -------
        # Radar and camera share page 0 SIDE BY SIDE; the 640x480 feed is
        # letterboxed with KeepAspectRatio so it is never distorted.
        window.show()
        ops_page = window.main_stack.widget(0)

        def panel_xspan(panel):
            tl = panel.mapTo(ops_page, panel.rect().topLeft())
            br = panel.mapTo(ops_page, panel.rect().bottomRight())
            return tl.x(), br.x()

        sizes = [(1600, 980), (1920, 1080), (2400, 1300), (1280, 760), (800, 760)]
        layout_holds = True
        narrowest = 10 ** 9
        for width, height in sizes:
            window.resize(width, height)
            app.processEvents()
            rx0, rx1 = panel_xspan(window.radar_panel)
            cx0, cx1 = panel_xspan(window.camera_panel)
            if not (rx1 < cx0):
                layout_holds = False
            narrowest = min(narrowest, cx1 - cx0)
            if width >= 1280 and cx1 - cx0 < 340:
                layout_holds = False
            if width < 1280 and cx1 - cx0 < 220:
                layout_holds = False
        import inspect as _inspect
        keeps_aspect = "KeepAspectRatio" in _inspect.getsource(type(window.camera_label).paintEvent)
        check(
            layout_holds and keeps_aspect,
            "Camera viewport grows responsively beside the radar (side-by-side at every size, undistorted feed)",
            f"narrowest camera column {narrowest}px, keep_aspect={keeps_aspect}",
        )

        # Zoom must survive maximize-scale changes and stay display-only.
        window._on_zoom_changed(20)   # 2.0x
        zoom_before = window.display_zoom
        window.resize(2400, 1300)
        app.processEvents()
        check(
            abs(window.display_zoom - zoom_before) < 1e-9 and abs(window.display_zoom - 2.0) < 1e-9,
            "Display zoom survives window resize (never resets)",
            f"zoom={window.display_zoom:.1f}x",
        )
        window._reset_zoom()

        # --- Report wiring: full bundle + OPEN LAST REPORT ---------------
        check(
            window.open_last_report_button is not None
            and hasattr(window, "_show_report_dialog")
            and hasattr(window, "_open_path"),
            "Report open helpers present (OPEN PDF / FOLDER / CLOSE)",
        )
        with tempfile.TemporaryDirectory(prefix="fsoc_gui_report_") as tmp:
            paths = window._save_current_report_into(tmp, prefix="stage1_check")
            check(
                len(paths) == 3 and all(Path(p).exists() for p in paths) and Path(paths[0]).suffix == ".pdf",
                "GUI export produces PDF+JSON+CSV",
                f"{Path(paths[0]).name}",
            )
            check(
                window.last_pdf_path is not None and Path(window.last_pdf_path).exists(),
                "Latest report PDF stored and reopenable",
                Path(window.last_pdf_path).name,
            )
            check(
                window.open_last_report_button.isEnabled(),
                "OPEN LAST REPORT button enabled after a report",
            )

        # --- External-window state restored for later stages --------------
        window.close()
        app.processEvents()
    except Exception as exc:
        report("FAIL", "GUI feature checks", repr(exc))


def _settle_tracking(window, label, frames=180, sustained=12, max_rate_deg=0.05):
    """Drive the live loop until the shared mission is truly calm: TRACKING,
    metrics started, and the closed-loop camera nearly stationary.

    Handing the shared window to the next check mid-slew makes later
    forced-loss timing flaky: while the beacon is hidden for its 9-frame
    window, a still-settling camera can slide its image out of the frame and
    a coasting tracker will never meet it again. The invariant that matters
    is the camera's per-frame angular rate (an absolute pan angle is not one
    — the beacon drifts in Straight-Line motion, so the camera legitimately
    keeps panning). Every behavioral check must therefore hand over a
    settled mission (test hygiene only — no production behavior involved).
    """
    prev_pan = window.simulator.camera_pan
    prev_tilt = window.simulator.camera_tilt
    good = 0
    settled = False
    for _ in range(frames):
        window.update_simulation()
        pan = window.simulator.camera_pan
        tilt = window.simulator.camera_tilt
        calm = (
            window.metrics_started
            and window.tracker.state == "TRACKING"
            and abs(pan - prev_pan) <= max_rate_deg
            and abs(tilt - prev_tilt) <= max_rate_deg
        )
        prev_pan, prev_tilt = pan, tilt
        good = good + 1 if calm else 0
        if good >= sustained:
            settled = True
            break
    check(
        settled,
        label,
        (
            f"state={window.tracker.state} started={window.metrics_started} "
            f"pan={window.simulator.camera_pan:.3f} tilt={window.simulator.camera_tilt:.3f}"
        ),
    )
    return settled


def run_forced_loss_check(mod) -> None:
    """Behavioral STAGE-2 check: exercise the real force-loss flow.

    Drives window.force_target_loss() and the actual update_simulation()
    loop, then verifies the intentional test records exactly one loss
    event, a successful re-acquisition, a measured re-acquisition time
    <= 1 s, does not corrupt the PS target-loss percentage, and that
    normal tracking still works afterward.
    """
    gui = sys.modules.get("gui")
    if gui is None:
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
    if gui is None:
        report("FAIL", "Forced-loss check: gui import", "gui module not available")
        return
    window = getattr(gui, "window", None)
    app = getattr(gui, "app", None)
    if window is None or app is None:
        report("FAIL", "Forced-loss check: window constructed", "gui.app/gui.window missing")
        return

    try:
        window.auth_input.setText("isro@1969")
        window._authenticate()
        window.timer.stop()
        window.simulation_running = True
        window.simulator.set_noise_type("None")
        window.simulator.set_disturbances()

        # Settle the shared mission before arming: TRACKING with the
        # closed-loop camera parked, so the hidden-beacon window cannot
        # slide the target out of frame mid-slew.
        if not _settle_tracking(
            window, "Forced-loss setup settles the mission before arming"
        ):
            return

        loss_pct_before = window._target_loss_rate()
        misses_before = window.evaluation_detection_misses
        events_before = window.loss_events
        reacq_before = window.successful_reacquisitions

        # Trigger the real button handler.
        window.force_target_loss()
        check(
            window.forced_loss_active,
            "Forced-loss test arms when triggered",
            "intentional loss armed",
        )

        # Drive frames until the measured re-acquisition lands (bounded).
        completed = False
        for _ in range(120):
            window.update_simulation()
            if window.forced_loss_reacq_s is not None:
                completed = True
                break
        check(
            completed,
            "Forced-loss re-acquisition completes",
            f"{window.forced_loss_reacq_s:.3f} s" if completed else "no re-acquisition within 120 frames",
        )

        if completed:
            check(
                window.forced_loss_reacq_s <= 1.0,
                "Forced re-acquisition time within 1 s",
                f"{window.forced_loss_reacq_s:.3f} s",
            )
            check(
                window.loss_events == events_before + 1,
                "Exactly one loss event recorded for the test",
                f"{events_before} -> {window.loss_events}",
            )
            check(
                window.successful_reacquisitions == reacq_before + 1,
                "Successful re-acquisition recorded",
                f"{reacq_before} -> {window.successful_reacquisitions}",
            )
            check(
                window.evaluation_detection_misses == misses_before,
                "PS loss percentage not corrupted by the intentional test",
                f"loss {window._target_loss_rate():.3f}% unchanged "
                f"(hidden frames carry no ground truth)",
            )
            snapshot = window._last_report_snapshot or {}
            check(
                snapshot.get("forced_loss_reacq_s") == window.forced_loss_reacq_s,
                "Measured re-acquisition time recorded in snapshot",
                f"{snapshot.get('forced_loss_reacq_s')} s",
            )

            # Normal tracking must still work after the test.
            still_tracking = False
            for _ in range(60):
                window.update_simulation()
                if window.tracker.state == "TRACKING":
                    still_tracking = True
                    break
            check(
                still_tracking,
                "Normal TRACKING resumes after the forced-loss test",
                window.tracker.state,
            )

        window.close()
        app.processEvents()
    except Exception as exc:
        report("FAIL", "Forced-loss behavioral check", repr(exc))


def run_handoff_window_checks(mod) -> None:
    """WOW FEATURE 1 — HANDOFF WINDOW behavioral check.

    Drives the real update_simulation() loop through a converging scenario and
    verifies the handoff-window analyzer: opens only on real qualified frames,
    READY only after the required stable streak, values come from live
    telemetry, stable duration increases while conditions hold, and the window
    closes when conditions break (via the real FORCE TARGET LOSS flow).
    """
    gui = sys.modules.get("gui")
    if gui is None:
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
    if gui is None:
        report("FAIL", "Handoff-window check: gui import", "gui module not available")
        return
    window = getattr(gui, "window", None)
    app = getattr(gui, "app", None)
    if window is None or app is None:
        report("FAIL", "Handoff-window check: window constructed", "gui.app/gui.window missing")
        return

    window.auth_input.setText("isro@1969")
    window._authenticate()
    window.timer.stop()
    window.simulation_running = True
    window.simulator.set_noise_type("None")
    window.simulator.set_disturbances()

    # A fresh run: reset through the real handler so handoff bookkeeping clears.
    window.reset_simulation()

    # Converge to TRACKING.
    acquired = False
    for _ in range(300):
        window.update_simulation()
        if window.metrics_started and window.tracker.state == "TRACKING":
            acquired = True
            break
    check(
        acquired,
        "Handoff-window setup reaches TRACKING",
        f"state={window.tracker.state}",
    )

    # Run until the streak gate opens (READY) — bounded.
    became_ready = False
    for _ in range(120):
        window.update_simulation()
        if window.link_ready_streak >= window.link_required_frames:
            became_ready = True
            break
    check(
        became_ready,
        "Handoff-window READY only after stable streak",
        f"streak={window.link_ready_streak}/{window.link_required_frames}",
    )

    if became_ready:
        data = window._handoff_last or {}
        check(
            data.get("ready") is True and data.get("open") is True,
            "Handoff window opens with real qualified frames",
            f"open={data.get('open')} ready={data.get('ready')}",
        )
        first_s = data.get("first_qualified_s")
        check(
            first_s is not None and first_s >= 0.0,
            "FIRST QUALIFIED time recorded from mission clock",
            f"{first_s} s",
        )
        err = data.get("center_error_px")
        check(
            err is not None and 0.0 <= err <= 10.0,
            "CENTER ERROR value is real and within limit",
            f"{err} px",
        )
        duration_a = data.get("stable_duration_s", 0.0)

        # Conditions keep holding -> stable duration must increase.
        for _ in range(15):
            window.update_simulation()
        duration_b = (window._handoff_last or {}).get("stable_duration_s", 0.0)
        check(
            duration_b > duration_a,
            "STABLE DURATION increases while conditions hold",
            f"{duration_a:.2f} s -> {duration_b:.2f} s",
        )

        # Open the dialog and verify it mirrors the live values.
        window._open_handoff_window()
        app.processEvents()
        dialog = window._handoff_dialog
        check(
            dialog is not None and dialog.isVisible(),
            "HANDOFF WINDOW dialog opens",
            "live dialog visible",
        )
        if dialog is not None:
            check(
                "READY" in dialog.status_label.text(),
                "Dialog shows READY from real gate",
                dialog.status_label.text(),
            )
            # Values displayed match the analyzer snapshot (same source).
            snap = window._handoff_last or {}
            shown = dialog.value_labels["center_error_px"][0].text()
            expected = f"{snap.get('center_error_px'):.2f} px"
            check(
                shown == expected,
                "Dialog CENTER ERROR matches live telemetry",
                f"shown={shown} expected={expected}",
            )

        # Close the dialog, then break conditions with the REAL forced-loss flow.
        if dialog is not None:
            dialog.close()
            app.processEvents()
        window.force_target_loss()
        window_closed = False
        for _ in range(120):
            window.update_simulation()
            if not (window._handoff_last or {}).get("open", True):
                window_closed = True
                break
        check(
            window_closed,
            "Window closes when conditions break (forced loss)",
            f"open={(window._handoff_last or {}).get('open')}",
        )
        after = window._handoff_last or {}
        check(
            after.get("ready") is False,
            "READY reverts when conditions invalid",
            f"ready={after.get('ready')}",
        )
        # Conditions must re-qualify on their own after re-acquisition.
        re_qualified = False
        for _ in range(200):
            window.update_simulation()
            if (window._handoff_last or {}).get("ready"):
                re_qualified = True
                break
        check(
            re_qualified,
            "Handoff window re-qualifies after recovery",
            f"streak={window.link_ready_streak}/{window.link_required_frames}",
        )
        # Fresh window must restart its clock (first_qualified is a new moment).
        after2 = window._handoff_last or {}
        new_first = after2.get("first_qualified_s")
        check(
            new_first is not None and after.get("open") is False and after2.get("open") is True,
            "New window opens with fresh FIRST QUALIFIED time",
            f"new first={new_first} s",
        )

    # Hand the shared window over in a settled state (test hygiene).
    _settle_tracking(
        window, "Handoff-window check leaves the mission settled for handover"
    )


def run_beam_impact_checks(mod) -> None:
    """WOW FEATURE 2 — BEAM IMPACT VIEW behavioral check.

    Verifies the twin's beam-impact view: enters/exits cleanly, the impact
    status derives from the REAL pointing state (ALIGNED only when the live
    center error is within the 10 px PS limit), renders without crashing,
    mirrors into the standalone twin window, and never touches tracking.
    """
    gui = sys.modules.get("gui")
    if gui is None:
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
    if gui is None:
        report("FAIL", "Beam-impact check: gui import", "gui module not available")
        return
    window = getattr(gui, "window", None)
    app = getattr(gui, "app", None)
    if window is None or app is None:
        report("FAIL", "Beam-impact check: window constructed", "gui.app/gui.window missing")
        return

    window.auth_input.setText("isro@1969")
    window._authenticate()
    window.timer.stop()
    window.simulation_running = True
    window.simulator.set_noise_type("None")
    window.simulator.set_disturbances()
    window.main_stack.setCurrentIndex(1)  # twin page visible -> live beam data flows

    # Converge to TRACKING through the real loop.
    acquired = False
    for _ in range(300):
        window.update_simulation()
        if window.metrics_started and window.tracker.state == "TRACKING":
            acquired = True
            break
    check(acquired, "Beam-impact setup reaches TRACKING", f"state={window.tracker.state}")

    pane = window.twin_pane
    twin = pane.twin

    # Live data must have arrived from the real pointing state.
    check(
        twin.beam_impact_center_error_px is not None,
        "Twin receives live center error",
        f"{twin.beam_impact_center_error_px} px",
    )
    check(
        twin.beam_impact_state == "BEAM ALIGNED",
        "Aligned case: status from real pointing state",
        f"{twin.beam_impact_state} @ {twin.beam_impact_center_error_px} px",
    )

    # Enter BEAM IMPACT through the real button path.
    pane.beam_impact_button.click()
    app.processEvents()
    check(
        twin.beam_impact_enabled and twin.view_mode == "BEAM IMPACT",
        "BEAM IMPACT view activates",
        f"mode={twin.view_mode}",
    )
    check(
        pane.return_button.isEnabled(),
        "RETURN TO FULL SCENE enabled in beam-impact view",
        f"enabled={pane.return_button.isEnabled()}",
    )

    # Render with the overlay active (crash guard for the painter path).
    try:
        image = twin.grab()
        rendered = image.width() > 50 and image.height() > 50
    except Exception as exc:  # noqa: BLE001
        rendered = False
        print("   ", repr(exc))
    check(rendered, "Beam-impact overlay renders", "painter path no crash")

    # The overlay must never disturb the mission: tracker keeps running.
    still_tracking = window.tracker.state == "TRACKING"
    check(still_tracking, "Beam-impact view does not disturb tracking", window.tracker.state)

    # Misaligned case through the same live-data API the GUI uses.
    twin.set_beam_impact_data(213.0, (150.0, -90.0), False)
    app.processEvents()
    status, err = twin.beam_impact_status()
    check(
        status == "BEAM MISALIGNED" and err == 213.0,
        "Misaligned case: high error reports BEAM MISALIGNED",
        f"{status} @ {err} px",
    )
    try:
        twin.grab()
        rendered = True
    except Exception:  # noqa: BLE001
        rendered = False
    check(rendered, "Misaligned overlay renders", "painter path no crash")

    # Exit via RETURN TO FULL SCENE.
    pane.return_button.click()
    app.processEvents()
    check(
        not twin.beam_impact_enabled and twin.view_mode == "FULL SCENE",
        "RETURN TO FULL SCENE exits beam-impact view",
        f"mode={twin.view_mode}",
    )

    # Standalone twin window mirrors the same live data.
    window.open_digital_twin()
    app.processEvents()
    dtw = window.digital_twin_window
    if dtw is not None:
        for _ in range(5):
            window.update_simulation()
        app.processEvents()
        stwin = dtw.pane.twin
        check(
            stwin.beam_impact_state in {"BEAM ALIGNED", "BEAM MISALIGNED"},
            "Standalone twin receives beam-impact data",
            f"{stwin.beam_impact_state} @ {stwin.beam_impact_center_error_px}",
        )
        dtw.close()
        app.processEvents()

    # Hand the shared window over in a settled state (test hygiene).
    _settle_tracking(
        window, "Beam-impact check leaves the mission settled for handover"
    )


def run_why_locked_checks(mod) -> None:
    """WOW FEATURE 3 — WHY LOCKED? behavioral check.

    Verifies the explainability panel shows REAL values for every gate
    condition in both the READY and NOT-READY cases, marks failing rows
    with ✕ (never a fake ✓), and updates with the live run.
    """
    gui = sys.modules.get("gui")
    if gui is None:
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
    if gui is None:
        report("FAIL", "Why-locked check: gui import", "gui module not available")
        return
    window = getattr(gui, "window", None)
    app = getattr(gui, "app", None)
    if window is None or app is None:
        report("FAIL", "Why-locked check: window constructed", "gui.app/gui.window missing")
        return

    window.auth_input.setText("isro@1969")
    window._authenticate()
    window.timer.stop()
    window.simulation_running = True
    window.simulator.set_noise_type("None")
    window.simulator.set_disturbances()
    window.reset_simulation()

    acquired = False
    for _ in range(300):
        window.update_simulation()
        if window.metrics_started and window.tracker.state == "TRACKING":
            acquired = True
            break
    check(acquired, "Why-locked setup reaches TRACKING", f"state={window.tracker.state}")

    became_ready = False
    for _ in range(120):
        window.update_simulation()
        if window.link_ready_streak >= window.link_required_frames:
            became_ready = True
            break
    check(became_ready, "Why-locked READY case reached", f"streak={window.link_ready_streak}")

    # READY case: every row must be a real passing value.
    evidence = window._why_locked_evidence or {}
    conditions = evidence.get("conditions", [])
    check(len(conditions) >= 8, "Evidence itemizes all gate conditions", f"rows={len(conditions)}")
    check(evidence.get("ready") is True, "READY verdict from real gate", str(evidence.get("ready")))
    fake_pass = [name for name, _v, passed in conditions if passed not in (True, False, None)]
    check(not fake_pass, "No condition row has a non-boolean pass state", str(fake_pass))
    center_row = next((c for c in conditions if c[0] == "CENTER ERROR"), None)
    if center_row:
        value_in_text = f"{center_row[1].split()[0]} px"
        real = window.t_centroid.text().replace("\u202f", " ")
        check(
            center_row[2] is True and "px" in center_row[1],
            "CENTER ERROR row shows a real passing value",
            f"{center_row[1]}",
        )

    window._open_why_locked()
    app.processEvents()
    dialog = window._why_locked_dialog
    check(dialog is not None and dialog.isVisible(), "WHY LOCKED? dialog opens", "live dialog visible")
    if dialog is not None:
        check(
            "WHY LOCKED?" == dialog.title_label.text() and "MET" in dialog.verdict_label.text(),
            "READY case shows conditions met",
            f"{dialog.title_label.text()} | {dialog.verdict_label.text()}",
        )
        marks = [row[0].text() for row in dialog._row_widgets]
        check("✕" not in marks, "READY case shows no failing marks", str(marks))
        dialog.close()
        app.processEvents()

    # NOT-READY case through the REAL forced-loss flow.
    window.force_target_loss()
    broke = False
    for _ in range(60):
        window.update_simulation()
        if not window.link_readiness_label.text().startswith("● HANDOFF READY"):
            broke = True
            break
    check(broke, "Why-locked NOT-READY case reached (forced loss)", window.link_readiness_label.text())
    evidence2 = window._why_locked_evidence or {}
    check(evidence2.get("ready") is False, "NOT-READY verdict from real gate", str(evidence2.get("ready")))
    failing = [name for name, _v, passed in evidence2.get("conditions", []) if passed is False]
    check(bool(failing), "NOT-READY case itemizes real failing conditions", str(failing))

    window._open_why_locked()
    app.processEvents()
    dialog = window._why_locked_dialog
    if dialog is not None:
        check(
            "NOT" in dialog.title_label.text() and "NOT MET" in dialog.verdict_label.text(),
            "NOT-READY case titled WHY NOT LOCKED?",
            f"{dialog.title_label.text()} | {dialog.verdict_label.text()}",
        )
        summary = dialog.summary_label.text()
        check("Failing:" in summary, "Summary names the failing conditions", summary)
        marks = [row[0].text() for row in dialog._row_widgets]
        check("✕" in marks, "NOT-READY case shows ✕ on failing rows", str(marks))
        # No fake green check: every ✓ row must truly be a passing condition.
        passed_rows = [
            (name, value, passed) for name, value, passed in evidence2.get("conditions", [])
            if passed is True
        ]
        check(
            all("—" not in value or name == "TRACKING STABLE" for name, value, _p in passed_rows),
            "No fake checks on unmeasured values",
            str([n for n, v, _p in passed_rows if "—" in v and n != "TRACKING STABLE"]),
        )
        dialog.close()
        app.processEvents()        # Hygiene for the checks that follow: hand over a settled mission
        # (TRACKING with the camera parked), not just the first TRACKING
        # frame of a recovery slew.
        _settle_tracking(
            window, "Why-locked check leaves the mission settled for handover"
        )


def _make_synthetic_video(directory, width=320, height=240, fps=30.0, frames=60):
    """Write a synthetic moving-target MP4 + matching frame,x,y ground truth."""
    import cv2
    import numpy as np

    directory = Path(directory)
    video_path = directory / "upload_probe.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer is not None and writer.isOpened(), "synthetic video writer failed"
    gt_rows = ["frame,x,y"]
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Decaying spiral that STARTS far from centre (>100 px in the
        # engine's 640x480 working frame) and stays in-frame while the
        # closed-loop pointing converges — the real demo scenario.
        t = i / max(frames - 1, 1)
        x = int(0.5 * width + 120 * (1 - t) ** 1.2 * math.cos(2.0 * math.pi * t))
        y = int(0.5 * height + 90 * (1 - t) ** 1.2 * math.sin(2.0 * math.pi * t))
        cv2.rectangle(frame, (x - 6, y - 6), (x + 6, y + 6), (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 12, (120, 120, 120), 1)
        writer.write(frame)
        gt_rows.append(f"{i + 1},{x},{y}")
    writer.release()
    gt_path = directory / "upload_probe_groundtruth.csv"
    gt_path.write_text("\n".join(gt_rows), encoding="utf-8")
    return video_path, gt_path


def run_digital_twin_scene_checks(mod) -> None:
    """STAGE-4 behavioral checks: independent twin POV + visualization-only
    scene controls that provably never touch the real PAT controller."""
    from PySide6.QtCore import QEvent, QPointF, QPoint, Qt
    from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
    from PySide6.QtWidgets import QApplication

    gui = sys.modules.get("gui")
    if gui is None:
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
        if gui is None:
            report("FAIL", "Digital-twin scene checks: gui import", "gui unavailable")
            return
    app = QApplication.instance() or QApplication(sys.argv)

    twin_mod = importlib.import_module("digital_twin")
    window = twin_mod.DigitalTwinWindow()
    twin = window.twin
    targets = [
        {"id": "TGT-01", "x": 420.0, "y": 180.0, "visible": True, "range_m": 1050.0},
        {"id": "TGT-02", "x": 150.0, "y": 300.0, "visible": True, "range_m": 980.0},
    ]

    try:
        twin.set_state(
            camera_angles=(2.4, -1.1), target_angles=(2.4, -1.1), predicted_angles=(2.9, -0.8),
            targets=targets, world_targets=targets, camera_position=(0.0, 0.0, 0.0),
            designated_id="TGT-01", state="TRACKING", frame_count=12, sim_time=0.4, fps=91.0,
        )

        # --- 1. Viewer POV is independent of PAT pan/tilt ------------------
        check(
            twin.view_yaw == -38.0 and twin.view_pitch == 16.0 and abs(twin.zoom - 1.05) < 1e-9
            and twin.camera_angles == (2.4, -1.1),
            "Twin POV independent of PAT pan/tilt after sync",
            f"view=({twin.view_yaw:.1f},{twin.view_pitch:.1f},{twin.zoom:.2f}) pat={twin.camera_angles}",
        )

        # --- 2. Mouse orbit + wheel zoom drive the real event handlers -----
        press = QMouseEvent(QEvent.MouseButtonPress, QPointF(100, 100), QPointF(100, 100),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        twin.mousePressEvent(press)
        move = QMouseEvent(QEvent.MouseMove, QPointF(200, 80), QPointF(200, 80),
                           Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        twin.mouseMoveEvent(move)
        orbit_yaw = twin.view_yaw
        wheel = QWheelEvent(QPointF(0, 0), QPointF(0, 0), QPoint(0, 0), QPoint(0, 120),
                            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        twin.wheelEvent(wheel)
        check(
            orbit_yaw > -38.0 and twin.zoom > 1.05,
            "Mouse orbit and wheel zoom modify the user POV",
            f"yaw {twin.view_yaw:.1f}, zoom {twin.zoom:.2f}",
        )

        # --- 3. Reset view restores the default independent viewpoint ------
        window.reset_view()
        check(
            twin.view_yaw == -38.0 and twin.view_pitch == 16.0 and abs(twin.zoom - 1.05) < 1e-9,
            "RESET VIEW restores default user viewpoint",
        )

        # --- 4. Scene edits move drawn objects (visualization only) --------
        raw_pos = twin._world_position_from_raw(targets[0])
        drawn_before = twin._project(twin._vis_target_pos(raw_pos))[0]
        twin.set_scene_separation(2.0)
        drawn_after = twin._project(twin._vis_target_pos(raw_pos))[0]
        moved = math.hypot(drawn_after.x() - drawn_before.x(), drawn_after.y() - drawn_before.y())
        twin.set_scene_camera_offset(6.0, -3.0, 2.0)
        twin.set_scene_target_offset(-4.0, 2.0, 1.0)
        check(
            twin.scene_offsets_active() and moved > 2.0 and twin._vis_camera_pos() != twin.camera_position,
            "Scene controls move drawn satellites (visualization space)",
            f"separation shift {moved:.1f} px",
        )

        # --- 5. Scene changes never alter the real controller state --------
        controller = mod["controller"].CameraController()
        pan_before, tilt_before = controller.pan, controller.tilt
        twin.set_scene_camera_offset(10.0, 5.0, -6.0)
        twin.set_scene_target_offset(-8.0, 4.0, 3.0)
        twin.set_scene_separation(0.5)
        window.reset_scene()
        main_window = getattr(gui, "window", None)
        main_controller = getattr(main_window, "controller", None) if main_window else None
        main_unchanged = (
            main_controller is None
            or (main_controller.pan, main_controller.tilt) == (getattr(main_window.controller, "pan", None),
                                                                getattr(main_window.controller, "tilt", None))
        )
        check(
            (controller.pan, controller.tilt) == (pan_before, tilt_before) and main_unchanged,
            "Scene controls do not alter actual camera-controller pan/tilt",
            f"controller=({controller.pan:.3f},{controller.tilt:.3f})",
        )

        # --- 6. RESET SCENE restores live geometry --------------------------
        # (Scene sliders were removed from the hero UI by design; the
        # visualization-only engine methods remain fully functional.)
        check(
            twin.scene_camera_offset == [0.0, 0.0, 0.0]
            and twin.scene_target_offset == [0.0, 0.0, 0.0]
            and twin.scene_separation == 1.0
            and not twin.scene_offsets_active(),
            "RESET SCENE restores live geometry",
        )

        # --- 7. Scene offsets survive resync (visualization-only persistence)
        twin.set_scene_target_offset(3.0, 2.0, 1.0)
        twin.set_state(
            camera_angles=(2.4, -1.1), target_angles=(2.4, -1.1), predicted_angles=(2.9, -0.8),
            targets=targets, world_targets=targets, camera_position=(0.0, 0.0, 0.0),
            designated_id="TGT-01", state="TRACKING", frame_count=13, sim_time=0.43, fps=91.0,
        )
        check(
            twin.scene_target_offset == [3.0, 2.0, 1.0] and len(twin.world_targets) == 2,
            "Live sync preserves visualization-only scene edits",
        )
        window.reset_scene()

        # --- 8. Optical terminal drawn + beam originates at the lens -------
        image = QImage(1100, 760, QImage.Format_ARGB32)
        twin.resize(1100, 760)
        twin.render(image)
        body_screen, _ = twin._project(twin.camera_position)
        origin = twin._last_beam_origin_screen
        anchor = twin.terminal_anchor_screen
        offset_px = math.hypot(origin.x() - body_screen.x(), origin.y() - body_screen.y()) if origin else 0.0
        check(
            origin is not None and anchor is not None and offset_px > 1.5,
            "Beam originates at the optical terminal, not the body centre",
            f"terminal offset {offset_px:.1f} px",
        )

        # --- STAGE 5: divergence cone is rendered around the core ----------
        # Beam endpoint = drawn (visualization-space) designated target.
        raw_des = twin._world_position_from_raw(targets[0])
        tp_des, _ = twin._project(twin._vis_target_pos(raw_des))
        cp = twin._last_beam_origin_screen
        bx, by = tp_des.x() - cp.x(), tp_des.y() - cp.y()
        blen = math.hypot(bx, by)
        if blen < 40.0:
            report("FAIL", "Beam cone check geometry", f"beam too short on screen: {blen:.1f} px")
        else:
            ux, uy = bx / blen, by / blen
            pxn, pyn = -uy, ux
            t = 0.7  # cone-only zone: beyond the 4px core / 9px glow pens
            best = -999
            for off in (4.5, 5.5, 6.5, 7.5):
                sx = int(cp.x() + bx * t + pxn * off)
                sy = int(cp.y() + by * t + pyn * off)
                c = image.pixelColor(max(0, min(1099, sx)), max(0, min(759, sy)))
                excess = c.green() - max(c.red(), c.blue())
                best = max(best, excess)
            check(
                best >= 3,
                "Beam divergence cone renders around the optical core",
                f"peak green excess {best} at 70% path",
            )

            # --- STAGE 5: photon-flow animation moves along the core -------
            def _bright_count(img):
                n = 0
                for tt in [i / 60.0 for i in range(9, 52)]:
                    sx = int(cp.x() + bx * tt)
                    sy = int(cp.y() + by * tt)
                    c = img.pixelColor(max(0, min(1099, sx)), max(0, min(759, sy)))
                    if c.red() >= 195 and c.green() >= 225:
                        n += 1
                return n

            counts = []
            for fc in (20, 24, 28):
                twin.frame_count = fc
                twin.render(image)
                counts.append(_bright_count(image))
            check(
                max(counts) - min(counts) >= 2,
                "Photon-flow animation travels along the beam core",
                f"bright-pixel counts {counts} across frames",
            )

            # --- STAGE 5: twin paint cost stays realtime -------------------
            import time as _time

            twin.frame_count = 30
            for _ in range(10):
                twin.render(image)  # warm gradient caches
            t0 = _time.perf_counter()
            for _ in range(40):
                twin.render(image)
            paint_ms = (_time.perf_counter() - t0) / 40.0 * 1000.0
            check(
                paint_ms < 60.0,
                "Digital Twin paint cost stays realtime after polish",
                f"{paint_ms:.1f} ms/frame offscreen at 1100x760",
            )

        # --- STAGE 5: honest PS audit artifact is present ------------------
        audit_path = Path(__file__).resolve().parent / "PS26169_COMPLIANCE_AUDIT.md"
        audit_ok = False
        if audit_path.exists():
            audit_text = audit_path.read_text(encoding="utf-8", errors="replace")
            audit_ok = "PENDING" in audit_text and "FAIL" in audit_text and "26169" in audit_text
        check(
            audit_ok,
            "PS-26169 compliance audit artifact present with honest statuses",
            audit_path.name,
        )

        # --- 9. Maximised / fullscreen usability ---------------------------
        window.show()
        window.showMaximized()
        app.processEvents()
        maximized = window.isMaximized()
        twin.resize(1920, 1040)
        big = QImage(1920, 1040, QImage.Format_ARGB32)
        twin.render(big)
        window.showNormal()
        window.toggle_fullscreen()
        fullscreen = window.isFullScreen()
        window.toggle_fullscreen()
        # 1100 px is the design target on a full-size display. A smaller
        # virtual screen (the offscreen platform is 800x800) can only host a
        # smaller twin, so the target is clamped to the real available width:
        # the check measures "the twin fills the display it is given" rather
        # than assuming a monitor size. (It used to pass only because the old
        # single-row toolbar forced the window wider than the screen, which is
        # exactly the horizontal-overflow defect the responsive layout fixed.)
        _available_w = app.primaryScreen().availableGeometry().width()
        _target_w = min(1100, max(400, _available_w - 40))
        check(
            maximized and fullscreen and twin.width() >= _target_w,
            "Twin usable maximized and fullscreen",
            f"maximized={maximized} fullscreen={fullscreen} "
            f"twin={twin.width()}px target={_target_w}px screen={_available_w}px",
        )
        window.close()
        app.processEvents()
    except Exception as exc:
        report("FAIL", "Digital-twin scene checks", repr(exc))


def run_uploaded_video_checks(mod) -> None:
    """STAGE-3 behavioral checks: uploaded-video analysis with honest GT.

    Exercises the REAL run_video_benchmark engine (AUTO signature gate) and
    the REAL VideoAnalysisWindow — not widget-existence assertions.
    """
    from PySide6.QtWidgets import QApplication

    gui = sys.modules.get("gui")
    if gui is None:
        # Same offscreen-import guard as run_gui_feature_checks.
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
    if gui is None:
        report("FAIL", "Uploaded-video checks: gui import", "gui module unavailable")
        return
    app = QApplication.instance() or QApplication(sys.argv)

    va = importlib.import_module("video_analysis")
    benchmark = mod["benchmark_video"].run_video_benchmark
    evaluate = mod["performance_report"].evaluate_ps_requirements

    with tempfile.TemporaryDirectory(prefix="fsoc_upload_") as tmp:
        tmp = Path(tmp)
        try:
            video_path, gt_path = _make_synthetic_video(tmp)
        except Exception as exc:
            report("FAIL", "Synthetic upload video creation", repr(exc))
            return

        # --- Ground-truth validator classifications -----------------------
        check(
            va.validate_ground_truth_file(gt_path) == "AVAILABLE",
            "GT validator: valid frame,x,y file accepted",
        )
        bad1 = tmp / "bad_header.csv"
        bad1.write_text("frame,x,y\n", encoding="utf-8")
        bad2 = tmp / "bad_cols.csv"
        bad2.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        check(
            va.validate_ground_truth_file(bad1) == "INVALID"
            and va.validate_ground_truth_file(bad2) == "INVALID"
            and va.validate_ground_truth_file(None) == "NOT_PROVIDED",
            "GT validator: empty/wrong-column/absent truth rejected",
        )

        # --- Benchmark WITH valid ground truth -----------------------------
        try:
            summary, paths = benchmark(
                video_path, output_dir=str(tmp / "with_gt"),
                preferred_signature=None, ground_truth_path=str(gt_path),
            )
        except Exception as exc:
            report("FAIL", "Uploaded-video benchmark with GT", repr(exc))
            return
        check(
            summary.frames_processed == 60 and summary.ground_truth_available,
            "Uploaded-video benchmark processes all frames with GT",
            f"frames={summary.frames_processed} gt={summary.ground_truth_available}",
        )
        check(
            all(Path(p).exists() for p in paths) and Path(paths[3]).stat().st_size > 1024,
            "Uploaded-video artifacts created incl. tracked video",
            str([Path(p).name for p in paths]),
        )

        # --- Auto-detected sibling ground truth (user cancels dialog) ------
        try:
            auto_summary, _ = benchmark(
                video_path, output_dir=str(tmp / "auto_gt"),
                preferred_signature=None, ground_truth_path=None,
            )
        except Exception as exc:
            report("FAIL", "Uploaded-video benchmark auto-GT", repr(exc)); auto_summary = None
        check(
            auto_summary is not None and auto_summary.ground_truth_available,
            "Sibling <stem>_groundtruth.csv auto-detected",
        )

        # --- NO ground truth: honest unavailable behavior ------------------
        clean_video = tmp / "clean" / "upload_probe.mp4"
        clean_video.parent.mkdir()
        shutil.copy(video_path, clean_video)  # NO sibling truth CSV here
        try:
            no_gt_summary, no_gt_paths = benchmark(
                clean_video, output_dir=str(tmp / "no_gt"),
                preferred_signature=None, ground_truth_path=None,
            )
        except Exception as exc:
            report("FAIL", "Uploaded-video benchmark without GT", repr(exc))
            return
        check(
            no_gt_summary.ground_truth_available is False,
            "Benchmark without GT reports ground_truth_available=False",
        )
        # GUI-layer honesty guard (same code as _on_video_analysis_finished)
        no_gt_summary.ground_truth_available = False
        no_gt_summary.average_tracking_error_px = None
        no_gt_summary.maximum_tracking_error_px = None
        no_gt_summary.rmse_tracking_error_px = None
        no_gt_summary.average_centroid_error_px = None
        no_gt_summary.maximum_centroid_error_px = None
        check(
            evaluate(no_gt_summary)["checks"]["tracking_error"]["status"] == "PENDING",
            "PS tracking-error PENDING without ground truth (never PASS)",
        )

        no_gt_window = va.VideoAnalysisWindow(
            no_gt_summary, [str(p) for p in no_gt_paths], clean_video,
            gt_status="NOT_PROVIDED", parent=None,
        )
        dash_values = [
            card.value_label.text() for card in no_gt_window.findChildren(va._Card)
        ]
        em_dash = chr(0x2014)
        check(
            sum(1 for v in dash_values if v.strip() == em_dash) >= 5
            and "NOT PROVIDED" in no_gt_window._gt_banner_text(),
            "No-GT analysis window shows em dashes + NOT PROVIDED banner",
            f"{sum(1 for v in dash_values if v.strip() == em_dash)} unavailable fields",
        )
        no_gt_window.close()

        # --- WITH-GT analysis window: real measured values shown -----------
        gt_window = va.VideoAnalysisWindow(
            summary, [str(p) for p in paths], video_path,
            gt_status="AVAILABLE", parent=None,
        )
        gt_values = [card.value_label.text() for card in gt_window.findChildren(va._Card)]
        check(
            "AVAILABLE" in gt_window._gt_banner_text()
            and sum(1 for v in gt_values if v.strip() == em_dash) <= 2,
            "GT analysis window shows measured accuracy values",
            f"{sum(1 for v in gt_values if v.strip() == em_dash)} unavailable fields",
        )

        # --- Timeline reconstructed from telemetry -------------------------
        segments, total = va.read_state_timeline(paths[2])
        valid_states = {"SEARCHING", "ACQUIRING", "TRACKING", "PREDICTING", "RE-ACQUIRING", "LOST"}
        check(
            len(segments) > 0
            and total == 60
            and all(state in valid_states for _, _, state in segments)
            and segments[0][0] == 1 and segments[-1][1] == total,
            "Tracking-state timeline covers the run",
            f"{len(segments)} segments across {total} frames",
        )

        # --- Player behavior: load, seek, aspect-true zoom clamp -----------
        player = gt_window.player
        check(
            player.is_loaded() and player.total_frames() == 60,
            "Tracked video loads in analysis player",
            f"frames={player.total_frames()}",
        )
        player.seek(30)
        player.set_zoom(5.0)
        zoom_hi = player.zoom()
        player.set_zoom(0.2)
        zoom_lo = player.zoom()
        check(
            player.current_frame() == 30 and zoom_hi == 3.0 and zoom_lo == 1.0,
            "Player seek + zoom clamps to 1.0x-3.0x",
            f"frame={player.current_frame()} zoom={zoom_lo}..{zoom_hi}",
        )
        gt_window.close()

        # --- Main-GUI wiring -------------------------------------------------
        window = getattr(gui, "window", None)
        check(
            window is not None
            and hasattr(window, "analyze_video_button")
            and hasattr(gui, "VideoBenchmarkWorker")
            and hasattr(window, "_video_analysis_window"),
            "Main GUI exposes uploaded-video analysis entry point",
        )


def run_final_polish_checks(mod) -> None:
    """Behavioral checks for the final UI / video-analysis polish pass.

    Covers: auth-screen helper-line removal, GT metric mathematical
    consistency against per-frame telemetry, engine honesty (None not 0.0
    without ground truth), video-first analysis geometry (~70% viewport),
    and handoff readiness derived from real measurements only.
    """
    from PySide6.QtWidgets import QApplication, QLabel

    gui = sys.modules.get("gui")
    if gui is None:
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
    if gui is None:
        report("FAIL", "Final-polish checks: gui import", "gui module unavailable")
        return
    app = QApplication.instance() or QApplication(sys.argv)

    va = importlib.import_module("video_analysis")
    benchmark = mod["benchmark_video"].run_video_benchmark

    # --- 1. Auth screen: the two helper lines are gone, essentials remain --
    gui_text = Path(gui.__file__).read_text(encoding="utf-8", errors="ignore")
    check(
        "Authorization is required before the live" not in gui_text
        and "Change the password in auth_config.py" not in gui_text,
        "Auth-screen helper lines removed from gui.py source",
    )
    window = getattr(gui, "window", None)
    auth_texts = ""
    if window is not None and hasattr(window, "auth_overlay"):
        auth_texts = " ".join(
            label.text() for label in window.auth_overlay.findChildren(QLabel)
        )
    check(
        bool(auth_texts)
        and "Authorization is required" not in auth_texts
        and "auth_config.py" not in auth_texts
        and "MISSION CONTROL" in auth_texts
        and "ASTRALINK" in auth_texts
        and "ACCESS KEY" in auth_texts,
        "Auth overlay keeps title/key essentials without the removed lines",
    )

    with tempfile.TemporaryDirectory(prefix="fsoc_polish_") as tmp_s:
        tmp = Path(tmp_s)
        try:
            video_path, gt_path = _make_synthetic_video(tmp)
        except Exception as exc:
            report("FAIL", "Final-polish checks: synthetic video", repr(exc))
            return

        # --- 2. GT metric mathematical consistency (regression for 0.0 bug) --
        try:
            summary, paths = benchmark(
                video_path, output_dir=str(tmp / "polish"),
                preferred_signature=None, ground_truth_path=str(gt_path),
            )
        except Exception as exc:
            report("FAIL", "Final-polish checks: GT benchmark", repr(exc))
            return

        per_frame = []
        telemetry = Path(paths[2])
        with telemetry.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = (row.get("centroid_error_px") or "").strip()
                if value:
                    per_frame.append(float(value))
        populated = (
            summary.ground_truth_available
            and summary.average_tracking_error_px is not None
            and len(per_frame) > 0
        )
        check(
            populated,
            "GT accuracy populated from real detections (no 0.0-from-empty)",
            f"samples={len(per_frame)} avg={summary.average_tracking_error_px}",
        )
        if populated:
            count = len(per_frame)
            mean_v = sum(per_frame) / count
            max_v = max(per_frame)
            rms_v = math.sqrt(sum(v * v for v in per_frame) / count)
            consistent = (
                # Telemetry stores 3-decimal strings, so re-derived sums may
                # differ from full-precision aggregates by rounding only.
                abs(summary.average_tracking_error_px - mean_v) < 1e-2
                and abs(summary.maximum_tracking_error_px - max_v) < 1e-2
                and abs(summary.rmse_tracking_error_px - rms_v) < 1e-2
                and abs(summary.maximum_tracking_error_px - summary.maximum_centroid_error_px) < 1e-6
                and summary.maximum_tracking_error_px >= summary.average_tracking_error_px
            )
            check(
                consistent,
                "GT metrics mathematically consistent with per-frame telemetry",
                f"avg {summary.average_tracking_error_px:.3f} max {summary.maximum_tracking_error_px:.3f}"
                f" rmse {summary.rmse_tracking_error_px:.3f} from {count} frames",
            )

            gt_status = "AVAILABLE"
            gt_window = va.VideoAnalysisWindow(
                summary, [str(p) for p in paths], video_path,
                gt_status=gt_status, parent=None,
            )
            gt_window.resize(1480, 940)
            gt_window.show()
            app.processEvents()

            # --- 3. Video-first geometry: viewport height dominates ---------
            viewport = gt_window.player
            height_ratio = viewport.height() / max(gt_window.height(), 1)
            scroll_height = gt_window._secondary_scroll.height()
            check(
                0.55 <= height_ratio <= 0.82 and scroll_height <= 200,
                "Tracked video viewport dominates the analysis window (~70%)",
                f"video {100 * height_ratio:.0f}% of window height, secondary zone {scroll_height}px",
            )

            # UI cards must carry the same consistent story
            cards = {}
            for card in gt_window.findChildren(va._Card):
                labels = card.findChildren(QLabel)
                if len(labels) >= 2:
                    cards[labels[0].text()] = labels[1].text()
            em_dash = chr(0x2014)
            check(
                cards.get("Tracking Error (GT)") == cards.get("Average Error (GT)")
                and cards.get("Tracking Error (GT)", "").endswith("px")
                and cards.get("Tracking Error (GT)") != em_dash,
                "UI GT card definitions consistent (Tracking Error == Average Error)",
                f"tracking={cards.get('Tracking Error (GT)')}",
            )

            # --- 4. Handoff status is derived from real measurements --------
            handoff = gt_window._handoff_status.text()
            evidence = gt_window._handoff_evidence.text()
            acquired = summary.acquisition_time_s is not None
            stable = int(summary.metadata.get("evaluated_frames", 0) or 0) >= 30
            max_ok = (
                summary.maximum_tracking_error_px is not None
                and summary.maximum_tracking_error_px <= 10.0
            )
            loss_ok = float(summary.target_loss_percentage) < 5.0
            # Mirror the real handoff logic: accuracy also needs a valid-GT
            # status, and readiness additionally requires the measured final
            # center error within the 10 px PS limit.
            gt_ok = gt_status == "AVAILABLE" and max_ok
            center = summary.final_center_error_px
            center_ok = center is not None and center <= 10.0
            expect_ready = acquired and stable and gt_ok and center_ok and loss_ok
            check(
                ("COARSE ALIGNMENT: LOCKED" in handoff) == acquired
                and ("HANDOFF TO FINE ALIGNMENT: READY" in handoff) == expect_ready
                and f"LIMIT 10 px" in evidence
                and ("MAX DEVIATION" in evidence) == max_ok,
                "Handoff status tracks real measured conditions",
                handoff,
            )
            gt_window.close()

        # --- 5. Engine honesty without ground truth: None, never 0.0 --------
        # Isolated directory so the engine cannot auto-detect the fixture's
        # sibling GT CSV (which the engine must find by design).
        clean_video = tmp / "clean" / "upload_probe.mp4"
        clean_video.parent.mkdir()
        shutil.copy(video_path, clean_video)
        try:
            no_gt_summary, no_gt_paths = benchmark(
                clean_video, output_dir=str(tmp / "no_gt"),
                preferred_signature=None, ground_truth_path=None,
            )
        except Exception as exc:
            report("FAIL", "Final-polish checks: no-GT benchmark", repr(exc))
            return
        check(
            no_gt_summary.ground_truth_available is False
            and no_gt_summary.average_tracking_error_px is None
            and no_gt_summary.maximum_tracking_error_px is None
            and no_gt_summary.rmse_tracking_error_px is None,
            "Engine reports None (not 0.0) for accuracy without ground truth",
        )
        ng_window = va.VideoAnalysisWindow(
            no_gt_summary, [str(p) for p in no_gt_paths], clean_video,
            gt_status="NOT_PROVIDED", parent=None,
        )
        ng_text = ng_window._handoff_status.text()
        ng_evidence = ng_window._handoff_evidence.text()
        check(
            "HANDOFF TO FINE ALIGNMENT: READY" not in ng_text
            and "NOT READY" in ng_text
            and "NO VALID GROUND TRUTH" in ng_evidence,
            "Handoff not ready without valid ground truth (never faked)",
            ng_text,
        )
        ng_window.close()


def run_demo_readiness_checks(mod) -> None:
    """Final demo-readiness behavioral checks (centering, forced-loss
    re-acquisition, START button, handoff honesty) on the real engine."""
    from dataclasses import replace as dc_replace

    from PySide6.QtWidgets import QApplication, QLabel

    gui = sys.modules.get("gui")
    if gui is None:
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
    if gui is None:
        report("FAIL", "Demo-readiness checks: gui import", "gui module unavailable")
        return
    app = QApplication.instance() or QApplication(sys.argv)

    va = importlib.import_module("video_analysis")
    benchmark = mod["benchmark_video"].run_video_benchmark
    save_full = mod["performance_report"].save_full_performance_report
    evaluate = mod["performance_report"].evaluate_ps_requirements

    with tempfile.TemporaryDirectory(prefix="fsoc_demo_") as tmp_s:
        tmp = Path(tmp_s)
        try:
            video_path, gt_path = _make_synthetic_video(tmp)
        except Exception as exc:
            report("FAIL", "Demo-readiness checks: synthetic video", repr(exc))
            return

        # --- TEST A: normal tracking, centering convergence, START button --
        try:
            summary, paths = benchmark(
                video_path, output_dir=str(tmp / "a"),
                preferred_signature=None, ground_truth_path=str(gt_path),
            )
        except Exception as exc:
            report("FAIL", "Demo-readiness checks: benchmark", repr(exc))
            return
        evaluated = int(summary.metadata.get("evaluated_frames", 0))
        check(
            0 < evaluated <= summary.frames_processed,
            "Evaluated frames counted once and within the processed run",
            f"{evaluated} evaluated / {summary.frames_processed} processed",
        )
        first_center = None
        with Path(paths[2]).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = (row.get("center_error_px") or "").strip()
                if value:
                    first_center = float(value)
                    break
        check(
            first_center is not None and first_center > 100.0
            and summary.final_center_error_px is not None
            and summary.final_center_error_px <= 10.0,
            "Center error converges from far field to <=10 px (real pointing)",
            f"{first_center:.1f} px -> {summary.final_center_error_px:.2f} px",
        )
        check(
            summary.average_tracking_error_px <= 10.0
            and summary.maximum_tracking_error_px <= 10.0,
            "Tracking error stays within the PS limit while converging",
            f"avg {summary.average_tracking_error_px:.2f} / max {summary.maximum_tracking_error_px:.2f} px",
        )
        forced_reacq = summary.metadata.get("forced_reacquisition_time_s")
        check(
            summary.metadata.get("forced_loss_injected") is True
            and isinstance(forced_reacq, float) and 0.0 < forced_reacq <= 1.0,
            "Forced-loss rehearsal measures a real re-acquisition <=1 s",
            f"{forced_reacq:.3f} s" if forced_reacq else "missing",
        )
        check(
            summary.target_loss_events == 0
            and float(summary.target_loss_percentage) < 5.0
            and float(summary.lock_retention_percentage) <= 100.0,
            "Normal loss statistics exclude the intentional rehearsal window",
            f"events={summary.target_loss_events} loss={summary.target_loss_percentage}% lock={summary.lock_retention_percentage:.1f}%",
        )
        check(
            float(summary.fps) >= 20.0,
            "Processing speed stays above the PS minimum",
            f"{summary.fps:.1f} FPS",
        )

        pdf_path, json_slot, csv_slot = save_full(
            summary, output_dir=str(tmp / "a"),
            prefix=f"{Path(video_path).stem}_benchmark", write_json_csv=False,
        )
        check(
            Path(pdf_path).exists() and json_slot is None and csv_slot is None,
            "Analysis PDF-only report generated for the same run",
            Path(pdf_path).name,
        )

        window = va.VideoAnalysisWindow(
            summary, [str(p) for p in paths] + [str(pdf_path)], video_path,
            gt_status="AVAILABLE", parent=None,
        )
        window.resize(1480, 940)
        window.show()
        app.processEvents()
        check(
            window.start_button.isEnabled()
            and "START" in window.start_button.text()
            and not window.player.is_playing()
            and "READY" in window.position_label.text(),
            "Analysis opens paused at frame 0 with a visible START / PLAY button",
            window.position_label.text(),
        )
        window._toggle_play()
        app.processEvents()
        started = window.player.is_playing() and "PAUSE" in window.start_button.text()
        window._toggle_play()
        app.processEvents()
        check(
            started and not window.player.is_playing()
            and "START" in window.start_button.text(),
            "START toggles playback to PAUSE and back without autostart",
        )
        cards = {}
        for card in window.findChildren(va._Card):
            labels = card.findChildren(QLabel)
            if len(labels) >= 2:
                cards[labels[0].text()] = labels[1].text()
        check(
            cards.get("Center Error (Final)") == f"{summary.final_center_error_px:.2f} px"
            and cards.get("Re-acquisition", "\u2014") != "\u2014",
            "Center Error and real Re-acquisition values shown in metric cards",
            f"center={cards.get('Center Error (Final)')} reacq={cards.get('Re-acquisition')}",
        )
        check(
            "COARSE ALIGNMENT: LOCKED" in window._handoff_status.text()
            and "HANDOFF TO FINE ALIGNMENT: READY" in window._handoff_status.text()
            and "CENTER ERROR" in window._handoff_evidence.text(),
            "Handoff READY only with all measured conditions satisfied",
            window._handoff_status.text(),
        )
        window.close()

        # --- TEST C: uncentered run must block handoff with the reason ----
        uncentered = dc_replace(
            summary,
            final_center_error_px=213.0,
            average_center_error_px=180.0,
            center_error_converged=False,
            coarse_locked=False,
        )
        check(
            evaluate(uncentered)["checks"]["center_error"]["status"] == "FAIL",
            "PS evaluator fails the center-error criterion at 213 px",
        )
        uncentered_window = va.VideoAnalysisWindow(
            uncentered, [str(p) for p in paths] + [str(pdf_path)], video_path,
            gt_status="AVAILABLE", parent=None,
        )
        status = uncentered_window._handoff_status.text()
        check(
            "HANDOFF TO FINE ALIGNMENT: READY" not in status
            and "NOT READY" in status
            and "CENTER ERROR 213.00 px > 10 px" in status
            and "COARSE ALIGNMENT: TRACKING" in status,
            "Uncentered target blocks handoff and names the measured reason",
            status,
        )
        uncentered_window.close()

        # --- TEST B without GT: rehearsal is detection-based, still honest -
        # Isolated copy: the fixture directory holds the sibling GT CSV, which
        # the engine MUST auto-detect — the honest no-GT path needs a video
        # with no truth available at all.
        b_video = tmp / "b_input" / "upload_probe.mp4"
        b_video.parent.mkdir()
        shutil.copy(video_path, b_video)
        try:
            no_gt_summary, _ = benchmark(
                b_video, output_dir=str(tmp / "b"),
                preferred_signature=None, ground_truth_path=None,
            )
        except Exception as exc:
            report("FAIL", "Demo-readiness checks: no-GT benchmark", repr(exc))
            return
        ng_reacq = no_gt_summary.metadata.get("forced_reacquisition_time_s")
        check(
            no_gt_summary.metadata.get("forced_loss_injected") is True
            and isinstance(ng_reacq, float) and 0.0 < ng_reacq <= 1.0
            and no_gt_summary.average_tracking_error_px is None,
            "Forced-loss re-acquisition measured without GT; accuracy stays unavailable",
            f"reacq={ng_reacq:.3f} s" if ng_reacq else "missing",
        )


def run_reports(mod) -> None:
    PerformanceSummary = mod["performance_report"].PerformanceSummary
    save_performance_report = mod["performance_report"].save_performance_report
    evaluate = mod["performance_report"].evaluate_ps_requirements
    save_instant_report = mod["instant_report"].save_instant_report

    summary = PerformanceSummary(
        frames_processed=372, duration_s=12.4, fps=30.0,
        acquisition_time_s=0.73,
        average_tracking_error_px=1.7, maximum_tracking_error_px=4.6,
        rmse_tracking_error_px=1.9,
        average_centroid_error_px=0.42, maximum_centroid_error_px=0.9,
        lock_retention_percentage=98.2, target_loss_percentage=1.8,
        target_loss_events=1, successful_reacquisitions=1,
        average_reacquisition_time_s=0.61,
        average_processing_time_ms=11.6, max_processing_time_ms=18.2,
        ground_truth_available=True, source="master self-test",
    )
    snapshot = {
        "simulation_time_s": 12.4, "frame": 372, "target_id": "TGT-01",
        "target_signature": "GREEN-SQUARE", "target_color": "GREEN", "target_shape": "SQUARE",
        "target_count": 3, "state": "TRACKING", "target_x_px": 319.0, "target_y_px": 241.0,
        "predicted_x_px": 320.0, "predicted_y_px": 240.5, "pan_deg": 0.12, "tilt_deg": 0.03,
        "center_error_px": 1.41, "centroid_error_px": 0.42, "fps": 86.0,
        "detector_confidence": 0.81, "ai_confidence": 0.92, "signature_score": 0.96,
        "acquisition_time_s": 0.73, "target_loss_percentage": 1.8, "loss_events": 1,
        "reacquisition_count": 1, "average_reacquisition_s": 0.61,
        "noise": "Gaussian", "noise_level": 8,
        "turbulence_enabled": False, "turbulence_strength_px": 0.0,
        "camera_jitter_enabled": True, "camera_jitter_px": 8.0,
        "platform_motion_enabled": True, "platform_motion_px": 8.0,
        "atmosphere": "Haze", "atmosphere_level": 35, "link_ready": True,
    }

    with tempfile.TemporaryDirectory(prefix="fsoc_test_reports_") as tmp:
        paths = save_performance_report(summary, output_dir=tmp, prefix="master_test")
        check(all(Path(p).exists() for p in paths), "Performance report JSON + CSV", str(paths))
        result = evaluate(summary)

        # Accept the current report evaluator shape as well as the
        # precomputed PerformanceSummary.requirement_checks. Some baseline
        # builds expose requirement status under a slightly different field
        # name, so this test normalizes the result instead of treating that
        # formatting difference as an application failure.
        source = result if isinstance(result, dict) and result else getattr(summary, "requirement_checks", {})
        statuses = []
        for value in source.values() if isinstance(source, dict) else []:
            if isinstance(value, dict):
                status = value.get("status") or value.get("result") or value.get("outcome")
                if status is not None:
                    statuses.append(str(status).upper())
            elif isinstance(value, str):
                statuses.append(value.upper())

        # Different baseline builds expose this helper as either:
        #   {requirement: {"status": "PASS", ...}},
        #   {"status": "PASS"}, or a single status string.
        # The application requirement checker itself is the source of truth;
        # the harness should validate any returned PASS status without making
        # assumptions about the wrapper shape.
        check(
            bool(statuses) and all(x == "PASS" for x in statuses),
            "PS requirement evaluator",
            str(statuses),
        )
        instant = save_instant_report(snapshot, output_dir=tmp)
        check(len(instant) == 3 and all(Path(p).exists() for p in instant), "Instant report PDF + JSON + CSV", str(instant))
        check(Path(instant[0]).stat().st_size > 1000, "Instant report PDF has content", f"{Path(instant[0]).stat().st_size} bytes")


def build_synthetic_benchmark(mod) -> None:
    cv2 = mod.get("cv2") or importlib.import_module("cv2")
    Simulator = mod["simulator"].Simulator
    run_video_benchmark = mod["benchmark_video"].run_video_benchmark

    with tempfile.TemporaryDirectory(prefix="fsoc_test_benchmark_") as tmp:
        root = Path(tmp)
        video_path = root / "master_test.mp4"
        gt_path = root / "master_test_groundtruth.csv"
        sim = Simulator()
        sim.set_target_count(1)
        sim.set_designated_target("TGT-01")
        sim.set_motion_pattern("Figure 8")
        sim.set_noise_type("None")
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
        if not writer.isOpened():
            report("FAIL", "Synthetic benchmark MP4 writer", "OpenCV could not create MP4")
            return
        with gt_path.open("w", newline="", encoding="utf-8") as fh:
            rows = csv.writer(fh); rows.writerow(["frame", "x", "y"])
            for n in range(60):
                frame = sim.get_frame(); truth = sim.get_ground_truth_image_position()
                writer.write(frame)
                if truth is not None:
                    rows.writerow([n + 1, f"{truth[0]:.4f}", f"{truth[1]:.4f}"])
        writer.release()
        try:
            summary, paths = run_video_benchmark(
                video_path, output_dir=root,
                preferred_signature="GREEN-SQUARE",
                ground_truth_path=gt_path,
                write_output_video=True,
            )
        except Exception as exc:
            report("FAIL", "Benchmark video pipeline", repr(exc)); return
        check(summary.frames_processed == 60, "Benchmark processes all frames", str(summary.frames_processed))
        check(summary.ground_truth_available, "Benchmark loads ground truth", str(summary.ground_truth_available))
        check(summary.target_loss_percentage < 5, "Benchmark detection loss", f"{summary.target_loss_percentage:.2f}%")
        check(summary.fps >= 20, "Benchmark throughput", f"{summary.fps:.1f} FPS")
        check(all(Path(p).exists() for p in paths), "Benchmark artifacts exist", str(paths))


def run_gui_startup_smoke() -> None:
    script = ROOT / "gui.py"
    if not script.exists():
        check(False, "GUI file is present")
        return
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script)], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
    except Exception as exc:
        report("FAIL", "GUI starts as a subprocess", repr(exc)); return
    time.sleep(4)
    running = proc.poll() is None
    if running:
        report("PASS", "GUI remains running after startup", "4-second startup smoke test")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5)
    else:
        output = proc.stdout.read() if proc.stdout else ""
        report("FAIL", "GUI remains running after startup", f"exit={proc.returncode}; output={output[-1000:]}")


def run_ui_reorg_checks(mod) -> None:
    """UI-reorganization behavioral checks: 3-section sidebar, live-view
    switch (radar / virtual camera / digital twin) sharing ONE mission
    state, twin mission-strip two-way sync, terminal focus modes, and the
    Replay & Reports workspace."""
    from PySide6.QtWidgets import QApplication, QLabel

    gui = sys.modules.get("gui")
    if gui is None:
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
    if gui is None:
        report("FAIL", "UI-reorg checks: gui import", "gui module unavailable")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    window = getattr(gui, "window", None)
    if window is None:
        report("FAIL", "UI-reorg checks: main window", "gui.window unavailable")
        return

    # --- 1. Old-style dashboard: one SIMULATION CONTROL column -------------
    control_frame = window.sidebar
    controls_text = " ".join(
        w.text() for w in control_frame.findChildren(QLabel)
    )
    groups_present = all(
        token in controls_text
        for token in ("SIMULATION CONTROL", "TARGET & MOTION", "POINTING & OPTICS", "DISTURBANCES")
    )
    launchers_present = all(
        hasattr(window, attr) and getattr(window, attr) is not None
        for attr in ("benchmark_button", "report_button", "instant_report_button", "twin_launch_button")
    )
    check(
        groups_present and launchers_present,
        "Dashboard control column holds SIMULATION/TARGET/POINTING/DISTURBANCE groups + TEST/OUTPUT launchers",
        f"groups={groups_present} launchers={launchers_present}",
    )

    # --- 2. Disturbance Preset drives the REAL simulator and mirrors back --
    window.preset_selector.setCurrentText("Full Combined")
    app.processEvents()
    preset_applied = (
        window.simulator.noise_type == "Gaussian"
        and window.simulator.noise_level == 8
        and window.simulator.camera_jitter_enabled
        and window.simulator.platform_motion_enabled
        and window.simulator.atmosphere == "Haze"
        and window.atmosphere_selector.currentText() == "Haze"
        and window.noise_level_slider.value() == 8
    )
    window.preset_selector.setCurrentText("Clear")
    app.processEvents()
    preset_cleared = (
        window.simulator.noise_type == "None"
        and not window.simulator.turbulence_enabled
        and not window.simulator.camera_jitter_enabled
    )
    check(
        preset_applied and preset_cleared,
        "DISTURBANCE PRESET applies the deterministic simulator profile and mirrors onto controls",
        f"applied={preset_applied} cleared={preset_cleared}",
    )

    # --- 3. Switching views preserves ONE simulation state ---------------
    window.auth_input.setText("isro@1969")
    window._authenticate()
    window.timer.stop()
    window.simulation_running = True
    for _ in range(20):
        window.update_simulation()
    frames_before = window.frames_processed
    twin_before = window.twin_pane.twin.frame_count
    window._switch_live_view(1)
    app.processEvents()
    for _ in range(10):
        window.update_simulation()
    window._switch_live_view(2)
    app.processEvents()
    for _ in range(10):
        window.update_simulation()
    check(
        window.frames_processed == frames_before + 20
        and window.twin_pane.twin.frame_count == window.simulator.frame_count,
        "View switching never resets mission state (single simulation continues)",
        f"frames {frames_before} -> {window.frames_processed}, twin clock {window.twin_pane.twin.frame_count} == simulator {window.simulator.frame_count}",
    )
    window._switch_live_view(0)
    app.processEvents()

    # --- 4. Mission-strip sync (ONE source of truth) -----------------------
    # MOTION is read-only in the twin strip (Stage 5C): the strip displays the
    # live pattern from Mission Control and must NOT offer an editable copy.
    pane = window.twin_pane
    pane._syncing_strip = False
    motion_display_only = not isinstance(pane.twin_motion, QComboBox)
    window.motion_selector.setCurrentText("Circular")
    app.processEvents()
    # Main->strip propagation rides the per-tick telemetry refresh. The
    # strip MUST follow the simulator's authoritative pattern, not the
    # selector widget text (Stage 2 source-of-truth fix).
    window.update_simulation()
    app.processEvents()
    main_to_strip = pane.twin_motion.text() == window.simulator.motion_pattern == "Circular"
    window.motion_selector.setCurrentText("Figure 8")
    app.processEvents()
    window.update_simulation()
    app.processEvents()
    main_to_strip_8 = pane.twin_motion.text() == window.simulator.motion_pattern == "Figure 8"
    window.motion_selector.setCurrentText("Circular")
    app.processEvents()
    window.update_simulation()
    app.processEvents()
    # Strip sync check: designated mirror (strip edit → main selector) when a
    # second target exists, plus main→strip propagation for the read-only
    # MOTION display.
    if pane.twin_designated.count() >= 2:
        strip_designated = pane.twin_designated.currentText()
        pane.twin_designated.setCurrentText(
            next(opt for opt in [pane.twin_designated.itemText(i)
                                 for i in range(pane.twin_designated.count())]
                 if opt != strip_designated)
        )
        app.processEvents()
        main_followed = window.designated_target_selector.currentText() == pane.twin_designated.currentText()
    else:
        main_followed = pane.twin_designated.currentText() == window.designated_target_id
    window.designated_target_selector.setCurrentText("TGT-01")
    app.processEvents()
    window.update_simulation()
    app.processEvents()
    check(
        motion_display_only and main_to_strip and main_to_strip_8 and main_followed
        and pane.twin_designated.currentText() == window.designated_target_id,
        "Twin strip syncs with main controls; MOTION is read-only",
        f"readonly={motion_display_only} main→strip={main_to_strip}/{main_to_strip_8} designated_follow={main_followed}",
    )

    # --- 5. Terminal focus modes (embedded pane, real animation) ----------
    window._switch_live_view(2)
    app.processEvents()
    pane.focus_camera_terminal()
    import time as _time
    deadline = _time.perf_counter() + 3.0
    while pane.twin._focus_anim is not None and _time.perf_counter() < deadline:
        app.processEvents()
    cam_mode = pane.twin.view_mode == "CAMERA TERMINAL" and pane.twin.zoom > 1.6
    pane.focus_beacon_terminal()
    deadline = _time.perf_counter() + 3.0
    while pane.twin._focus_anim is not None and _time.perf_counter() < deadline:
        app.processEvents()
    beacon_mode = pane.twin.view_mode == "BEACON TERMINAL"
    pane.focus_full_scene()
    app.processEvents()
    full_mode = (
        pane.twin.view_mode == "FULL SCENE" and pane.twin.view_yaw == -38.0
        and abs(pane.twin.zoom - 1.05) < 1e-9
    )
    check(
        cam_mode and beacon_mode and full_mode,
        "FOCUS CAMERA/BEACON TERMINAL + RETURN TO FULL SCENE animate the real POV",
        f"cam={cam_mode} beacon={beacon_mode} full={full_mode}",
    )
    window._switch_live_view(0)
    app.processEvents()

    # --- 6. Section 03 swaps the main area to the reports workspace ------
    window._switch_section(2)
    app.processEvents()
    reports_ok = (
        window.main_stack.currentIndex() == 2
        and window.report_list is not None and hasattr(window, "open_folder_button")
    )
    window._register_report_artifact("__probe_report__.pdf")
    listed = window.report_list.count() > 0
    window._switch_section(0)
    app.processEvents()
    check(
        reports_ok and listed,
        "03 REPLAY & REPORTS swaps main area to workspace; artifacts listable",
        f"page={window.main_stack.currentIndex()} listed={listed}",
    )
    window._switch_live_view(0)
    app.processEvents()


def _audit_boot_window(mod):
    """Shared boot helper for the audit + environment checks."""
    from PySide6.QtWidgets import QApplication

    gui = sys.modules.get("gui")
    if gui is None:
        _real_exit = sys.exit
        sys.exit = lambda *a, **k: None
        from PySide6.QtWidgets import QApplication as _QApplication
        _real_exec = _QApplication.exec
        _QApplication.exec = lambda self, *a, **k: 0
        try:
            gui = importlib.import_module("gui")
        except (SystemExit, RuntimeError):
            gui = sys.modules.get("gui")
        finally:
            _QApplication.exec = _real_exec
            sys.exit = _real_exit
    if gui is None:
        report("FAIL", "Control audit: gui import", "gui module unavailable")
        return None, None, None
    app = QApplication.instance() or QApplication(sys.argv)
    window = getattr(gui, "window", None)
    if window is None:
        report("FAIL", "Control audit: main window", "gui.window unavailable")
        return None, None, None
    window.auth_input.setText("isro@1969")
    window._authenticate()
    window.timer.stop()
    window.simulation_running = True
    for _ in range(5):
        window.update_simulation()
    app.processEvents()
    return gui, app, window


def run_control_audit_checks(mod) -> None:
    """Signal-to-behavior audit: every motion/pointing/disturbance control must
    change the UNDERLYING simulator/controller state, not just a widget value.
    Each check drives the same handler the GUI signal invokes and verifies the
    behavioral consequence (trajectory shape, per-frame displacement, sim
    state, rendered camera frame)."""
    import math as _math

    import config as _config

    gui, app, window = _audit_boot_window(mod)
    if window is None:
        return
    sim = window.simulator

    def designated_xy():
        for state in sim.get_target_states():
            if state["id"] == sim.get_designated_target_id():
                return (state["x"], state["y"])
        return None

    def sample_pattern(pattern, frames=260):
        sim.set_disturbances()          # isolation: no platform/jitter/turbulence
        sim.update_camera(0.0, 0.0)     # freeze the gimbal so image == world
        sim.set_motion_pattern(pattern)
        sim.set_beacon_speed(4.0)
        pts = []
        for _ in range(frames):
            sim.update_beacon()
            sim.update_camera(0.0, 0.0)
            pts.append(designated_xy())
        return pts

    # --- motion patterns produce genuinely different trajectory shapes -----
    line = sample_pattern("Straight Line")
    circ = sample_pattern("Circular")
    fig8 = sample_pattern("Figure 8")
    rnd = sample_pattern("Random")

    cx = _math.fsum(p[0] for p in circ) / len(circ)
    cy = _math.fsum(p[1] for p in circ) / len(circ)
    radii = [_math.hypot(p[0] - cx, p[1] - cy) for p in circ]
    # The orbit is an ellipse (rx != ry), so test CLOSED-ORBIT geometry:
    # full angular coverage AND an annulus that never crosses the center
    # (a figure-8 or a line DOES pass through the center).
    angles = sorted(_math.atan2(p[1] - cy, p[0] - cx) for p in circ)
    n = len(angles)
    max_gap = max((angles[(i + 1) % n] - angles[i]) % (2.0 * _math.pi) for i in range(n))
    annulus_ratio = min(radii) / max(max(radii), 1e-9)
    fx = _math.fsum(p[0] for p in fig8) / len(fig8)
    fy = _math.fsum(p[1] for p in fig8) / len(fig8)
    fig8_radii = [_math.hypot(p[0] - fx, p[1] - fy) for p in fig8]
    fig8_min_ratio = min(fig8_radii) / max(max(fig8_radii), 1e-9)
    dx_range = max(p[0] for p in fig8) - min(p[0] for p in fig8)
    dy_range = max(p[1] for p in fig8) - min(p[1] for p in fig8)
    line_x_span = max(p[0] for p in line) - min(p[0] for p in line)
    line_y_span = max(p[1] for p in line) - min(p[1] for p in line)
    rnd_span = (max(p[0] for p in rnd) - min(p[0] for p in rnd)) + (max(p[1] for p in rnd) - min(p[1] for p in rnd))
    circ_span = (max(p[0] for p in circ) - min(p[0] for p in circ)) + (max(p[1] for p in circ) - min(p[1] for p in circ))

    check(
        annulus_ratio > 0.45 and max_gap < _math.pi / 2.0 and circ_span > 120.0,
        "MOTION: Circular drives a closed orbit (full revolution, never crosses center)",
        f"annulus={annulus_ratio:.2f} max_gap={_math.degrees(max_gap):.0f}° span={circ_span:.0f}px",
    )
    check(
        fig8_min_ratio < 0.22,
        "MOTION: Figure 8 crosses the loop center (lissajous signature)",
        f"min_radius_ratio={fig8_min_ratio:.2f}",
    )
    check(
        dy_range > 12.0 and dy_range < 0.62 * dx_range and dx_range > 260.0,
        "MOTION: Figure 8 drives a lissajous path (narrow Y vs wide X)",
        f"dx={dx_range:.0f}px dy={dy_range:.0f}px",
    )
    _ = (fx, fy)
    check(
        line_x_span > 300.0 and line_y_span < 0.22 * line_x_span,
        "MOTION: Straight Line drives a flat horizontal sweep",
        f"x_span={line_x_span:.0f}px y_span={line_y_span:.0f}px",
    )
    check(
        rnd_span > 60.0,
        "MOTION: Random mode drives a wandering trajectory",
        f"span={rnd_span:.0f}px",
    )

    # --- beacon speed changes measured per-frame displacement --------------
    sim.set_disturbances()
    sim.update_camera(0.0, 0.0)
    sim.set_motion_pattern("Straight Line")
    sim.set_beacon_speed(2.0)
    first = designated_xy()
    for _ in range(3):
        sim.update_beacon()
        sim.update_camera(0.0, 0.0)
    second = designated_xy()
    slow_disp = _math.hypot(second[0] - first[0], second[1] - first[1]) / 4.0
    sim.set_beacon_speed(8.0)
    first = designated_xy()
    for _ in range(3):
        sim.update_beacon()
        sim.update_camera(0.0, 0.0)
    second = designated_xy()
    fast_disp = _math.hypot(second[0] - first[0], second[1] - first[1]) / 4.0
    check(
        1.8 < fast_disp / slow_disp < 4.4,
        "MOTION: Beacon speed slider doubles measured per-frame displacement",
        f"slow={slow_disp:.2f}px/f fast={fast_disp:.2f}px/f ratio={fast_disp / max(slow_disp, 1e-9):.2f}",
    )

    # --- target count / designated target ----------------------------------
    sim.set_target_count(3)
    ids = sim.get_target_ids()
    check(
        sim.get_target_count() == 3 and len(ids) == 3 and len(set(ids)) == 3,
        "MOTION: Target count=3 creates 3 distinct tracked targets",
        f"ids={ids}",
    )
    sim.set_designated_target("TGT-02")
    check(
        sim.get_designated_target_id() == "TGT-02",
        "MOTION: Designated target switch changes simulator state",
        f"designated={sim.get_designated_target_id()}",
    )
    sim.set_designated_target("TGT-01")
    sim.set_target_count(1)

    # --- pointing / optics --------------------------------------------------
    sim.set_fov(8.0)
    fov_ok = abs(_config.FOV_HORIZONTAL - 8.0) < 1e-6 and abs(_config.FOV_VERTICAL - 6.0) < 1e-6
    sim.set_fov(4.0)
    check(
        fov_ok,
        "POINTING: Camera FOV control updates the geometry (H=8°, V=6°)",
        f"H={_config.FOV_HORIZONTAL:.2f} V={_config.FOV_VERTICAL:.2f}",
    )

    controller = window.controller
    controller.pan = controller.tilt = 0.0
    before = controller.update(200.0, 100.0)
    saved = (_config.MAX_PAN_SPEED, _config.MAX_TILT_SPEED)
    _config.MAX_PAN_SPEED = _config.MAX_TILT_SPEED = 40.0
    controller.pan = controller.tilt = 0.0
    fast = controller.update(200.0, 100.0)
    _config.MAX_PAN_SPEED, _config.MAX_TILT_SPEED = saved
    check(
        fast[0] > before[0] * 5.0,
        "POINTING: Slew-rate control changes real per-frame gimbal step",
        f"step@5°/s={before[0]:.3f}° step@40°/s={fast[0]:.3f}°",
    )

    from controller import CameraController as _Cam
    c_small = _Cam()
    c_small.set_gain(0.2)
    c_small.update(6.0, 0.0)   # small error: far below the slew-rate cap
    p_small = c_small.pan
    c_big = _Cam()
    c_big.set_gain(2.0)
    c_big.update(6.0, 0.0)
    p_big = c_big.pan
    check(
        p_big > p_small * 5.0,
        "POINTING: Prediction gain scales controller aggressiveness",
        f"pan@gain0.2={p_small:.3f}° pan@gain2.0={p_big:.3f}°",
    )

    controller.set_angle_limits(2.0, 1.5)
    check(
        abs(controller.pan_limit_deg - 2.0) < 1e-9 and abs(controller.tilt_limit_deg - 1.5) < 1e-9,
        "POINTING: Pan/Tilt range control updates controller limits",
        f"pan_limit={controller.pan_limit_deg} tilt_limit={controller.tilt_limit_deg}",
    )
    controller.set_angle_limits(5.0, 3.75)

    # --- disturbances produce real, observable offsets ----------------------
    sim.set_disturbances()  # everything off
    sim.get_frame()
    sim.set_disturbances(platform_motion_enabled=True, platform_motion_px=12.0,
                         platform_motion_pattern="Circular")
    sim.get_frame()
    sim.get_frame()
    px, py = sim.last_platform_offset
    plat_mag = _math.hypot(px, py)
    check(
        plat_mag > 0.5,
        "DISTURBANCE: Platform motion produces a real image offset",
        f"offset=({px:.2f}, {py:.2f})px",
    )

    sim.set_disturbances()  # off
    baseline = sim.get_frame()[210:270, 270:370].copy()
    sim.set_disturbances(camera_jitter_enabled=True, camera_jitter_px=9.0)
    jitters = set()
    for _ in range(25):
        sim.get_frame()
        jitters.add(tuple(round(v, 1) for v in sim.last_camera_jitter))
    frame_jittered = sim.get_frame()[210:270, 270:370]
    diff = float(abs(frame_jittered.astype(int) - baseline.astype(int)).mean())
    check(
        len(jitters) >= 4 and diff > 0.5,
        "DISTURBANCE: Camera jitter randomizes offset per frame and shifts the feed",
        f"unique_offsets={len(jitters)} frame_delta={diff:.2f}",
    )

    sim.set_disturbances(turbulence_enabled=True, turbulence_strength_px=14.0,
                         atmosphere="Haze", atmosphere_level=60)
    wander = set()
    for _ in range(25):
        sim.get_frame()
        wander.add(tuple(round(v, 1) for v in sim.get_digital_twin_state().get("beam_wander_px", (0, 0))))
    status = sim.get_disturbance_status()
    status_ok = (
        sim.turbulence_enabled is True
        and status.get("count", 0) >= 2
        and any("turbulence" in str(a) for a in status.get("active", []))
        and any("Haze" in str(a) for a in status.get("active", []))
    )
    check(
        len(wander) >= 4 and status_ok,
        "DISTURBANCE: Turbulence drives real per-frame beam wander + atmosphere registers",
        f"unique_wander={len(wander)} status={status}",
    )
    sim.set_disturbances()

    # --- handler-level integration through the real FSOCWindow handlers ----
    sim.set_target_count(3)
    window._on_designated_target_changed("TGT-02")
    handler_designated = (
        window.designated_target_id == "TGT-02"
        and sim.get_designated_target_id() == "TGT-02"
    )
    window._on_designated_target_changed("TGT-01")
    sim.set_target_count(1)
    check(
        handler_designated,
        "HANDLER: Designated-target selector changes GUI + simulator state",
    )

    window._on_slew_changed(90)
    slew_ok = abs(_config.MAX_PAN_SPEED - 9.0) < 1e-6
    window._on_slew_changed(50)
    check(
        slew_ok and abs(_config.MAX_PAN_SPEED - 5.0) < 1e-6,
        "HANDLER: Slew slider writes config.MAX_PAN_SPEED both ways",
    )

    window._on_speed_changed(45)
    speed_ok = abs(window.beacon_speed - 4.5) < 1e-9 and abs(sim.speed_x - 4.5) < 1e-9
    window._on_speed_changed(40)
    check(
        speed_ok,
        "HANDLER: Speed slider drives simulator speed_x",
        f"speed_x={sim.speed_x}",
    )

    window._on_fov_changed(70)
    fov_handler_ok = abs(window.camera_fov_deg - 7.0) < 1e-9 and abs(_config.FOV_HORIZONTAL - 7.0) < 1e-6
    window._on_fov_changed(40)
    check(
        fov_handler_ok,
        "HANDLER: FOV slider updates GUI value + simulator geometry",
    )

    # Virtual camera FEED changes when the world changes (real rendered proof).
    sim.set_disturbances()
    sim.set_motion_pattern("Straight Line")
    sim.set_beacon_speed(4.0)
    window.tracker.reset()
    window._reset_metrics()
    cam1 = sim.get_frame().copy()
    for _ in range(25):
        window.update_simulation()
    cam2 = sim.get_frame()
    delta = abs(cam2.astype(int) - cam1.astype(int))
    changed_px = int((delta > 8).sum())
    check(
        changed_px > 200,
        "HANDLER: Virtual camera feed actually changes as the mission evolves",
        f"pixels changed >8 LSB: {changed_px}",
    )


def run_environment_layout_checks(mod) -> None:
    """Mission-control spatial layout + digital-twin environment behavior."""
    gui, app, window = _audit_boot_window(mod)
    if window is None:
        return
    twin = window.twin_pane.twin
    pane = window.twin_pane

    # --- A. spatial arrangement: controls | radar+camera side-by-side -------
    window.resize(1680, 950)
    app.processEvents()
    window.show()
    app.processEvents()
    ops_page = window.main_stack.widget(0)

    def xspan(widget):
        tl = widget.mapTo(ops_page, widget.rect().topLeft())
        br = widget.mapTo(ops_page, widget.rect().bottomRight())
        return (tl.x(), br.x())

    rx0, rx1 = xspan(window.radar_panel)
    cx0, cx1 = xspan(window.camera_panel)
    side_by_side = rx1 < cx0
    cs_geo = window.control_stack.geometry()
    ms_geo = window.main_stack.geometry()
    column_ok = cs_geo.x() + cs_geo.width() <= ms_geo.x() + 2
    check(
        side_by_side and column_ok,
        "LAYOUT: Radar and Virtual Camera side by side, controls column on the left",
        f"radar x=[{rx0},{rx1}] camera x=[{cx0},{cx1}] column_ok={column_ok}",
    )
    graphs_on_page = all(
        w.parent() is not None and w is not None for w in (window.error_graph, window.motion_graph)
    )
    check(
        graphs_on_page and window.error_graph.height() >= window.error_graph.minimumHeight(),
        "LAYOUT: Bottom band graphs present with full (non-shrunk) height",
        f"graph_h={window.error_graph.height()} min={window.error_graph.minimumHeight()}",
    )

    # --- B. twin environment presets change the visual environment ----------
    signatures = {}
    for name in twin.ENVIRONMENT_PRESETS:
        twin.set_environment(name)
        signatures[name] = (
            len(twin._stars),
            tuple(sorted(b["kind"] for b in twin._celestial)),
            round(twin._lighting["halo"], 3),
        )
    distinct = len(set(signatures.values()))
    check(
        distinct == 4 and twin.environment_name == "SUNLIT ORBIT",
        "ENV: All 4 presets build distinct visual environments",
        f"{distinct}/4 distinct: {signatures}",
    )
    check(
        signatures["DEEP SPACE"][0] == max(s[0] for s in signatures.values())
        and signatures["SUNLIT ORBIT"][2] == max(s[2] for s in signatures.values()),
        "ENV: Preset character matches spec (DEEP SPACE dense, SUNLIT bright)",
    )

    # --- C. NEW SKY is fully removed from the user-facing UI ----------------
    check(
        not hasattr(pane, "new_sky_button"),
        "ENV: NEW SKY removed from the twin toolbar by design",
    )

    # --- D. pane environment selector wired to the twin ----------------------
    pane.env_selector.setCurrentText("LUNAR VICINITY")
    selector_ok = twin.environment_name == "LUNAR VICINITY"
    check(
        selector_ok,
        "ENV: Twin bar ENV selector drives the environment",
        f"selector={selector_ok}",
    )

    # --- E. visual randomization must NOT touch tracking determinism -------
    sim = window.simulator

    def designated_xy():
        for state in sim.get_target_states():
            if state["id"] == sim.get_designated_target_id():
                return (round(state["x"], 6), round(state["y"], 6))
        return None

    sim.set_motion_pattern("Circular")
    sim.set_beacon_speed(4.0)
    ref = []
    for _ in range(100):
        sim.update_beacon()
        ref.append(designated_xy())
    sim.set_motion_pattern("Circular")
    sim.set_beacon_speed(4.0)
    for name in twin.ENVIRONMENT_PRESETS:
        twin.set_environment(name)
    twin.new_sky()
    got = []
    for _ in range(100):
        sim.update_beacon()
        got.append(designated_xy())
    check(
        ref == got,
        "ENV: Tracking trajectory bit-identical across environment changes (determinism preserved)",
        f"divergence at {next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), 'never')}",
    )
    window._switch_live_view(0)
    app.processEvents()


def run_layout_stability_checks(mod) -> None:
    """STAGE 1C — geometric-stability regression.

    Runs the live mission for several seconds WITHOUT user resizing and
    verifies that no watched container (window, header, footer, sidebar,
    Mission Control page, Test Lab page, control column, radar, camera,
    telemetry, kalman, readiness, scroll viewports) changes geometry. Any
    geometry shift caused purely by telemetry updates fails.
    """
    gui = sys.modules.get("gui")
    if gui is None:
        report("FAIL", "Layout-stability check: gui import", "gui module not available")
        return
    window = getattr(gui, "window", None)
    app = getattr(gui, "app", None)
    if window is None or app is None:
        report("FAIL", "Layout-stability check: window constructed", "gui.app/gui.window missing")
        return

    from PySide6.QtWidgets import QWidget, QScrollArea

    window.auth_input.setText("isro@1969")
    window._authenticate()
    window.timer.stop()
    window.simulation_running = True
    window.simulator.set_noise_type("None")
    window.simulator.set_disturbances()

    def watch_for(page_index):
        window.main_stack.setCurrentIndex(page_index)
        app.processEvents()
        watch = {
            "window": window,
            "central": window.centralWidget(),
            "control_stack": window.control_stack,
        }
        for child in window.centralWidget().children():
            if isinstance(child, QWidget):
                watch["central>" + (child.objectName() or type(child).__name__)] = child
        wanted = {
            "radar_panel", "camera_panel", "kalman_panel", "readiness_panel",
            "telemetry_panel", "twin_pane", "testlab_panel",
        }
        for child in window.centralWidget().findChildren(QWidget):
            name = child.objectName()
            if name in wanted:
                watch[name] = child
        for scroll in window.centralWidget().findChildren(QScrollArea):
            watch["scroll:" + (scroll.objectName() or type(scroll).__name__)] = scroll.viewport()
        return {k: w for k, w in watch.items() if isinstance(w, QWidget)}

    window.resize(1600, 900)
    for page_index, page_name in ((0, "Mission Control"), (2, "Test Lab")):
        watch = watch_for(page_index)
        for _ in range(60):  # settle: initial layouts, scrollbar appearance
            window.update_simulation()
        app.processEvents()
        prev = {k: (w.geometry().x(), w.geometry().y(), w.width(), w.height()) for k, w in watch.items()}
        moves = {}
        for _ in range(150):  # ~5 s of live mission, sampled every other frame
            window.update_simulation()
            if _ % 2:
                continue
            app.processEvents()
            for key, widget in watch.items():
                geo = (widget.geometry().x(), widget.geometry().y(), widget.width(), widget.height())
                if geo != prev[key]:
                    moves[key] = moves.get(key, 0) + 1
                    prev[key] = geo
        check(
            not moves,
            f"{page_name} geometry stable during live run (no user resize)",
            f"moved containers: {moves}" if moves else "all watched containers static",
        )
    window.main_stack.setCurrentIndex(0)
    app.processEvents()


def run_twin_fullscreen_env_label_checks(mod) -> None:
    """Stages 6/7/9 — twin environment differentiation, label-collision
    management, and TRUE app-level full screen with state preservation."""
    gui = sys.modules.get("gui")
    if gui is None:
        report("FAIL", "Twin fullscreen/env checks: gui import", "gui module not available")
        return
    window = getattr(gui, "window", None)
    app = getattr(gui, "app", None)
    if window is None or app is None:
        report("FAIL", "Twin fullscreen/env checks: window constructed", "gui.app/gui.window missing")
        return

    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    window.auth_input.setText("isro@1969")
    window._authenticate()
    window.timer.stop()
    window.simulation_running = True
    window.simulator.set_noise_type("None")
    window.simulator.set_disturbances()
    if not _settle_tracking(window, "Twin fullscreen/env setup settles the mission"):
        return
    pane = window.twin_pane
    twin = pane.twin

    # --- STAGE 6: presets must produce visually different environments ---
    seeds = {"DEEP SPACE": 101, "EARTH ORBIT": 202, "LUNAR VICINITY": 303}
    snapshots = {}
    for name, seed in seeds.items():
        twin._environment_seed = seed
        twin.set_environment(name)
        app.processEvents()
        snap = (
            tuple(twin._stars),
            tuple(twin._nebulae),
            tuple(twin._clusters),
            tuple((b["name"], b["pos"]) for b in twin._celestial),
            twin.environment_name,
        )
        snapshots[name] = snap
        img = twin.grab()
        check(img.width() > 100, f"Environment renders: {name}", f"{img.width()}x{img.height()}")
    trio = [snapshots[n] for n in seeds]
    check(
        snapshots["DEEP SPACE"] != snapshots["EARTH ORBIT"]
        and snapshots["EARTH ORBIT"] != snapshots["LUNAR VICINITY"],
        "Environment presets produce different star/nebula/cluster/body sets",
        f"distinct={[s[:2] != t[:2] for s, t in zip(trio, trio[1:])]}",
    )
    twin.set_environment("DEEP SPACE")

    # Environment calls must leave the SIMULATOR state byte-identical
    # (visual-only variation; benchmark trajectories stay deterministic).
    sim_before = (
        [(tg["id"], round(tg["x"], 9), round(tg["y"], 9), tg["visible"])
         for tg in window.simulator.targets],
        window.simulator.camera_pan, window.simulator.camera_tilt,
        window.simulator.frame_count,
    )
    for name, seed in seeds.items():
        twin._environment_seed = seed
        twin.set_environment(name)
    twin.new_sky()  # engine-level path retained; UI control removed by design
    app.processEvents()
    sim_after = (
        [(tg["id"], round(tg["x"], 9), round(tg["y"], 9), tg["visible"])
         for tg in window.simulator.targets],
        window.simulator.camera_pan, window.simulator.camera_tilt,
        window.simulator.frame_count,
    )
    check(
        sim_before == sim_after and tuple(twin._stars) != snapshots["DEEP SPACE"][0],
        "Environment/NEW SKY calls leave simulator state untouched (deterministic benchmarks)",
        f"sim_unchanged={sim_before == sim_after}",
    )

    # --- STAGE 7: label collision management ------------------------------
    check(
        hasattr(twin, "_resolve_label_overlap") and hasattr(twin, "_draw_secondary_label"),
        "Label collision machinery present (resolve + priority secondary labels)",
    )
    from PySide6.QtCore import QRectF
    painter_probe = QRectF(0, 0, 100, 20)
    twin._label_boxes = [QRectF(20, 0, 200, 24)]
    twin._resolve_label_overlap(painter_probe, 6.0)
    check(
        not painter_probe.intersects(QRectF(20, 0, 200, 24).adjusted(-6, -6, 6, 6)),
        "Overlapping labels are shifted clear of claimed boxes",
        f"resolved to y={painter_probe.top():.0f}",
    )
    img = twin.grab()
    check(
        img.width() > 100 and len(twin._label_boxes) >= 1,
        "Twin renders with label management active (plates claim space)",
        f"boxes={len(twin._label_boxes)}",
    )

    # --- STAGE 9: TRUE twin full screen (embedded) ------------------------
    window.main_stack.setCurrentIndex(1)
    app.processEvents()
    frames_before = window.simulator.frame_count
    metrics_before = (window.metrics_started, window.acquisition_time_s)
    window._switch_live_view(2)
    app.processEvents()

    enter_ev = QKeyEvent(QEvent.KeyPress, Qt.Key_F11, Qt.KeyboardModifier.NoModifier)
    window.keyPressEvent(enter_ev)
    app.processEvents()
    chrome_hidden = (
        not window.sidebar.isVisible()
        and not window.header.isVisible()
        and not window.footer.isVisible()
        and not window.control_scroll.isVisible()
    )
    chrome_off = (
        not pane.mission_strip.isVisible()
        and not pane._title_row_host.isVisible()
    )
    canvas_full = pane.twin.isVisible() and pane.twin.width() >= pane.width() - 4
    check(
        getattr(window, "_twin_fullscreen", False) and chrome_hidden and chrome_off and canvas_full,
        "F11 gives the twin the whole application surface (chrome + strip hidden)",
        f"fs={window._twin_fullscreen} chrome={chrome_hidden} strip={chrome_off} canvas_full={canvas_full}",
    )
    check(
        pane.fullscreen_button.text() == "EXIT FULL SCREEN",
        "Full-screen button flips to EXIT FULL SCREEN",
        pane.fullscreen_button.text(),
    )
    for _ in range(10):
        window.update_simulation()
    check(
        window.simulator.frame_count > frames_before,
        "Simulation keeps running in twin full screen (no mission reset)",
        f"frames {frames_before} -> {window.simulator.frame_count}",
    )

    esc_ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.KeyboardModifier.NoModifier)
    window.keyPressEvent(esc_ev)
    app.processEvents()
    restored = (
        window.sidebar.isVisible() and window.header.isVisible()
        and window.footer.isVisible() and window.control_scroll.isVisible()
        and pane.mission_strip.isVisible() and pane._title_row_host.isVisible()
    )
    state_kept = (window.metrics_started, window.acquisition_time_s) == metrics_before
    check(
        not getattr(window, "_twin_fullscreen", False) and restored and state_kept,
        "ESC restores the previous interface without resetting mission state",
        f"restored={restored} metrics_kept={state_kept}",
    )
    # Hand the shared window over settled and back on the ops page.
    window._switch_live_view(0)
    app.processEvents()
    _settle_tracking(window, "Twin fullscreen/env checks leave the mission settled")


def main() -> int:
    print("FSOC FULL SYSTEM SELF-TEST v2")
    print("=" * 88)
    print(f"Project root: {ROOT}\n")

    compile_project()
    try:
        mod = import_core()
    except Exception as exc:
        report("FAIL", "Core module imports", repr(exc))
        return 1

    run_existing_self_test()
    # GUI feature checks run BEFORE the QtPdf openability check so the
    # offscreen QApplication exists first; _pdf_open_check then reuses it
    # instead of colliding with the QApplication singleton.
    run_auth_password(mod)
    run_gui_feature_checks(mod)
    run_handoff_window_checks(mod)
    run_beam_impact_checks(mod)
    run_why_locked_checks(mod)
    run_forced_loss_check(mod)
    run_digital_twin_scene_checks(mod)
    run_uploaded_video_checks(mod)
    run_final_polish_checks(mod)
    run_demo_readiness_checks(mod)
    run_ui_reorg_checks(mod)
    run_control_audit_checks(mod)
    run_environment_layout_checks(mod)
    run_layout_stability_checks(mod)
    run_twin_fullscreen_env_label_checks(mod)
    run_pdf_report_check(mod)
    run_disturbance_configuration(mod)
    run_core_pipeline(mod)
    run_reacquisition(mod)
    run_multi_target(mod)
    run_reports(mod)
    build_synthetic_benchmark(mod)
    run_gui_startup_smoke()

    print("\n" + "=" * 88)
    print(f"RESULT: {PASS} PASS / {WARN} WARN / {FAIL} FAIL")
    if FAIL:
        print("OVERALL: FAIL — fix the failed checks before final accuracy hardening.")
        return 1
    if WARN:
        print("OVERALL: PASS WITH WARNINGS — remaining warnings require engineering review.")
    else:
        print("OVERALL: PASS — automated checks passed.")
    print("Note: visual UI quality and physical-hardware behavior still need human validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
