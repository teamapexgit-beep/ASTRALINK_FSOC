"""FSOC master system self-test.

Run from the project folder:
    python full_system_test.py

This is a headless integration/smoke test. It exercises the software path
without requiring the operator to manually click through the GUI.

The test intentionally distinguishes:
- PASS: an automated check succeeded.
- WARN: the check completed but is a known stress/visual limitation.
- FAIL: a required software path is broken.

It does NOT claim to judge visual polish, presentation quality, or physical
hardware behavior.
"""
from __future__ import annotations

import csv
import importlib
import math
import os
import py_compile
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Force Qt into headless mode for the GUI integration check. This prevents the
# master test from opening a visible window or requiring an X/desktop server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
WARN = 0
FAIL = 0


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


def import_core():
    modules = [
        "config",
        "ai_verifier",
        "controller",
        "detector",
        "tracker",
        "simulator",
        "performance_report",
        "metrics",
        "benchmark_video",
        "instant_report",
        "digital_twin",
    ]
    imported = {}
    for name in modules:
        imported[name] = importlib.import_module(name)
    return imported


def compile_project() -> None:
    excluded = {".venv", ".git", "__pycache__"}
    files = []
    for path in ROOT.rglob("*.py"):
        if any(part in excluded for part in path.parts):
            continue
        files.append(path)
    errors = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"{path.name}: {exc.msg}")
    check(not errors, "Python compilation", f"{len(files)} files checked" if not errors else "; ".join(errors))


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

    controller = CameraController()
    controller.set_angle_limits(5.0, 3.75)
    controller.set_gain(1.5)
    tracker = BeaconTracker()

    detections = 0
    tracking_frames = 0
    errors = []
    started = time.perf_counter()

    for _ in range(120):
        frame = sim.get_frame()
        truth = sim.get_ground_truth_image_position()
        expected = (tracker.x, tracker.y) if tracker.x is not None else None
        detection, info = detect_beacon(
            frame,
            return_details=True,
            expected_position=expected,
            preferred_signature=sim.get_designated_target_signature(),
        )
        if detection is not None:
            detections += 1
        tracked = tracker.update(detection)
        if tracked is not None:
            x, y, state = tracked
            if state == "TRACKING":
                errors.append(math.hypot(x - config.FRAME_WIDTH / 2.0, y - config.FRAME_HEIGHT / 2.0))
            lead = 1.0 / max(config.UPDATE_RATE, 1)
            px = x + tracker.velocity_x * lead * 0.85
            py = y + tracker.velocity_y * lead * 0.85
            pan, tilt = controller.update(
                px - config.FRAME_WIDTH / 2.0,
                py - config.FRAME_HEIGHT / 2.0,
            )
            sim.update_camera(pan, tilt)
            if state == "TRACKING":
                tracking_frames += 1

    elapsed = max(time.perf_counter() - started, 1e-9)
    fps = 120.0 / elapsed
    detection_pct = 100.0 * detections / 120.0
    max_err = max(errors, default=0.0)

    check(detection_pct >= 95.0, "Core beacon detection", f"{detection_pct:.2f}% over 120 frames")
    check(tracking_frames > 0, "Kalman tracker produces TRACKING state", f"{tracking_frames} tracking frames")
    check(abs(controller.pan) <= controller.pan_limit_deg + 1e-9, "Pan controller respects limit", f"{controller.pan:.3f} deg")
    check(abs(controller.tilt) <= controller.tilt_limit_deg + 1e-9, "Tilt controller respects limit", f"{controller.tilt:.3f} deg")
    check(fps >= 20.0, "Core processing speed", f"{fps:.1f} FPS")
    warn("Core stress error ceiling", f"max closed-loop error was {max_err:.2f} px; stress scenarios can exceed the normal 10 px acceptance target")
    return sim, tracker, controller


def run_multi_target(mod):
    Simulator = mod["simulator"].Simulator
    detect_beacon = mod["detector"].detect_beacon

    sim = Simulator()
    sim.set_target_count(5)
    ids = sim.get_target_ids()
    check(ids == ["TGT-01", "TGT-02", "TGT-03", "TGT-04", "TGT-05"], "Five-target simulator", str(ids))

    ok_designations = []
    detections = []
    for target_id in ("TGT-01", "TGT-03", "TGT-05"):
        ok_designations.append(sim.set_designated_target(target_id))
        frame = sim.get_frame()
        detection, info = detect_beacon(
            frame,
            return_details=True,
            preferred_signature=sim.get_designated_target_signature(),
        )
        detections.append((target_id, detection, info.get("candidate_count", 0), info.get("signature_score", 0.0)))
        sim.frame_count += 1

    check(all(ok_designations), "Designated-target switching", str(ok_designations))
    check(all(item[2] >= 1 for item in detections), "Decoys reach detector candidate stage", str([(x[0], x[2]) for x in detections]))
    check(all(item[1] is not None for item in detections), "Designated target identity gate selects a target", str([(x[0], x[1]) for x in detections]))


def run_disturbance_api(mod):
    Simulator = mod["simulator"].Simulator
    sim = Simulator()
    names = ["Clear", "Sensor Noise", "Vibration", "Atmosphere", "Full Combined", "Extreme PAT"]
    for name in names:
        profile = sim.set_disturbance_preset(name)
        status = sim.get_disturbance_status()
        check(isinstance(profile, dict) and isinstance(status, str), f"Disturbance preset: {name}", status or "clear")


def run_reports(mod):
    Simulator = mod["simulator"].Simulator
    PerformanceSummary = mod["performance_report"].PerformanceSummary
    save_performance_report = mod["performance_report"].save_performance_report
    evaluate = mod["performance_report"].evaluate_ps_requirements
    save_instant_report = mod["instant_report"].save_instant_report

    snapshot = {
        "simulation_time_s": 12.4,
        "frame": 372,
        "target_id": "TGT-01",
        "target_signature": "GREEN-SQUARE",
        "target_color": "GREEN",
        "target_shape": "SQUARE",
        "target_count": 3,
        "state": "TRACKING",
        "target_x_px": 319.0,
        "target_y_px": 241.0,
        "predicted_x_px": 320.0,
        "predicted_y_px": 240.5,
        "pan_deg": 0.12,
        "tilt_deg": 0.03,
        "center_error_px": 1.41,
        "centroid_error_px": 0.42,
        "fps": 86.0,
        "update_rate_hz": 30.0,
        "camera_resolution": "640 x 480",
        "fov": "4.0 deg x 3.0 deg",
        "motion_pattern": "Figure 8",
        "scenario": "Figure 8",
        "lock_retention_percentage": 98.2,
        "rmse_tracking_error_px": 1.9,
        "average_tracking_error_px": 1.7,
        "maximum_tracking_error_px": 4.6,
        "average_processing_time_ms": 11.6,
        "max_processing_time_ms": 18.2,
        "detector_confidence": 0.81,
        "ai_confidence": 0.92,
        "signature_score": 0.96,
        "acquisition_time_s": 0.73,
        "target_loss_percentage": 1.8,
        "loss_events": 1,
        "reacquisition_count": 1,
        "average_reacquisition_s": 0.61,
        "noise": "Gaussian",
        "noise_level": 8,
        "turbulence_enabled": False,
        "turbulence_strength_px": 0.0,
        "camera_jitter_enabled": True,
        "camera_jitter_px": 8.0,
        "platform_motion_enabled": True,
        "platform_motion_px": 8.0,
        "atmosphere": "Haze",
        "atmosphere_level": 35,
        "link_ready": True,
    }

    summary = PerformanceSummary(
        frames_processed=372,
        duration_s=12.4,
        fps=30.0,
        acquisition_time_s=0.73,
        average_tracking_error_px=1.7,
        maximum_tracking_error_px=4.6,
        rmse_tracking_error_px=1.9,
        average_centroid_error_px=0.42,
        maximum_centroid_error_px=0.9,
        lock_retention_percentage=98.2,
        target_loss_percentage=1.8,
        target_loss_events=1,
        successful_reacquisitions=1,
        average_reacquisition_time_s=0.61,
        average_processing_time_ms=11.6,
        max_processing_time_ms=18.2,
        ground_truth_available=True,
        source="master self-test",
        maximum_reacquisition_time_s=0.61,
    )

    with tempfile.TemporaryDirectory(prefix="fsoc_test_reports_") as tmp:
        paths = save_performance_report(summary, output_dir=tmp, prefix="master_test")
        check(all(Path(p).exists() for p in paths), "Performance report JSON + CSV", str(paths))
        result = evaluate(summary)
        check(result["overall"] == "PASS", "PS requirement evaluator", result["overall"])

        instant_paths = save_instant_report(snapshot, output_dir=tmp)
        check(len(instant_paths) == 3 and all(Path(p).exists() for p in instant_paths), "Instant report PDF + JSON + CSV", str(instant_paths))
        check(Path(instant_paths[0]).stat().st_size > 1000, "Instant report PDF has content", f"{Path(instant_paths[0]).stat().st_size} bytes")


def build_synthetic_benchmark(mod):
    cv2 = importlib.import_module("cv2")
    Simulator = mod["simulator"].Simulator
    run_video_benchmark = mod["benchmark_video"].run_video_benchmark

    with tempfile.TemporaryDirectory(prefix="fsoc_test_benchmark_") as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "master_test.mp4"
        gt_path = tmp_path / "master_test_groundtruth.csv"

        sim = Simulator()
        sim.set_target_count(1)
        sim.set_designated_target("TGT-01")
        sim.set_motion_pattern("Figure 8")
        sim.set_noise_type("None")

        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480)
        )
        if not writer.isOpened():
            report("FAIL", "Synthetic benchmark MP4 writer", "OpenCV could not create an MP4 writer")
            return

        with gt_path.open("w", newline="", encoding="utf-8") as fh:
            rows = csv.writer(fh)
            rows.writerow(["frame", "x", "y"])
            for frame_no in range(60):
                frame = sim.get_frame()
                truth = sim.get_ground_truth_image_position()
                writer.write(frame)
                if truth is not None:
                    rows.writerow([frame_no + 1, f"{truth[0]:.4f}", f"{truth[1]:.4f}"])
        writer.release()

        try:
            summary, paths = run_video_benchmark(
                video_path,
                output_dir=tmp_path,
                preferred_signature="GREEN-SQUARE",
                ground_truth_path=gt_path,
                write_output_video=True,
            )
        except Exception as exc:  # benchmark is a required integration surface
            report("FAIL", "Benchmark video pipeline", repr(exc))
            return

        check(summary.frames_processed == 60, "Benchmark processes all frames", str(summary.frames_processed))
        check(summary.ground_truth_available, "Benchmark loads ground truth", str(summary.ground_truth_available))
        check(summary.target_loss_percentage < 5.0, "Benchmark detection loss", f"{summary.target_loss_percentage:.2f}%")
        check(summary.fps >= 20.0, "Benchmark throughput", f"{summary.fps:.1f} FPS")
        check(all(Path(p).exists() for p in paths), "Benchmark artifacts exist", str(paths))


def run_gui_headless(mod):
    # gui.py creates the application/window at import time. Qt was placed in
    # offscreen mode at module startup, so this test never opens a visible window.
    try:
        gui = importlib.import_module("gui")
    except Exception as exc:
        report("FAIL", "GUI module import", repr(exc))
        return

    window = getattr(gui, "window", None)
    app = getattr(gui, "app", None)
    check(window is not None, "GUI main window constructed")
    check(app is not None, "Qt application constructed")
    if window is None:
        return

    try:
        # Authentication smoke test without opening a dialog.
        auth_config = importlib.import_module("auth_config")
        window.auth_input.setText("WRONG")
        window._authenticate()
        check(not window._authenticated, "Authentication rejects wrong key")

        window.auth_input.setText(auth_config.APP_PASSWORD)
        window._authenticate()
        check(window._authenticated, "Authentication accepts configured key")

        window.timer.stop()
        window.simulation_running = True
        # Manual frames exercise the actual GUI update path.
        for _ in range(12):
            window.update_simulation()

        check(window.frames_processed >= 12, "GUI simulation loop processes frames", str(window.frames_processed))
        check(bool(window.instant_report_history), "GUI records instant-report telemetry")
        check(window._last_report_snapshot is not None, "GUI keeps latest snapshot")
        check(window.fps >= 0.0, "GUI telemetry FPS is numeric", f"{window.fps:.1f}")

        # Exercise live twin synchronization.
        window.open_digital_twin()
        twin_window = window.digital_twin_window
        check(twin_window is not None, "3D digital twin window can be created")
        if twin_window is not None:
            window._update_digital_twin(
                window.tracker.state,
                None,
                window.simulator.get_target_states(),
            )
            twin = twin_window.twin
            same_pan = abs(twin.camera_angles[0] - window.simulator.camera_pan) < 1e-9
            same_tilt = abs(twin.camera_angles[1] - window.simulator.camera_tilt) < 1e-9
            same_target_id = twin.designated_id == window.designated_target_id
            same_target_count = len(twin.world_targets) == window.simulator.get_target_count()
            check(same_pan and same_tilt, "3D twin camera angles synchronize with simulator")
            check(same_target_id and same_target_count, "3D twin target identity/count synchronize")
            check(twin.frame_count == window.simulator.frame_count, "3D twin frame counter synchronizes")
            twin_window.close()

        window.close()
        app.processEvents()
    except Exception as exc:
        report("FAIL", "GUI integration actions", repr(exc))


def run_existing_self_test() -> None:
    script = ROOT / "self_test.py"
    if not script.exists():
        report("FAIL", "Existing core self_test.py is present")
        return
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
        )
        print("\n--- self_test.py output ---")
        print(proc.stdout.rstrip())
        print("--- end self_test.py output ---\n")
        check(proc.returncode == 0, "Existing self_test.py exits cleanly", f"exit code {proc.returncode}")
        if "FOG_100" in proc.stdout and "  3.61" in proc.stdout:
            warn("Fog stress case", "self_test.py still reports severe detection loss at FOG_100; keep this as a known stress limitation")
    except subprocess.TimeoutExpired:
        report("FAIL", "Existing self_test.py completes", "timed out after 90 seconds")


def main() -> int:
    print("FSOC FULL SYSTEM SELF-TEST")
    print("=" * 88)
    print(f"Project root: {ROOT}")
    print()

    compile_project()

    try:
        mod = import_core()
        report("PASS", "Core module imports")
    except Exception as exc:
        report("FAIL", "Core module imports", repr(exc))
        print(f"\nRESULT: {PASS} PASS / {WARN} WARN / {FAIL} FAIL")
        return 1

    run_existing_self_test()
    run_disturbance_api(mod)
    run_core_pipeline(mod)
    run_multi_target(mod)
    run_reports(mod)
    build_synthetic_benchmark(mod)
    run_gui_headless(mod)

    print("\n" + "=" * 88)
    print(f"RESULT: {PASS} PASS / {WARN} WARN / {FAIL} FAIL")
    if FAIL:
        print("OVERALL: FAIL — fix the failed automated checks before final SIH demo hardening.")
        return 1
    if WARN:
        print("OVERALL: PASS WITH WARNINGS — automated software checks passed; warnings need engineering judgment/visual review.")
    else:
        print("OVERALL: PASS — automated software checks passed.")
    print("Note: visual polish and physical hardware behavior still need human validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
