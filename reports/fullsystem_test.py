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
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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
        warn("Core nominal alignment", f"max closed-loop error {max(errors):.2f} px; keep stress results separate")


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
        statuses = [v.get("status") for v in result.values() if isinstance(v, dict) and "status" in v]
        check(bool(statuses) and all(x == "PASS" for x in statuses), "PS requirement evaluator", str(statuses))
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
