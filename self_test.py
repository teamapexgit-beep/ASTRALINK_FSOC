"""Headless smoke/coverage test for the FSOC core.

Run with:
    python self_test.py

This checks the PS disturbance generators, detection/tracking pipeline, and
basic parameter controls without opening the PySide6 GUI.
"""
from __future__ import annotations

import math
import time

import config
from controller import CameraController
from detector import detect_beacon
from simulator import Simulator
from tracker import BeaconTracker


def run_case(name, noise="None", noise_level=0, turbulence=0, jitter=0,
             platform=0, atmosphere="Clear", atmosphere_level=0, frames=120):
    sim = Simulator()
    sim.set_motion_pattern("Figure 8")
    sim.set_beacon_speed(4.0)
    sim.set_beacon_range(1000.0)
    sim.set_beacon_size(10.0)
    sim.set_noise_type(noise)
    sim.noise_level = noise_level
    sim.set_disturbances(
        turbulence_enabled=turbulence > 0,
        turbulence_strength_px=turbulence,
        camera_jitter_enabled=jitter > 0,
        camera_jitter_px=jitter,
        platform_motion_enabled=platform > 0,
        platform_motion_px=platform,
        platform_motion_pattern="Linear",
        atmosphere=atmosphere,
        atmosphere_level=atmosphere_level,
    )

    controller = CameraController()
    controller.set_angle_limits(5.0, 5.0)
    controller.set_gain(2.0)
    config.MAX_PAN_SPEED = 5.0
    config.MAX_TILT_SPEED = 5.0

    tracker = BeaconTracker()
    errors = []
    centroid_errors = []
    detection_count = 0
    visible_count = 0
    started = time.perf_counter()

    for _ in range(frames):
        frame = sim.get_frame()
        truth = sim.get_ground_truth_image_position()
        expected = (tracker.x, tracker.y) if tracker.x is not None else None
        detection = detect_beacon(frame, expected_position=expected)
        tracked = tracker.update(detection)

        if truth is not None:
            visible_count += 1
        if detection is not None:
            detection_count += 1
            if truth is not None:
                centroid_errors.append(math.hypot(
                    detection[0] - truth[0], detection[1] - truth[1]
                ))

        if tracked is not None:
            x, y, state = tracked
            if state == "TRACKING":
                errors.append(math.hypot(x - 320.0, y - 240.0))
            lead = 1.0 / config.UPDATE_RATE
            predicted_x = x + tracker.velocity_x * lead * 0.55
            predicted_y = y + tracker.velocity_y * lead * 0.55
            pan, tilt = controller.update(predicted_x - 320.0, predicted_y - 240.0)
            sim.update_camera(pan, tilt)

    elapsed = max(time.perf_counter() - started, 1e-9)
    result = {
        "name": name,
        "detection_rate_pct": 100.0 * detection_count / max(visible_count, 1),
        "avg_centroid_error_px": sum(centroid_errors) / len(centroid_errors) if centroid_errors else 0.0,
        "max_centroid_error_px": max(centroid_errors) if centroid_errors else 0.0,
        "avg_tracking_error_px": sum(errors) / len(errors) if errors else 0.0,
        "max_tracking_error_px": max(errors) if errors else 0.0,
        "core_fps": frames / elapsed,
    }
    return result


def main():
    cases = [
        run_case("BASELINE"),
        run_case("SALT_PEPPER_10", noise="Salt & Pepper", noise_level=10),
        run_case("GAUSSIAN_20", noise="Gaussian", noise_level=20),
        run_case("POISSON_20", noise="Poisson", noise_level=20),
        run_case("TURBULENCE_20", turbulence=20),
        run_case("CAMERA_JITTER_20", jitter=20),
        run_case("PLATFORM_MOTION_20", platform=20),
        run_case("FOG_100", atmosphere="Fog", atmosphere_level=100),
        run_case("LOW_LIGHT_100", atmosphere="Low Light", atmosphere_level=100),
    ]

    print("FSOC CORE SELF-TEST")
    print("=" * 88)
    print(f"{'CASE':22} {'DET%':>8} {'CENT AVG':>10} {'CENT MAX':>10} {'TRACK AVG':>11} {'TRACK MAX':>11} {'FPS':>8}")
    print("-" * 88)
    for r in cases:
        print(
            f"{r['name']:22} {r['detection_rate_pct']:8.2f} "
            f"{r['avg_centroid_error_px']:10.2f} {r['max_centroid_error_px']:10.2f} "
            f"{r['avg_tracking_error_px']:11.2f} {r['max_tracking_error_px']:11.2f} {r['core_fps']:8.1f}"
        )

    print("\nInterpretation:")
    print("- This test verifies that disturbances are actually inserted before detection.")
    print("- Worst-case ±20 px jitter/turbulence can legitimately increase alignment error;")
    print("  use the benchmark report rather than changing errors artificially.")
    print("- The SIH target is ≥20 FPS, ≤10 px tracking error, <5% loss, ≤2 s acquisition,")
    print("  and ≤1 s re-acquisition; verify these on your final scenario and benchmark video.")


if __name__ == "__main__":
    main()
