"""Headless verification for FSOC Refinement 1.

Checks that each designated target can be selected and followed by the same
CV + Kalman + pan/tilt closed loop used by the GUI.
"""
from __future__ import annotations

import math

import config
from controller import CameraController
from detector import detect_beacon
from simulator import Simulator
from tracker import BeaconTracker


def run_target(target_id: str, frames: int = 120):
    sim = Simulator()
    sim.set_target_count(5)
    sim.set_designated_target(target_id)

    controller = CameraController()
    controller.set_angle_limits(5.0, 3.75)
    controller.set_gain(2.0)

    tracker = BeaconTracker()
    errors = []
    states = []
    candidate_counts = []

    for _ in range(frames):
        frame = sim.get_frame()
        expected = (tracker.x, tracker.y) if tracker.x is not None else None
        detected, details = detect_beacon(
            frame,
            return_details=True,
            expected_position=expected,
            preferred_signature=sim.get_designated_target_signature(),
        )
        candidate_counts.append(len(details.get("candidates", [])))

        tracked = tracker.update(detected)
        if tracked is None:
            states.append("SEARCHING")
            continue

        x, y, state = tracked
        states.append(state)
        error = math.hypot(x - config.FRAME_WIDTH / 2, y - config.FRAME_HEIGHT / 2)
        if state == "TRACKING" and len(states) > 20:
            errors.append(error)

        if state in {"TRACKING", "RE-ACQUIRING", "PREDICTING"}:
            pan, tilt = controller.update(
                x - config.FRAME_WIDTH / 2,
                y - config.FRAME_HEIGHT / 2,
            )
            sim.update_camera(pan, tilt)

    return {
        "target": target_id,
        "final_state": tracker.state,
        "avg_error": sum(errors) / max(len(errors), 1),
        "max_error": max(errors, default=0.0),
        "max_candidates": max(candidate_counts, default=0),
    }


def main():
    print("FSOC REFINEMENT 1 — CLOSED-LOOP MULTI-TARGET TEST")
    print("=" * 70)
    failed = []
    for target_id in ["TGT-01", "TGT-02", "TGT-03", "TGT-04", "TGT-05"]:
        result = run_target(target_id)
        print(
            f"{result['target']:7s}  state={result['final_state']:12s}  "
            f"avg error={result['avg_error']:6.2f} px  "
            f"max error={result['max_error']:6.2f} px  "
            f"candidates={result['max_candidates']}"
        )
        if result["final_state"] != "TRACKING" or result["avg_error"] > 10.0:
            failed.append(result["target"])

    if failed:
        raise SystemExit(f"FAIL: closed-loop tracking failed for {', '.join(failed)}")
    print("\nPASS: all 5 designated targets can be detected and camera-follow tracked.")


if __name__ == "__main__":
    main()
