"""Headless validation for FSOC Refinement 1: multiple moving targets."""
from __future__ import annotations

from detector import detect_beacon
from simulator import Simulator


def run_case(count: int, selected: str):
    sim = Simulator()
    sim.set_target_count(count)
    assert selected in sim.get_target_ids()
    sim.set_designated_target(selected)

    for _ in range(30):
        frame = sim.get_frame()
        detection, info = detect_beacon(
            frame,
            return_details=True,
            preferred_signature=sim.get_designated_target_signature(),
        )
        assert frame.shape == (480, 640, 3)
        assert len(sim.get_target_states()) == count
        assert info["candidate_count"] >= count, (count, selected, info)
        assert detection is not None, (count, selected, info)
        assert info["signature_score"] >= 0.75, (count, selected, info)

    return detection, info


def main():
    print("FSOC REFINEMENT 1 SELF-TEST")
    print("=" * 72)
    for count in (2, 3, 4, 5):
        detection, info = run_case(count, "TGT-01")
        print(f"TARGETS={count}  DESIGNATED=TGT-01  CANDIDATES={info['candidate_count']}  "
              f"SIGNATURE={info['signature_score']:.2f}  AI={info['ai_score']:.2f}  DET={detection}")

    # Verify that the tracker input can switch to a non-default designated target.
    for selected in ("TGT-02", "TGT-03", "TGT-04", "TGT-05"):
        detection, info = run_case(5, selected)
        print(f"TARGETS=5  DESIGNATED={selected}  CANDIDATES={info['candidate_count']}  "
              f"SIGNATURE={info['signature_score']:.2f}  AI={info['ai_score']:.2f}  DET={detection}")

    print("\nPASS: multiple objects render, candidates are returned, and the designated signature is selected.")


if __name__ == "__main__":
    main()
