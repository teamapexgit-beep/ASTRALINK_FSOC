from __future__ import annotations
import config
from controller import CameraController
from detector import detect_beacon
from simulator import Simulator
from tracker import BeaconTracker


def run_identity_rejection():
    sim = Simulator()
    sim.set_target_count(5)
    sim.set_designated_target("TGT-01")
    sim.set_beacon_loss(0, 5)
    frame = sim.get_frame()
    detection, info = detect_beacon(
        frame,
        return_details=True,
        preferred_signature=sim.get_designated_target_signature(),
    )
    assert detection is None, "A decoy was incorrectly accepted while the designated target was hidden"
    assert info.get("identity_rejected") is True
    print("IDENTITY REJECTION: PASS")


def run_loss_recovery(frames=240, loss_start=90, loss_duration=18):
    sim = Simulator()
    sim.set_target_count(3)
    sim.set_designated_target("TGT-01")
    sim.set_motion_pattern("Straight Line")
    sim.set_beacon_loss(loss_start, loss_duration)

    ctl = CameraController()
    ctl.set_angle_limits(5.0, 3.75)
    ctl.set_gain(2.0)
    tr = BeaconTracker()

    loss_frame = None
    reacq_frame = None
    predicting_seen = False
    lost_seen = False

    for f in range(frames):
        frame = sim.get_frame()
        expected = (
            (float(tr.x), float(tr.y))
            if tr.state_vector is not None and tr.x is not None
            else None
        )
        det, _ = detect_beacon(
            frame,
            return_details=True,
            expected_position=expected,
            preferred_signature=sim.get_designated_target_signature(),
        )
        out = tr.update(det)

        if tr.state == "PREDICTING":
            predicting_seen = True
            if loss_frame is None:
                loss_frame = f
        if tr.state == "LOST":
            lost_seen = True
            if loss_frame is None:
                loss_frame = f

        if (
            loss_frame is not None
            and f >= loss_start + loss_duration
            and tr.state in {"RE-ACQUIRING", "TRACKING"}
            and reacq_frame is None
        ):
            reacq_frame = f

        if out is not None:
            x, y, state = out
            if state in {"TRACKING", "RE-ACQUIRING", "PREDICTING"}:
                lead = 1.0 / config.UPDATE_RATE
                px = x + tr.velocity_x * lead
                py = y + tr.velocity_y * lead
                pan, tilt = ctl.update(
                    px - config.FRAME_WIDTH / 2,
                    py - config.FRAME_HEIGHT / 2,
                )
                sim.update_camera(pan, tilt)

    reacq_time = (
        (reacq_frame - loss_frame) / config.UPDATE_RATE
        if reacq_frame is not None and loss_frame is not None
        else None
    )
    print("PREDICTING seen:", predicting_seen)
    print("LOST seen:", lost_seen)
    print("Re-acquisition:", f"{reacq_time:.3f} s" if reacq_time is not None else "FAILED")
    assert predicting_seen, "Prediction phase not reached"
    assert lost_seen, "Search-after-prediction phase not reached"
    assert reacq_time is not None and reacq_time <= 1.0, "Re-acquisition exceeded 1 s"
    print("RECOVERY: PASS")


if __name__ == "__main__":
    print("FSOC REFINEMENT 2 SELF-TEST")
    print("=" * 70)
    run_identity_rejection()
    run_loss_recovery()
