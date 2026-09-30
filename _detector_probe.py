"""Raw detector probe: render frames under each preset and inspect what the
detector actually sees (candidates, mode, identity rejection, beacon pixels).

Run:  python _detector_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import config  # noqa: E402
import simulator  # noqa: E402
from detector import detect_beacon  # noqa: E402

PRESETS = ["Clear", "Sensor Noise", "Vibration", "Atmosphere", "Full Combined"]


def beacon_stats(frame):
    """Find the green beacon pixels (the simulator renders GREEN-SQUARE)."""
    b, g, r = frame[:, :, 0].astype(int), frame[:, :, 1].astype(int), frame[:, :, 2].astype(int)
    mask = (g > r + 20) & (g > b + 8) & (g > 60)
    n = int(mask.sum())
    if n == 0:
        return 0, None, None
    ys, xs = np.nonzero(mask)
    return n, float(xs.mean()), float(ys.mean())


for preset in PRESETS:
    sim = simulator.Simulator()
    sim.set_target_count(1)
    sim.set_designated_target("TGT-01")
    sim.set_disturbance_preset(preset)

    det_hits = 0
    modes = {}
    ident_rej = 0
    no_cand = 0
    gate_rej = 0
    px_min, px_max, px_sum = 10**9, 0, 0
    beacon_vis = 0
    for i in range(90):
        sim.update_beacon()
        frame = sim.get_frame()
        npx, bx, by = beacon_stats(frame)
        if npx > 0:
            beacon_vis += 1
            px_min, px_max = min(px_min, npx), max(px_max, npx)
            px_sum += npx
        det, info = detect_beacon(
            frame, return_details=True,
            expected_position=None,  # raw detection, no track assist
            preferred_signature=sim.get_designated_target_signature(),
        )
        if det is not None:
            det_hits += 1
        mode = info.get("mode", "?")
        modes[mode] = modes.get(mode, 0) + 1
        if info.get("identity_rejected"):
            ident_rej += 1
        if info.get("candidate_count", 0) == 0:
            no_cand += 1

    print(f"\n=== {preset} ===")
    print(f"  raw detect: {det_hits}/90   identity_rejected: {ident_rej}   zero-candidate frames: {no_cand}")
    print(f"  modes: {modes}")
    print(f"  beacon pixels visible in {beacon_vis}/90 frames"
          + (f"  (min={px_min}, max={px_max}, mean={px_sum/max(beacon_vis,1):.0f})" if beacon_vis else ""))
