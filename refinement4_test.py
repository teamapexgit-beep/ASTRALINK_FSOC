"""Refinement 4 self-test: create a small synthetic MP4 and benchmark it."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

from simulator import Simulator
from benchmark_video import run_video_benchmark


def main():
    work = Path("reports")
    work.mkdir(exist_ok=True)
    video_path = work / "refinement4_synthetic.mp4"
    gt_path = work / "refinement4_synthetic_groundtruth.csv"

    sim = Simulator()
    sim.set_target_count(1)
    sim.set_designated_target("TGT-01")
    sim.set_motion_pattern("Figure 8")
    sim.set_noise_type("None")

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (640, 480),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic MP4")

    with gt_path.open("w", newline="", encoding="utf-8") as handle:
        gt_writer = csv.writer(handle)
        gt_writer.writerow(["frame", "x", "y"])
        for frame_no in range(120):
            frame = sim.get_frame()
            gt = sim.get_ground_truth_image_position()
            writer.write(frame)
            if gt is not None:
                gt_writer.writerow([frame_no + 1, f"{gt[0]:.4f}", f"{gt[1]:.4f}"])
    writer.release()

    summary, paths = run_video_benchmark(
        video_path,
        output_dir=work,
        preferred_signature="GREEN-SQUARE",
        ground_truth_path=gt_path,
    )

    print("FSOC REFINEMENT 4 MP4 BENCHMARK SELF-TEST")
    print("=" * 72)
    print(f"Frames processed: {summary.frames_processed}")
    print(f"Ground truth:     {summary.ground_truth_available}")
    print(f"Processed FPS:    {summary.fps:.2f}")
    print(f"Detection loss:   {summary.target_loss_percentage:.2f}%")
    print(f"Avg centroid:     {summary.average_centroid_error_px:.2f} px")
    print(f"Max centroid:     {summary.maximum_centroid_error_px:.2f} px")
    print(f"Tracked MP4:      {paths[3]}")
    print(f"Telemetry CSV:    {paths[2]}")
    print(f"Summary JSON:     {paths[0]}")
    print(f"Summary CSV:      {paths[1]}")

    assert summary.frames_processed == 120, "Benchmark did not process all synthetic frames"
    assert summary.ground_truth_available, "Ground truth was not loaded"
    assert paths[2].exists(), "Telemetry CSV missing"
    assert paths[3].exists(), "Tracked MP4 missing"
    assert paths[0].exists() and paths[1].exists(), "Summary report missing"
    assert summary.fps >= 20.0, "Benchmark throughput fell below 20 FPS"
    print("REFINEMENT 4 CORE TEST: PASS")


if __name__ == "__main__":
    main()
