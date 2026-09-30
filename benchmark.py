"""MP4 benchmark engine for FSOC coarse pointing.

The benchmark uses the same detector and Kalman tracker used by the simulation.
For a video input, the PTZ/camera itself cannot be physically repositioned, so
its virtual pan/tilt command is recorded alongside the tracking result and
rendered into the output video.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config
from controller import CameraController
from detector import detect_beacon
from performance_report import PerformanceSummary, save_performance_report
from tracker import BeaconTracker


@dataclass
class BenchmarkArtifacts:
    summary: PerformanceSummary
    json_path: Path
    summary_csv_path: Path
    telemetry_csv_path: Path
    tracked_video_path: Path


def _read_ground_truth(path: Path | None):
    if path is None or not path.exists():
        return None
    truth = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "x", "y"}
        if not required.issubset({(h or "").strip().lower() for h in (reader.fieldnames or [])}):
            raise ValueError("Ground-truth CSV must contain columns: frame,x,y")
        for row in reader:
            try:
                frame_no = int(row.get("frame", row.get("Frame", "")))
                x = float(row.get("x", row.get("X", "")))
                y = float(row.get("y", row.get("Y", "")))
            except (TypeError, ValueError):
                continue
            truth[frame_no] = (x, y)
    return truth


def _default_ground_truth(video_path: Path) -> Path:
    return video_path.with_name(video_path.stem + "_groundtruth.csv")


def _fourcc_candidates():
    return [cv2.VideoWriter_fourcc(*code) for code in ("mp4v", "avc1", "MJPG")]


def run_video_benchmark(
    video_path,
    output_dir="reports",
    preferred_signature=None,
    ground_truth_path=None,
    write_output_video=True,
):
    video_path = Path(video_path)
    if not video_path.exists():
        raise OSError(f"Video not found: {video_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1.0 or not math.isfinite(source_fps):
        source_fps = float(config.UPDATE_RATE)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or config.FRAME_WIDTH)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.FRAME_HEIGHT)
    frame_count_hint = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Temporarily use source dimensions for the geometry expected by the
    # detector/controller without changing the saved application defaults.
    old_w, old_h = config.FRAME_WIDTH, config.FRAME_HEIGHT
    old_update = config.UPDATE_RATE
    config.FRAME_WIDTH = width
    config.FRAME_HEIGHT = height
    config.UPDATE_RATE = max(1, int(round(source_fps)))

    controller = CameraController()
    tracker = BeaconTracker()
    controller.set_angle_limits(5.0, 3.75)
    controller.set_gain(1.5)

    gt_path = Path(ground_truth_path) if ground_truth_path else _default_ground_truth(video_path)
    ground_truth = _read_ground_truth(gt_path) if gt_path.exists() else None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_stem = video_path.stem.replace(" ", "_")
    telemetry_path = out_dir / f"{safe_stem}_tracked_telemetry_{stamp}.csv"
    tracked_video_path = out_dir / f"{safe_stem}_tracked_{stamp}.mp4"

    writer = None
    if write_output_video:
        fps_for_writer = max(source_fps, 1.0)
        for fourcc in _fourcc_candidates():
            candidate = cv2.VideoWriter(str(tracked_video_path), fourcc, fps_for_writer, (width, height))
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
        if writer is None:
            config.FRAME_WIDTH, config.FRAME_HEIGHT, config.UPDATE_RATE = old_w, old_h, old_update
            capture.release()
            raise RuntimeError("Could not create an MP4 output writer on this system.")

    telemetry_fields = [
        "frame", "timestamp_s", "source_fps", "detected", "target_x", "target_y",
        "predicted_x", "predicted_y", "center_error_px", "centroid_error_px",
        "pan_deg", "tilt_deg", "fps_running", "state", "filter_confidence",
        "detector_confidence", "ai_confidence", "signature_score", "acquisition_s",
        "loss_event", "reacquisition_event", "reacquisition_time_s", "link_ready",
    ]

    frame_no = 0
    processing_times = []
    detections = 0
    tracking_frames = 0
    visible_frames = 0
    loss_events = 0
    successful_reacq = 0
    reacq_times = []
    in_loss = False
    loss_start_time = None
    first_tracking_time = None
    lock_streak = 0
    link_ready_streak = 0
    error_values = []
    centroid_values = []

    with telemetry_path.open("w", newline="", encoding="utf-8") as telemetry_file:
        telemetry_writer = csv.DictWriter(telemetry_file, fieldnames=telemetry_fields)
        telemetry_writer.writeheader()

        run_start = time.perf_counter()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_no += 1
                frame_start = time.perf_counter()
                timestamp_s = (frame_no - 1) / max(source_fps, 1.0)

                expected_position = None
                if tracker.state_vector is not None and tracker.x is not None:
                    expected_position = (float(tracker.x), float(tracker.y))

                detection, info = detect_beacon(
                    frame,
                    return_details=True,
                    expected_position=expected_position,
                    preferred_signature=preferred_signature,
                )
                tracked = tracker.update(detection)
                detected = detection is not None
                detections += int(detected)

                gt = ground_truth.get(frame_no) if ground_truth is not None else None
                if gt is not None:
                    visible_frames += 1
                elif ground_truth is None:
                    visible_frames += 1

                center_error = None
                centroid_error = None
                predicted_x = predicted_y = None
                state = tracker.state

                if tracked is not None:
                    x, y, state = tracked
                    center_error = math.hypot(x - width / 2.0, y - height / 2.0)
                    error_values.append(center_error)
                    tracking_frames += int(state in {"TRACKING", "RE-ACQUIRING", "PREDICTING"})
                    lead_dt = 1.0 / max(float(config.UPDATE_RATE), 1.0)
                    predicted_x = float(x) + tracker.velocity_x * lead_dt
                    predicted_y = float(y) + tracker.velocity_y * lead_dt
                    pan, tilt = controller.update(
                        predicted_x - width / 2.0,
                        predicted_y - height / 2.0,
                    ) if state in {"TRACKING", "RE-ACQUIRING", "PREDICTING"} else (controller.pan, controller.tilt)
                else:
                    pan, tilt = controller.pan, controller.tilt

                if gt is not None and detection is not None:
                    centroid_error = math.hypot(detection[0] - gt[0], detection[1] - gt[1])
                    centroid_values.append(centroid_error)

                if first_tracking_time is None and state == "TRACKING":
                    first_tracking_time = timestamp_s

                if detection is None:
                    if not in_loss:
                        in_loss = True
                        loss_start_time = timestamp_s
                        loss_events += 1
                elif in_loss:
                    successful_reacq += 1
                    reacq_time = timestamp_s - (loss_start_time if loss_start_time is not None else timestamp_s)
                    reacq_times.append(reacq_time)
                    in_loss = False
                    loss_start_time = None

                if state == "TRACKING" and center_error is not None and center_error <= 10.0:
                    lock_streak += 1
                else:
                    lock_streak = 0
                if state == "TRACKING" and center_error is not None and center_error <= 10.0 and info.get("ai_score", 0.0) >= config.LINK_READY_AI_THRESHOLD:
                    link_ready_streak += 1
                else:
                    link_ready_streak = 0
                link_ready = link_ready_streak >= config.LINK_READY_STREAK_FRAMES

                elapsed = time.perf_counter() - frame_start
                processing_times.append(elapsed)
                running_duration = max(time.perf_counter() - run_start, 1e-9)
                running_fps = frame_no / running_duration

                # Evidence overlay on the saved video.
                overlay = frame.copy()
                cv2.line(overlay, (width // 2 - 12, height // 2), (width // 2 + 12, height // 2), (255, 255, 255), 1)
                cv2.line(overlay, (width // 2, height // 2 - 12), (width // 2, height // 2 + 12), (255, 255, 255), 1)
                if detection is not None:
                    cv2.circle(overlay, (int(detection[0]), int(detection[1])), 10, (60, 255, 90), 2)
                    cv2.drawMarker(overlay, (int(detection[0]), int(detection[1])), (60, 255, 90), cv2.MARKER_CROSS, 15, 1)
                if predicted_x is not None:
                    cv2.circle(overlay, (int(round(predicted_x)), int(round(predicted_y))), 8, (0, 220, 255), 2)
                lines = [
                    f"STATE: {state}",
                    f"FPS: {running_fps:.1f}",
                    f"CENTER ERROR: {center_error:.1f} px" if center_error is not None else "CENTER ERROR: --",
                    f"PAN/TILT: {pan:.2f} / {tilt:.2f} deg",
                    f"DETECT: {info.get('confidence', 0.0):.2f}  AI: {info.get('ai_score', 0.0):.2f}",
                    "LINK READY" if link_ready else "LINK NOT READY",
                ]
                y_text = 22
                for text in lines:
                    cv2.putText(overlay, text, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 240, 245), 1, cv2.LINE_AA)
                    y_text += 20
                if writer is not None:
                    writer.write(overlay)

                telemetry_writer.writerow({
                    "frame": frame_no,
                    "timestamp_s": f"{timestamp_s:.6f}",
                    "source_fps": f"{source_fps:.3f}",
                    "detected": int(detected),
                    "target_x": f"{detection[0]:.3f}" if detection is not None else "",
                    "target_y": f"{detection[1]:.3f}" if detection is not None else "",
                    "predicted_x": f"{predicted_x:.3f}" if predicted_x is not None else "",
                    "predicted_y": f"{predicted_y:.3f}" if predicted_y is not None else "",
                    "center_error_px": f"{center_error:.3f}" if center_error is not None else "",
                    "centroid_error_px": f"{centroid_error:.3f}" if centroid_error is not None else "",
                    "pan_deg": f"{pan:.5f}",
                    "tilt_deg": f"{tilt:.5f}",
                    "fps_running": f"{running_fps:.3f}",
                    "state": state,
                    "filter_confidence": f"{tracker.confidence:.5f}",
                    "detector_confidence": f"{info.get('confidence', 0.0):.5f}",
                    "ai_confidence": f"{info.get('ai_score', 0.0):.5f}",
                    "signature_score": f"{info.get('signature_score', 0.0):.5f}",
                    "acquisition_s": f"{first_tracking_time:.5f}" if first_tracking_time is not None else "",
                    "loss_event": int(detection is None and in_loss and loss_start_time == timestamp_s),
                    "reacquisition_event": int(detected and not in_loss and successful_reacq > 0 and abs((timestamp_s - (reacq_times[-1] if reacq_times else 0.0)) - timestamp_s) < 1e-9),
                    "reacquisition_time_s": f"{reacq_times[-1]:.5f}" if reacq_times and detected and not in_loss else "",
                    "link_ready": int(link_ready),
                })
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            config.FRAME_WIDTH, config.FRAME_HEIGHT, config.UPDATE_RATE = old_w, old_h, old_update

    duration = max(time.perf_counter() - run_start, 1e-9)
    processed_fps = frame_no / duration if frame_no else 0.0
    avg_err = sum(error_values) / len(error_values) if error_values else 0.0
    max_err = max(error_values, default=0.0)
    rmse = math.sqrt(sum(e * e for e in error_values) / len(error_values)) if error_values else 0.0
    avg_centroid = sum(centroid_values) / len(centroid_values) if centroid_values else 0.0
    max_centroid = max(centroid_values, default=0.0)
    lock_retention = 100.0 * tracking_frames / max(visible_frames, 1)
    loss_pct = 100.0 * (visible_frames - detections) / max(visible_frames, 1)
    avg_proc_ms = (sum(processing_times) / len(processing_times) * 1000.0) if processing_times else 0.0
    max_proc_ms = (max(processing_times) * 1000.0) if processing_times else 0.0

    summary = PerformanceSummary(
        frames_processed=frame_no,
        duration_s=duration,
        fps=processed_fps,
        acquisition_time_s=first_tracking_time,
        average_tracking_error_px=avg_err,
        maximum_tracking_error_px=max_err,
        rmse_tracking_error_px=rmse,
        average_centroid_error_px=avg_centroid,
        maximum_centroid_error_px=max_centroid,
        lock_retention_percentage=lock_retention,
        target_loss_percentage=loss_pct,
        target_loss_events=loss_events,
        successful_reacquisitions=successful_reacq,
        average_reacquisition_time_s=(sum(reacq_times) / len(reacq_times)) if reacq_times else None,
        average_processing_time_ms=avg_proc_ms,
        max_processing_time_ms=max_proc_ms,
        ground_truth_available=ground_truth is not None,
        source=f"benchmark video: {video_path.name}",
        metadata={
            "video_path": str(video_path),
            "source_fps": source_fps,
            "source_width": width,
            "source_height": height,
            "frame_count_hint": frame_count_hint,
            "preferred_signature": preferred_signature or "AUTO",
            "ground_truth_path": str(gt_path) if gt_path.exists() else None,
            "output_video": str(tracked_video_path) if writer is not None else None,
            "telemetry_csv": str(telemetry_path),
        },
    )
    json_path, summary_csv = save_performance_report(summary, output_dir=out_dir, prefix=f"{safe_stem}_benchmark")
    return summary, (json_path, summary_csv, telemetry_path, tracked_video_path)
