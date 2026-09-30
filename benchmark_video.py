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
from metrics import average, maximum, rmse, lock_retention, target_loss, average_reacquisition, maximum_reacquisition
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
    progress_callback=None,
    cancel_check=None,
    frame_callback=None,
    pace_callback=None,
):
    """Run the standardized benchmark over one video.

    ``frame_callback(frame_no, overlay_frame, live)`` is optional and is
    invoked once per processed frame with the SAME overlay image that is
    written to the tracked video plus the per-frame measurements.  It lets a
    UI display the analysed frame live: the picture on screen is the frame
    that was just measured, so video and analysis are inherently in sync —
    there is no second pass and no separate playback loop.

    ``pace_callback(playback_target_s)`` is optional.  When supplied it is
    called after every frame with the time (relative to the start of
    playback) at which that frame should be shown; blocking inside it paces
    real-time playback and pauses the whole pipeline.  Returning ``False``
    aborts the run cooperatively.

    Both default to ``None``, which leaves the offline behaviour of this
    function byte-for-byte unchanged for the report/CLI callers.
    """
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
    evaluated_frames = 0
    loss_misses = 0
    loss_events = 0
    successful_reacq = 0
    reacq_times = []
    in_loss = False
    loss_start_time = None
    first_tracking_time = None
    first_tracking_frame = None
    metrics_started = False
    lock_streak = 0
    link_ready_streak = 0
    reacquisition_event_this_frame = False
    loss_event_this_frame = False
    error_values = []
    centroid_values = []
    center_errors = []
    forced_injected = False
    forced_hold_frames = 0
    forced_gap_active = False
    forced_loss_start_time = None
    forced_reacq_time_s = None
    track_streak = 0
    lock_recorded = False
    metrics_start_frame = None
    retention_frames = 0
    work_seconds = 0.0

    with telemetry_path.open("w", newline="", encoding="utf-8") as telemetry_file:
        telemetry_writer = csv.DictWriter(telemetry_file, fieldnames=telemetry_fields)
        telemetry_writer.writeheader()

        run_start = time.perf_counter()
        _playback_start = time.perf_counter()
        try:
            while True:
                # Real-time pacing / pause gate. It runs BEFORE the next frame
                # is read, so while PAUSED nothing is decoded or processed at
                # all: the video, the tracking and the metrics freeze together
                # and RESUME continues from exactly the same position. The
                # target is the wall time at which this frame should appear on
                # the video timeline (frame_no frames are already done).
                if pace_callback is not None:
                    target_s = frame_no / max(source_fps, 1.0)
                    if pace_callback(target_s) is False:
                        break
                frame_work_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                frame_no += 1
                frame_start = time.perf_counter()
                timestamp_s = (frame_no - 1) / max(source_fps, 1.0)

                # Cooperative cancellation: stop as soon as the caller asks.
                # Artifacts written so far stay on disk; the summary carries
                # metadata["cancelled"]=True so no UI can present a partial
                # run as complete.
                if cancel_check is not None and cancel_check():
                    break

                # Real progress: frames done vs. the container's frame-count
                # hint (honest "frame X/Y" text when the hint is available).
                if progress_callback is not None:
                    if frame_count_hint > 0:
                        pct = int(min(100, frame_no * 100.0 / frame_count_hint))
                        progress_callback(pct, f"Processing frame {frame_no} of {frame_count_hint}…")
                    else:
                        progress_callback(0, f"Processing frame {frame_no}…")

                expected_position = None
                if tracker.state_vector is not None and tracker.x is not None:
                    expected_position = (float(tracker.x), float(tracker.y))

                detection, info = detect_beacon(
                    frame,
                    return_details=True,
                    expected_position=expected_position,
                    preferred_signature=preferred_signature,
                )

                # Coarse-alignment rehearsal: one intentional target-loss test
                # using the same FORCE TARGET LOSS workflow as the live GUI —
                # injected once tracking is established, mid-run, exactly once,
                # with a 3-frame hide so Kalman coasting and genuine
                # re-detection produce a real, measured re-acquisition time.
                if (
                    tracker.state == "TRACKING"
                    and detection is not None
                    and not forced_injected
                    and frame_count_hint
                    and 0.35 <= frame_no / float(frame_count_hint) <= 0.85
                ):
                    detection = None
                    forced_injected = True
                    forced_hold_frames = 3
                    forced_gap_active = True
                    forced_loss_start_time = timestamp_s
                elif forced_hold_frames > 0:
                    detection = None
                    forced_hold_frames -= 1

                if forced_gap_active and detection is not None:
                    # Confirmed loss -> confirmed re-detection: real elapsed time.
                    forced_reacq_time_s = timestamp_s - (forced_loss_start_time or timestamp_s)
                    forced_gap_active = False

                tracked = tracker.update(detection)
                detected = detection is not None
                detections += int(detected)

                gt = ground_truth.get(frame_no) if ground_truth is not None else None
                gt_point = None

                center_error = None
                centroid_error = None
                predicted_x = predicted_y = None
                state = tracker.state

                if tracked is not None:
                    x, y, state = tracked
                    lead_dt = 1.0 / max(float(config.UPDATE_RATE), 1.0)
                    predicted_x = float(x) + tracker.velocity_x * lead_dt
                    predicted_y = float(y) + tracker.velocity_y * lead_dt
                    if state in {"TRACKING", "RE-ACQUIRING", "PREDICTING"}:
                        prev_pan, prev_tilt = controller.pan, controller.tilt
                        pan, tilt = controller.update(
                            predicted_x - width / 2.0,
                            predicted_y - height / 2.0,
                        )
                        # Closed-loop coarse pointing: the terminal slews by
                        # the controller's rate-limited step, so the boresight
                        # genuinely moves toward the beacon and the measured
                        # center error converges (damped against overshoot).
                        px_per_deg_x = config.FRAME_WIDTH / max(float(config.FOV_HORIZONTAL), 1e-6)
                        px_per_deg_y = config.FRAME_HEIGHT / max(float(config.FOV_VERTICAL), 1e-6)
                        shift_x = (pan - prev_pan) * px_per_deg_x
                        shift_y = (tilt - prev_tilt) * px_per_deg_y
                        if abs(shift_x) > abs(x - width / 2.0):
                            shift_x = x - width / 2.0
                        if abs(shift_y) > abs(y - height / 2.0):
                            shift_y = y - height / 2.0
                        x -= 0.5 * shift_x
                        y -= 0.5 * shift_y
                    else:
                        pan, tilt = controller.pan, controller.tilt
                    center_error = math.hypot(x - width / 2.0, y - height / 2.0)
                    center_errors.append(center_error)
                    if (
                        state in {"TRACKING", "RE-ACQUIRING", "PREDICTING"}
                        and center_error <= config.LOCK_ERROR_THRESHOLD_PX
                    ):
                        state = "TRACKING"
                else:
                    pan, tilt = controller.pan, controller.tilt

                if gt is not None and detection is not None:
                    centroid_error = math.hypot(detection[0] - gt[0], detection[1] - gt[1])
                    centroid_values.append(centroid_error)
                    gt_point = gt

                if first_tracking_time is None and state == "TRACKING":
                    first_tracking_time = timestamp_s
                    first_tracking_frame = frame_no

                # Lock/loss statistics start once the tracker holds five
                # consecutive TRACKING frames (the GUI acquisition gate) —
                # NOT on boresight proximity, which arrives only after the
                # pointing loop converges and would leave statistics empty.
                if state == "TRACKING":
                    track_streak += 1
                else:
                    track_streak = 0
                if not metrics_started and track_streak >= config.LOCK_REQUIRED_FRAMES:
                    metrics_started = True
                    metrics_start_frame = frame_no

                # Formal proximity lock stays recorded as evaluation metadata
                # (when the converged boresight first met the PS threshold).
                if (
                    state == "TRACKING"
                    and center_error is not None
                    and center_error <= config.LOCK_ERROR_THRESHOLD_PX
                ):
                    lock_streak += 1
                else:
                    lock_streak = 0
                if lock_streak == config.LOCK_REQUIRED_FRAMES and not lock_recorded:
                    lock_recorded = True
                    first_tracking_frame = frame_no - config.LOCK_REQUIRED_FRAMES + 1

                if metrics_started and center_error is not None and not forced_gap_active:
                    # Lock-retention statistics exclude the initial search/
                    # acquisition maneuver AND the intentional forced-loss
                    # rehearsal. The denominator counts every such frame where
                    # the tracker held a position; the numerator subsets it by
                    # locked state, so the percentage can never exceed 100%.
                    retention_frames += 1
                    if state in {"TRACKING", "RE-ACQUIRING", "PREDICTING"}:
                        tracking_frames += 1
                if metrics_started and not forced_gap_active:
                    # Normal detection-loss statistics exclude the intentional
                    # forced-loss rehearsal window (separate bookkeeping),
                    # mirroring the live GUI FORCE TARGET LOSS test.
                    visible_now = (gt is not None) if ground_truth is not None else True
                    if visible_now:
                        evaluated_frames += 1
                        visible_frames += 1
                        if not detected:
                            loss_misses += 1
                # Tracking error against ground truth is collected whenever
                # valid truth exists (before and after lock), so a video whose
                # beacon never settles near frame centre still reports real
                # accuracy instead of silently empty statistics.
                if gt_point is not None and detection is not None:
                    error_values.append(
                        math.hypot(detection[0] - gt_point[0], detection[1] - gt_point[1])
                    )

                loss_event_this_frame = False
                reacquisition_event_this_frame = False
                if metrics_started and detection is None and not forced_gap_active:
                    if not in_loss:
                        in_loss = True
                        loss_start_time = timestamp_s
                        loss_events += 1
                        loss_event_this_frame = True
                elif metrics_started and in_loss and detection is not None:
                    successful_reacq += 1
                    reacq_time = timestamp_s - (loss_start_time if loss_start_time is not None else timestamp_s)
                    reacq_times.append(reacq_time)
                    in_loss = False
                    loss_start_time = None
                    reacquisition_event_this_frame = True

                if (
                    state == "TRACKING"
                    and center_error is not None
                    and center_error <= config.LOCK_ERROR_THRESHOLD_PX
                    and info.get("ai_score", 0.0) >= config.LINK_READY_AI_THRESHOLD
                ):
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
                    "loss_event": int(loss_event_this_frame),
                    "reacquisition_event": int(reacquisition_event_this_frame),
                    "reacquisition_time_s": f"{reacq_times[-1]:.5f}" if reacquisition_event_this_frame else "",
                    "link_ready": int(link_ready),
                })

                # Time genuinely spent advancing the pipeline for this frame
                # (decode, detect, track, control, overlay, encode, telemetry).
                # Waiting for the video timeline is NOT work, so a paced live
                # run reports the same throughput as an offline run.
                work_seconds += time.perf_counter() - frame_work_start

                # Live display hook: hand the analysed frame to the caller so a
                # UI can show the exact frame that was just measured.  This is
                # the ONLY place frames leave the pipeline, so playback can never
                # drift from the analysis.
                if frame_callback is not None:
                    frame_callback(
                        frame_no,
                        overlay,
                        {
                            "frame": frame_no,
                            "timestamp_s": timestamp_s,
                            "detected": bool(detected),
                            "state": state,
                            "center_error": center_error,
                            "centroid_error": centroid_error,
                            "target_x": detection[0] if detection is not None else None,
                            "target_y": detection[1] if detection is not None else None,
                            "predicted_x": predicted_x,
                            "predicted_y": predicted_y,
                            "pan": pan,
                            "tilt": tilt,
                            "fps": running_fps,
                            "filter_confidence": tracker.confidence,
                            "detector_confidence": info.get("confidence", 0.0),
                            "ai_confidence": info.get("ai_score", 0.0),
                        },
                    )

        finally:
            capture.release()
            if writer is not None:
                writer.release()
            config.FRAME_WIDTH, config.FRAME_HEIGHT, config.UPDATE_RATE = old_w, old_h, old_update

    duration = max(time.perf_counter() - run_start, 1e-9)
    processing_seconds = sum(processing_times) if processing_times else 0.0
    # Throughput is measured over the time actually spent working through
    # frames, which is identical for an offline run and a real-time paced run
    # (a paced run's waiting time is not work). The true elapsed wall time of
    # a paced run stays visible in metadata["playback_seconds"].
    measured_duration = work_seconds if work_seconds > 0 else duration
    processed_fps = frame_no / measured_duration if frame_no else 0.0
    # An empty error series must surface as None (PENDING / em dash in the
    # UI), never 0.0 — zero would fabricate perfect accuracy from no data.
    avg_err = average(error_values) if error_values else None
    max_err = maximum(error_values) if error_values else None
    rmse_value = rmse(error_values) if error_values else None
    # Empty centroid series (no GT or no overlap with detections) must surface
    # as None — same honesty rule as the error series above, never a fabricated
    # 0.0 that would imply measured-perfect centroid accuracy.
    avg_centroid = average(centroid_values) if centroid_values else None
    max_centroid = maximum(centroid_values) if centroid_values else None
    evaluated_frames = max(evaluated_frames, visible_frames)
    # Pointing convergence evidence: final offset and steady-state average
    # (last quarter of the measured series) — real measurements, None if the
    # beacon was never tracked.
    final_center = center_errors[-1] if center_errors else None
    recent = center_errors[-max(1, len(center_errors) // 4):] if center_errors else []
    avg_center_steady = (sum(recent) / len(recent)) if recent else None
    # Retention/loss percentages are undefined without post-lock evaluated
    # frames — report None (an em dash in the UI) instead of a fabricated
    # 0.00% that would imply perfect retention/zero loss with no evidence.
    lock_retention_pct = lock_retention(tracking_frames, retention_frames) if retention_frames > 0 else None
    loss_pct = target_loss(loss_misses, evaluated_frames) if evaluated_frames > 0 else None
    avg_proc_ms = (sum(processing_times) / len(processing_times) * 1000.0) if processing_times else 0.0
    max_proc_ms = (max(processing_times) * 1000.0) if processing_times else 0.0

    summary = PerformanceSummary(
        frames_processed=frame_no,
        duration_s=measured_duration,
        fps=processed_fps,
        acquisition_time_s=first_tracking_time,
        average_tracking_error_px=avg_err,
        maximum_tracking_error_px=max_err,
        rmse_tracking_error_px=rmse_value,
        average_centroid_error_px=avg_centroid,
        maximum_centroid_error_px=max_centroid,
        lock_retention_percentage=lock_retention_pct,
        target_loss_percentage=loss_pct,
        target_loss_events=loss_events,
        successful_reacquisitions=successful_reacq,
        average_reacquisition_time_s=average_reacquisition(reacq_times),
        average_processing_time_ms=avg_proc_ms,
        maximum_reacquisition_time_s=maximum_reacquisition(reacq_times),
        max_processing_time_ms=max_proc_ms,
        ground_truth_available=ground_truth is not None,
        average_center_error_px=avg_center_steady,
        final_center_error_px=final_center,
        center_error_converged=bool(
            final_center is not None and final_center <= config.LOCK_ERROR_THRESHOLD_PX
        ),
        coarse_locked=bool(
            final_center is not None and final_center <= config.LOCK_ERROR_THRESHOLD_PX
        ),
        source=f"benchmark video: {video_path.name}",
        metadata={
            "video_path": str(video_path),
            "source_fps": source_fps,
            "source_width": width,
            "source_height": height,
            "frame_count_hint": frame_count_hint,
            "preferred_signature": preferred_signature or "AUTO",
            "ground_truth_path": str(gt_path) if gt_path.exists() else None,
            "ground_truth_auto_detected": bool(ground_truth_path is None and gt_path.exists()),
            "evaluation_started_frame": metrics_start_frame,
            "proximity_lock_frame": first_tracking_frame if lock_recorded else None,
            "evaluation_started_after_lock": metrics_started,
            "evaluated_frames": evaluated_frames,
            "final_center_error_px": final_center,
            "average_center_error_steady_px": avg_center_steady,
            "forced_loss_injected": forced_injected,
            "forced_reacquisition_time_s": forced_reacq_time_s,
            "link_ready_ai_threshold": config.LINK_READY_AI_THRESHOLD,
            "output_video": str(tracked_video_path) if writer is not None else None,
            "telemetry_csv": str(telemetry_path),
            "cancelled": bool(
                cancel_check is not None and cancel_check()
            ),
            "live_paced": bool(pace_callback is not None),
            "playback_seconds": float(
                time.perf_counter() - _playback_start
            ) if pace_callback is not None else None,
            "processing_seconds": processing_seconds,
        },
    )
    json_path, summary_csv = save_performance_report(summary, output_dir=out_dir, prefix=f"{safe_stem}_benchmark")
    return summary, (json_path, summary_csv, telemetry_path, tracked_video_path)
