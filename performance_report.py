"""Performance summary + automatic PS requirement checks."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

# The normal performance report's PDF deliberately reuses the instant report's
# drawing helpers and palette so every report from the application looks
# visually consistent. instant_report.py itself is NOT modified by this import.
from instant_report import (
    A4,
    BORDER,
    BLUE_LIGHT,
    BLUE_LIGHTER,
    GREEN,
    MARGIN,
    MUTED,
    NAVY,
    PAGE_H,
    PAGE_W,
    RED,
    TEXT,
    WHITE,
    _draw_metric_card,
    _draw_right,
    _draw_section_title,
    _draw_text,
)
from reportlab.lib import colors
from reportlab.pdfgen import canvas


@dataclass
class PerformanceSummary:
    frames_processed: int
    duration_s: float
    fps: float
    acquisition_time_s: float | None
    average_tracking_error_px: float | None
    maximum_tracking_error_px: float | None
    rmse_tracking_error_px: float | None
    average_centroid_error_px: float
    maximum_centroid_error_px: float
    lock_retention_percentage: float | None
    target_loss_percentage: float | None
    target_loss_events: int
    successful_reacquisitions: int
    average_reacquisition_time_s: float | None
    average_processing_time_ms: float
    max_processing_time_ms: float
    ground_truth_available: bool
    source: str
    # Coarse-pointing convergence evidence (uploaded-video analysis):
    average_center_error_px: float | None = None
    final_center_error_px: float | None = None
    center_error_converged: bool = False
    coarse_locked: bool = False
    handoff_ready: bool = False
    maximum_reacquisition_time_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_ps_requirements(summary: PerformanceSummary) -> dict[str, Any]:
    """Evaluate the five measurable PS performance limits.

    A requirement with unavailable evidence is marked ``PENDING`` rather than
    being incorrectly treated as a pass. Target-loss uses the strict PS
    requirement ``< 5%``; processing speed uses sustained measured FPS.
    ``None`` tracking-error values (no valid ground-truth samples) also
    evaluate as PENDING rather than PASS.
    """
    checks: dict[str, dict[str, Any]] = {}

    acq = summary.acquisition_time_s
    checks["acquisition_time"] = {
        "requirement": "<= 2.0 s",
        "value": acq,
        "status": "PENDING" if acq is None else ("PASS" if acq <= config.PS_ACQUISITION_MAX_S else "FAIL"),
    }

    max_err = summary.maximum_tracking_error_px
    checks["tracking_error"] = {
        "requirement": "<= 10 px",
        "value": max_err,
        "status": "PENDING" if max_err is None else ("PASS" if max_err <= config.PS_TRACKING_ERROR_MAX_PX else "FAIL"),
    }

    loss = summary.target_loss_percentage
    checks["target_loss"] = {
        "requirement": "< 5%",
        "value": loss,
        "status": "PENDING" if loss is None else ("PASS" if loss < config.PS_TARGET_LOSS_MAX_PERCENT else "FAIL"),
    }

    # Coarse-pointing convergence: the controller must drive the beacon to
    # the boresight. Without pointing evidence the value defers to the
    # tracking-error check (real simulations always have boresight error).
    center = summary.final_center_error_px
    if center is None:
        center = summary.maximum_tracking_error_px
    checks["center_error"] = {
        "requirement": "<= 10 px",
        "value": center,
        "status": "PENDING" if center is None else ("PASS" if center <= config.PS_TRACKING_ERROR_MAX_PX else "FAIL"),
        "basis": "measured final pointing offset" if summary.final_center_error_px is not None else "tracking-error proxy",
    }

    reacq = summary.maximum_reacquisition_time_s
    reacq_basis = "maximum observed"
    if reacq is None:
        reacq = summary.average_reacquisition_time_s
    if reacq is None:
        # Same-run measured forced-loss rehearsal (the intentional
        # TARGET LOSS -> RE-ACQUIRING -> re-detection test).
        reacq = summary.metadata.get("forced_reacquisition_time_s")
        reacq_basis = "forced-loss rehearsal (measured)"
    checks["reacquisition_time"] = {
        "requirement": "<= 1.0 s",
        "value": reacq,
        "status": "PENDING" if reacq is None else ("PASS" if reacq <= config.PS_REACQUISITION_MAX_S else "FAIL"),
        "basis": (
            reacq_basis
            if reacq is not None
            else ("maximum observed" if summary.maximum_reacquisition_time_s is not None else "average observed")
        ),
    }

    fps = float(summary.fps)
    checks["processing_speed"] = {
        "requirement": ">= 20 FPS",
        "value": fps,
        "status": "PASS" if fps >= config.PS_PROCESSING_MIN_FPS else "FAIL",
    }

    statuses = [item["status"] for item in checks.values()]
    overall = "FAIL" if "FAIL" in statuses else ("PENDING" if "PENDING" in statuses else "PASS")
    return {"overall": overall, "checks": checks}


def _summary_payload(summary: PerformanceSummary) -> dict[str, Any]:
    data = asdict(summary)
    data["ps_requirement_check"] = evaluate_ps_requirements(summary)
    data["generated_utc"] = datetime.now(timezone.utc).isoformat()
    return data


def _write_json_csv(data: dict[str, Any], out: Path, prefix: str, stamp: str):
    """Write the machine-readable half of a report bundle (unchanged format)."""
    json_path = out / f"{prefix}_{stamp}.json"
    csv_path = out / f"{prefix}_{stamp}.csv"

    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value"])
        for key, value in data.items():
            if isinstance(value, dict):
                value = json.dumps(value, sort_keys=True)
            writer.writerow([key, value])
    return json_path, csv_path


def save_performance_report(summary: PerformanceSummary, output_dir="reports", prefix="performance"):
    """Write JSON + CSV only. Public contract (2-tuple return) is unchanged;
    benchmark_video.run_video_benchmark() unpacks this return order."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    data = _summary_payload(summary)
    return _write_json_csv(data, out, prefix, stamp)


def _fmt(value: Any, digits: int = 2, na: str = "\u2014") -> str:
    """Format a metric for the PDF. Missing values are shown as an em dash,
    never as a fabricated number."""
    if value is None:
        return na
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return na
    return f"{f:.{digits}f}"


def _draw_performance_footer(c) -> None:
    c.setStrokeColor(BORDER)
    c.line(MARGIN, 25, PAGE_W - MARGIN, 25)
    _draw_text(c, MARGIN, 13, "PS 26169 | Department of Space / ISRO | Smart Automation / Space Technology", 6.4, MUTED)
    _draw_right(c, PAGE_W - MARGIN, 13, "FSOC Performance Report", 6.4, MUTED)


def _draw_requirement_table(c, x: float, y_top: float, width: float, rows) -> float:
    """Requirement table with explicit PASS / FAIL / PENDING colors.

    Kept local because instant_report._draw_status_table colors only
    PASS/CHECK, which would render a genuine FAIL as neutral gray.
    """
    header_h = 18
    row_h = 18
    c.setFillColor(BLUE_LIGHT)
    c.setStrokeColor(BORDER)
    c.roundRect(x, y_top - header_h, width, header_h, 4, fill=1, stroke=1)
    cols = [0, width * 0.43, width * 0.68, width * 0.84, width]
    for i, head in enumerate(["Parameter", "Measured", "Requirement", "Status"]):
        if i == 0:
            _draw_text(c, x + 7, y_top - 13, head, 6.8, NAVY, True)
        else:
            _draw_text(c, x + cols[i] + 5, y_top - 13, head, 6.8, NAVY, True)
    y = y_top - header_h
    status_colors = {"PASS": GREEN, "FAIL": RED, "PENDING": MUTED}
    for r, (param, measured, req, status) in enumerate(rows):
        y -= row_h
        c.setFillColor(WHITE if r % 2 == 0 else BLUE_LIGHTER)
        c.rect(x, y, width, row_h, fill=1, stroke=0)
        c.setStrokeColor(BORDER)
        c.rect(x, y, width, row_h, fill=0, stroke=1)
        _draw_text(c, x + 7, y + 6, param, 6.8, TEXT, False)
        _draw_text(c, x + cols[1] + 5, y + 6, measured, 6.8, TEXT, True)
        _draw_text(c, x + cols[2] + 5, y + 6, req, 6.8, TEXT, False)
        sc = status_colors.get(status, MUTED)
        _draw_text(c, x + cols[3] + 5, y + 6, status, 6.8, sc, True)
    return y


def _render_performance_pdf(summary: PerformanceSummary, data: dict[str, Any], pdf_path: Path) -> None:
    """Render the one-page A4 performance PDF (ReportLab canvas, same visual
    language as the instant report)."""
    c = canvas.Canvas(str(pdf_path), pagesize=A4, pageCompression=1)
    c.setTitle("ASTRALINK Performance Report")
    c.setAuthor("ASTRALINK — AI-Based Virtual Camera Tracking System for Mobile FSOC")

    soft = colors.HexColor("#D7E6F2")
    top = PAGE_H - MARGIN
    inner_w = PAGE_W - 2 * MARGIN

    # Header
    header_h = 58
    c.setFillColor(NAVY)
    c.roundRect(MARGIN, top - header_h, inner_w, header_h, 6, fill=1, stroke=0)
    _draw_text(c, MARGIN + 12, top - 19, "ASTRALINK — AI-BASED VIRTUAL CAMERA TRACKING SYSTEM FOR MOBILE FSOC", 12, WHITE, True)
    _draw_text(c, MARGIN + 12, top - 35, "Coarse Alignment - Performance Report", 8.5, soft)
    _draw_right(c, PAGE_W - MARGIN - 12, top - 19, str(summary.source)[:46], 7.6, WHITE, True)
    _draw_right(c, PAGE_W - MARGIN - 12, top - 35, f"UTC {str(data.get('generated_utc', ''))[:19]}", 7.0, soft)

    y = top - header_h - 8

    # 1. Run configuration
    y = _draw_section_title(c, MARGIN, y, inner_w, "1", "RUN CONFIGURATION")
    cfg_h = 46
    col_w = inner_w / 5
    config_items = [
        ("Frames Processed", f"{int(summary.frames_processed)}"),
        ("Duration", f"{_fmt(summary.duration_s)} s"),
        ("Processing FPS", _fmt(summary.fps, 1)),
        ("Avg Processing", f"{_fmt(summary.average_processing_time_ms)} ms"),
        ("Max Processing", f"{_fmt(summary.max_processing_time_ms)} ms"),
    ]
    for i, (label, value) in enumerate(config_items):
        _draw_metric_card(c, MARGIN + i * col_w, y - cfg_h, col_w - 3, cfg_h, label, value)
    y -= cfg_h + 6

    # 2. Tracking performance
    y = _draw_section_title(c, MARGIN, y, inner_w, "2", "TRACKING PERFORMANCE")
    gap = 5
    card_w = (inner_w - 2 * gap) / 3
    card_h = 50
    gt = bool(summary.ground_truth_available)
    avg_reacq = summary.average_reacquisition_time_s
    max_reacq = summary.maximum_reacquisition_time_s
    reacq_text = (
        "\u2014"
        if avg_reacq is None and max_reacq is None
        else f"{_fmt(avg_reacq)} / {_fmt(max_reacq)} s"
    )
    # Centroid accuracy is reported ONLY when ground truth was available.
    centroid_text = (
        "\u2014" if not gt
        else f"{_fmt(summary.average_centroid_error_px)} / {_fmt(summary.maximum_centroid_error_px)} px"
    )
    metrics = [
        ("Acquisition Time", f"{_fmt(summary.acquisition_time_s)} s", "<= 2 s"),
        ("Average Tracking Error", f"{_fmt(summary.average_tracking_error_px)} px", None),
        ("Maximum Tracking Error", f"{_fmt(summary.maximum_tracking_error_px)} px", "<= 10 px"),
        ("RMSE Tracking Error", f"{_fmt(summary.rmse_tracking_error_px)} px", None),
        ("Centroid Error Avg/Max", centroid_text, "ground truth only"),
        ("Lock Retention", f"{_fmt(summary.lock_retention_percentage)} %", None),
        ("Target Loss", f"{_fmt(summary.target_loss_percentage)} %", "< 5 %"),
        ("Re-acquisition Avg/Max", reacq_text, "<= 1 s"),
        ("Successful Re-acq.", str(int(summary.successful_reacquisitions)), f"{int(summary.target_loss_events)} loss events"),
    ]
    for i, (label, value, sub) in enumerate(metrics):
        row, col = divmod(i, 3)
        x = MARGIN + col * (card_w + gap)
        yy = y - row * (card_h + gap) - card_h
        color = NAVY
        if (
            label == "Target Loss"
            and summary.target_loss_percentage is not None
            and summary.target_loss_percentage < config.PS_TARGET_LOSS_MAX_PERCENT
        ):
            color = GREEN
        _draw_metric_card(c, x, yy, card_w, card_h, label, value, sub, color)
    y -= 3 * (card_h + gap) + 5

    # 3. PS requirement check (statuses may be PASS / FAIL / PENDING)
    y = _draw_section_title(c, MARGIN, y, inner_w, "3", "PS REQUIREMENT CHECK")
    checks = data.get("ps_requirement_check", {}).get("checks", {})
    display_names = {
        "acquisition_time": "Acquisition Time",
        "tracking_error": "Tracking Error (Max)",
        "center_error": "Center Error (Final)",
        "target_loss": "Target Loss",
        "reacquisition_time": "Re-acquisition Time",
        "processing_speed": "Processing Speed",
    }
    rows = []
    for key in ("acquisition_time", "tracking_error", "center_error", "target_loss", "reacquisition_time", "processing_speed"):
        item = checks.get(key, {})
        value = item.get("value")
        if value is None:
            measured = "\u2014"
        elif key == "target_loss":
            measured = f"{float(value):.2f} %"
        elif key == "processing_speed":
            measured = f"{float(value):.1f} FPS"
        else:
            measured = f"{float(value):.2f}"
        status = str(item.get("status", "PENDING"))
        rows.append((display_names.get(key, key), measured, str(item.get("requirement", "-")), status))
    y = _draw_requirement_table(c, MARGIN, y, inner_w, rows)

    note = (
        "Ground truth was available: centroid accuracy is measured."
        if gt
        else "No ground truth: centroid accuracy is intentionally not reported (never fabricated)."
    )
    _draw_text(c, MARGIN + 4, y - 12, note, 6.8, MUTED)
    y -= 24

    # Overall band
    summary_h = 34
    c.setFillColor(BLUE_LIGHTER)
    c.setStrokeColor(BORDER)
    c.roundRect(MARGIN, y - summary_h, inner_w, summary_h, 5, fill=1, stroke=1)
    overall = str(data.get("ps_requirement_check", {}).get("overall", "PENDING")).upper()
    _draw_text(c, MARGIN + 10, y - 14, f"OVERALL: {overall}", 9, NAVY, True)
    _draw_text(
        c, MARGIN + 10, y - 27,
        f"Source: {summary.source}  |  Frames: {int(summary.frames_processed)}  |  FPS: {_fmt(summary.fps, 1)}",
        6.6, MUTED,
    )

    _draw_performance_footer(c)
    c.showPage()
    c.save()


def save_full_performance_report(
    summary: PerformanceSummary,
    output_dir="reports",
    prefix="performance",
    write_json_csv=True,
):
    """Write the normal performance report bundle: PDF (+ JSON + CSV).

    With ``write_json_csv=True`` (the GUI EXPORT REPORT default) returns
    ``(pdf_path, json_path, csv_path)`` — paths[0] is the PDF, matching
    instant_report.save_instant_report()'s documented order. With
    ``write_json_csv=False`` only the PDF is written and the JSON/CSV slots
    are ``None`` — used when a bundle already exists for the same run
    (uploaded-video analysis) and OPEN REPORT PDF must open exactly that
    bundle. JSON/CSV content is byte-identical to save_performance_report()
    for the same summary, and all artifacts share one timestamp stem.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    data = _summary_payload(summary)
    json_path = csv_path = None
    if write_json_csv:
        json_path, csv_path = _write_json_csv(data, out, prefix, stamp)

    pdf_path = out / f"{prefix}_{stamp}.pdf"
    _render_performance_pdf(summary, data, pdf_path)
    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        raise OSError("Performance PDF generation completed but the PDF file is missing or invalid.")
    return pdf_path, json_path, csv_path
