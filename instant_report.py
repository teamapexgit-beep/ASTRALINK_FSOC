from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# PDF design constants
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
MARGIN = 13 * 2.83464567  # 13 mm in points

NAVY = colors.HexColor("#173A63")
NAVY_2 = colors.HexColor("#234D7B")
BLUE_LIGHT = colors.HexColor("#EAF2F9")
BLUE_LIGHTER = colors.HexColor("#F6F9FC")
GREEN = colors.HexColor("#0C9A6A")
GREEN_LIGHT = colors.HexColor("#EAF8F2")
AMBER = colors.HexColor("#D28A14")
RED = colors.HexColor("#C74B4B")
TEXT = colors.HexColor("#1E2B36")
MUTED = colors.HexColor("#627384")
BORDER = colors.HexColor("#C9D6E2")
WHITE = colors.white
BLACK = colors.black


def parse_report_time(text: str) -> float:
    """Parse 35s, 2m, 1:30, 90, 500ms -> seconds."""
    raw = str(text).strip().lower().replace(" ", "")
    if not raw:
        raise ValueError("Enter a time such as 35s, 2m, 90, or 1:30.")

    try:
        if raw.endswith("ms"):
            value = float(raw[:-2]) / 1000.0
        elif raw.endswith("min"):
            value = float(raw[:-3]) * 60.0
        elif raw.endswith("m"):
            value = float(raw[:-1]) * 60.0
        elif raw.endswith("s"):
            value = float(raw[:-1])
        elif ":" in raw:
            parts = raw.split(":")
            if len(parts) != 2:
                raise ValueError
            value = float(parts[0]) * 60.0 + float(parts[1])
        else:
            value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Use a time like 35s, 2m, 90, or 1:30.") from exc

    if not math.isfinite(value) or value < 0:
        raise ValueError("Report time must be a non-negative finite value.")
    return value


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return text or "target"


def _num(value: Any, digits: int = 2, fallback: str = "-") -> str:
    if value is None:
        return fallback
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return fallback
    return f"{x:.{digits}f}"


def _text(value: Any, fallback: str = "-") -> str:
    return fallback if value is None or str(value) == "" else str(value)


def _draw_text(c: canvas.Canvas, x: float, y: float, text: str, size: float,
               color=TEXT, bold: bool = False) -> None:
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.drawString(x, y, text)


def _draw_right(c: canvas.Canvas, x: float, y: float, text: str, size: float,
                color=TEXT, bold: bool = False) -> None:
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.drawRightString(x, y, text)


def _draw_section_title(c: canvas.Canvas, x: float, y: float, width: float, number: str, title: str) -> float:
    h = 22
    c.setFillColor(BLUE_LIGHT)
    c.setStrokeColor(BORDER)
    c.roundRect(x, y - h, width, h, 4, fill=1, stroke=1)
    _draw_text(c, x + 8, y - 15, f"{number}.  {title}", 10.5, NAVY, True)
    return y - h - 7


def _draw_metric_card(c: canvas.Canvas, x: float, y: float, w: float, h: float,
                      label: str, value: str, sub: str | None = None,
                      value_color= NAVY) -> None:
    c.setFillColor(WHITE)
    c.setStrokeColor(BORDER)
    c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
    _draw_text(c, x + 8, y + h - 14, label, 7.1, MUTED, True)
    _draw_text(c, x + 8, y + h - 33, value, 15, value_color, True)
    if sub:
        _draw_text(c, x + 8, y + 7, sub, 6.6, MUTED, False)


def _draw_kv_box(c: canvas.Canvas, x: float, y: float, w: float, h: float,
                 rows: list[tuple[str, str]]) -> None:
    c.setFillColor(BLUE_LIGHTER)
    c.setStrokeColor(BORDER)
    c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
    row_h = h / max(len(rows), 1)
    for i, (label, value) in enumerate(rows):
        yy = y + h - (i + 0.68) * row_h
        _draw_text(c, x + 8, yy, label, 6.8, MUTED, True)
        _draw_right(c, x + w - 8, yy, value, 7.1, TEXT, True)
        if i < len(rows) - 1:
            c.setStrokeColor(colors.HexColor("#E1E8EE"))
            c.line(x + 7, y + h - (i + 1) * row_h, x + w - 7, y + h - (i + 1) * row_h)


def _status_for(value: Any, requirement: float, mode: str) -> tuple[str, colors.Color]:
    if value is None:
        return "-", MUTED
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-", MUTED
    if not math.isfinite(v):
        return "-", MUTED
    ok = v >= requirement if mode == ">=" else v <= requirement if mode == "<=" else v < requirement
    return ("PASS", GREEN) if ok else ("CHECK", RED)


def _draw_status_table(c: canvas.Canvas, x: float, y_top: float, width: float, rows: list[tuple[str, str, str, str]]) -> float:
    header_h = 18
    row_h = 18
    c.setFillColor(BLUE_LIGHT)
    c.setStrokeColor(BORDER)
    c.roundRect(x, y_top - header_h, width, header_h, 4, fill=1, stroke=1)
    cols = [0, width * 0.43, width * 0.68, width * 0.84, width]
    heads = ["Parameter", "Measured", "Requirement", "Status"]
    for i, head in enumerate(heads):
        if i == 0:
            _draw_text(c, x + 7, y_top - 13, head, 6.8, NAVY, True)
        else:
            _draw_text(c, x + cols[i] + 5, y_top - 13, head, 6.8, NAVY, True)
    y = y_top - header_h
    for r, (p, measured, req, status) in enumerate(rows):
        y -= row_h
        c.setFillColor(WHITE if r % 2 == 0 else BLUE_LIGHTER)
        c.rect(x, y, width, row_h, fill=1, stroke=0)
        c.setStrokeColor(BORDER)
        c.rect(x, y, width, row_h, fill=0, stroke=1)
        _draw_text(c, x + 7, y + 6, p, 6.8, TEXT)
        _draw_text(c, x + cols[1] + 5, y + 6, measured, 6.8, TEXT, True)
        _draw_text(c, x + cols[2] + 5, y + 6, req, 6.8, TEXT)
        status_color = GREEN if status == "PASS" else RED if status == "CHECK" else MUTED
        _draw_text(c, x + cols[3] + 5, y + 6, status, 6.8, status_color, True)
    return y


def _draw_radar_snapshot(c: canvas.Canvas, x: float, y: float, w: float, h: float, snapshot: Mapping[str, Any]) -> None:
    c.setFillColor(colors.HexColor("#03070A"))
    c.setStrokeColor(colors.HexColor("#2D4658"))
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    cx, cy = x + w * 0.5, y + h * 0.53
    radius = min(w, h) * 0.28
    c.setStrokeColor(colors.HexColor("#476070"))
    for f in (0.4, 0.7, 1.0):
        c.circle(cx, cy, radius * f, fill=0, stroke=1)
    c.setStrokeColor(colors.HexColor("#355366"))
    c.line(cx - radius, cy, cx + radius, cy)
    c.line(cx, cy - radius, cx, cy + radius)

    # Approximate current angular position from pixel target position.
    tx = snapshot.get("target_x_px")
    ty = snapshot.get("target_y_px")
    pan = snapshot.get("pan_deg", 0.0)
    tilt = snapshot.get("tilt_deg", 0.0)
    fx = float(tx - 320.0) / 320.0 if tx is not None else float(pan) / 5.0
    fy = float(240.0 - ty) / 240.0 if ty is not None else float(tilt) / 4.0
    scale = radius * 0.85
    bx = cx + max(-1.0, min(1.0, fx)) * scale
    by = cy + max(-1.0, min(1.0, fy)) * scale

    c.setStrokeColor(colors.HexColor("#4EA8FF"))
    c.setLineWidth(1.3)
    c.line(cx, cy, bx, by)
    c.setFillColor(colors.HexColor("#45F57A"))
    c.circle(bx, by, 4.0, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#FFD447"))
    c.circle(min(w + x - 9, bx + 9), min(y + h - 11, by + 8), 2.7, fill=1, stroke=0)
    _draw_text(c, x + 7, y + h - 12, "RADAR", 7.0, WHITE, True)
    _draw_text(c, x + 7, y + 7, "Target / Boresight / Prediction", 6.0, colors.HexColor("#B9CBD8"))


def _draw_camera_snapshot(c: canvas.Canvas, x: float, y: float, w: float, h: float, snapshot: Mapping[str, Any]) -> None:
    c.setFillColor(colors.HexColor("#020507"))
    c.setStrokeColor(colors.HexColor("#2D4658"))
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    _draw_text(c, x + 7, y + h - 12, "VIRTUAL CAMERA", 7.0, WHITE, True)
    cx, cy = x + w * 0.5, y + h * 0.52
    c.setStrokeColor(colors.HexColor("#294251"))
    c.line(cx - w * .38, cy, cx + w * .38, cy)
    c.line(cx, cy - h * .30, cx, cy + h * .30)
    c.setStrokeColor(colors.HexColor("#43F26F"))
    c.setLineWidth(1.0)
    c.rect(cx - 8, cy - 8, 16, 16, fill=0, stroke=1)
    c.circle(cx, cy, 3.5, fill=1, stroke=0)
    _draw_text(c, x + 7, y + 7, f"Target {_text(snapshot.get('target_id'))} • Error {_num(snapshot.get('center_error_px'))} px", 6.0, colors.HexColor("#B9CBD8"))


def _draw_3d_snapshot(c: canvas.Canvas, x: float, y: float, w: float, h: float, snapshot: Mapping[str, Any]) -> None:
    c.setFillColor(colors.HexColor("#03070E"))
    c.setStrokeColor(colors.HexColor("#2D4658"))
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    # Stars
    c.setFillColor(colors.HexColor("#DCE8F1"))
    for sx, sy in ((.10,.78),(.20,.63),(.35,.82),(.53,.70),(.72,.88),(.88,.57),(.62,.53),(.30,.50)):
        c.circle(x + w*sx, y + h*sy, 0.7, fill=1, stroke=0)
    _draw_text(c, x + 7, y + h - 12, "3D DIGITAL TWIN", 7.0, WHITE, True)
    # Earth-ish lower body
    c.setFillColor(colors.HexColor("#1E4C68"))
    c.circle(x + w*.23, y + h*.20, h*.17, fill=1, stroke=0)
    # Camera satellite
    sx, sy = x + w*.70, y + h*.63
    c.setFillColor(colors.HexColor("#6A7882"))
    c.rect(sx-8, sy-5, 16, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#9AB2C2"))
    c.rect(sx-22, sy-3, 12, 6, fill=1, stroke=0)
    c.rect(sx+10, sy-3, 12, 6, fill=1, stroke=0)
    # Beacon satellite
    bx, by = x + w*.43, y + h*.43
    c.setFillColor(colors.HexColor("#6A7882"))
    c.rect(bx-7, by-5, 14, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#39F76E"))
    c.circle(bx, by, 3, fill=1, stroke=0)
    # Beam
    c.setStrokeColor(colors.HexColor("#39F76E"))
    c.setLineWidth(1.8)
    c.line(sx-8, sy, bx+2, by)
    _draw_text(c, x + 7, y + 7, "Camera terminal -> designated beacon", 6.0, colors.HexColor("#B9CBD8"))


def _draw_footer(c: canvas.Canvas) -> None:
    c.setStrokeColor(BORDER)
    c.line(MARGIN, 25, PAGE_W - MARGIN, 25)
    _draw_text(c, MARGIN, 13, "PS 26169 | Department of Space / ISRO | Smart Automation / Space Technology", 6.4, MUTED)
    _draw_right(c, PAGE_W - MARGIN, 13, "FSOC Point-in-Time Performance Report", 6.4, MUTED)


def _snapshot_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(snapshot)
    payload["report_type"] = "point_in_time_fsoc_performance"
    payload["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return payload


def save_instant_report(snapshot: Mapping[str, Any], output_dir: str = "reports") -> tuple[Path, Path, Path]:
    """Create one-page PDF + machine-readable JSON + CSV.

    No HTML and no ReportLab Platypus layout flow are used; the PDF is drawn
    directly onto one A4 canvas so the page count and placement are deterministic.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping/dict of telemetry values")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = _snapshot_payload(snapshot)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = _safe_name(payload.get("target_id", "target"))

    pdf_path = out / f"instant_report_{target}_{stamp}.pdf"
    json_path = out / f"instant_report_{target}_{stamp}.json"
    csv_path = out / f"instant_report_{target}_{stamp}.csv"

    # Machine-readable files first.
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key, value in payload.items():
            writer.writerow([key, value])

    c = canvas.Canvas(str(pdf_path), pagesize=A4, pageCompression=1)
    c.setTitle("ASTRALINK Point-in-Time Performance Report")
    c.setAuthor("ASTRALINK — AI-Based Virtual Camera Tracking System for Mobile FSOC")

    # Header
    header_h = 66
    top = PAGE_H - MARGIN
    c.setFillColor(NAVY)
    c.roundRect(MARGIN, top - header_h, PAGE_W - 2 * MARGIN, header_h, 6, fill=1, stroke=0)
    _draw_text(c, MARGIN + 12, top - 20, "ASTRALINK — AI-BASED VIRTUAL CAMERA TRACKING SYSTEM FOR MOBILE FSOC", 13, WHITE, True)
    _draw_text(c, MARGIN + 12, top - 38, "Coarse Alignment - Point-in-Time Performance Report", 8.5, colors.HexColor("#D7E6F2"))
    right_x = PAGE_W - MARGIN - 12
    _draw_right(c, right_x, top - 20, f"TARGET: {_text(payload.get('target_id'), 'TGT-01')}", 8.2, WHITE, True)
    _draw_right(c, right_x, top - 38, f"T = {_num(payload.get('simulation_time_s'))} s", 8.2, colors.HexColor("#D7E6F2"), True)

    y = top - header_h - 8
    inner_w = PAGE_W - 2 * MARGIN

    # Mission snapshot
    y = _draw_section_title(c, MARGIN, y, inner_w, "1", "TEST CONFIGURATION")
    cfg_h = 46
    col_w = inner_w / 5
    config_items = [
        ("Scenario", _text(payload.get("motion_pattern", payload.get("scenario", "-")))),
        ("Target", _text(payload.get("target_id"))),
        ("Camera", _text(payload.get("camera_resolution", "640 x 480"))),
        ("FOV", _text(payload.get("fov", "4 deg x 3 deg"))),
        ("Update Rate", f"{_num(payload.get('update_rate_hz', 30), 1)} Hz"),
    ]
    for i, (label, value) in enumerate(config_items):
        _draw_metric_card(c, MARGIN + i * col_w, y - cfg_h, col_w - 3, cfg_h, label, value)
    y -= cfg_h + 6

    disturbance_text = (
        f"Noise: {_text(payload.get('noise', 'None'))} {_num(payload.get('noise_level'))}"
        f"    |    Camera jitter: {'ON' if payload.get('camera_jitter_enabled') else 'OFF'} { _num(payload.get('camera_jitter_px')) } px/frame"
        f"    |    Platform motion: {'ON' if payload.get('platform_motion_enabled') else 'OFF'} { _num(payload.get('platform_motion_px')) } px"
        f"    |    Turbulence: {'ON' if payload.get('turbulence_enabled') else 'OFF'} { _num(payload.get('turbulence_strength_px')) } px"
    )
    c.setFillColor(BLUE_LIGHTER)
    c.setStrokeColor(BORDER)
    c.roundRect(MARGIN, y - 21, inner_w, 21, 4, fill=1, stroke=1)
    _draw_text(c, MARGIN + 8, y - 14, disturbance_text, 6.5, MUTED)
    y -= 27

    # Performance metrics
    y = _draw_section_title(c, MARGIN, y, inner_w, "2", "PERFORMANCE")
    gap = 5
    card_w = (inner_w - 2 * gap) / 3
    card_h = 50
    metrics = [
        ("Processing FPS", _num(payload.get("fps"), 1), ">= 20 FPS"),
        ("Average Tracking Error", f"{_num(payload.get('average_tracking_error_px', payload.get('center_error_px')))} px", None),
        ("Maximum Tracking Error", f"{_num(payload.get('maximum_tracking_error_px', payload.get('center_error_px')))} px", "<= 10 px"),
        ("Acquisition Time", f"{_num(payload.get('acquisition_time_s'))} s", "<= 2 s"),
        ("Re-acquisition", f"{_num(payload.get('average_reacquisition_s'))} s", "<= 1 s"),
        ("Lock Retention", f"{_num(payload.get('lock_retention_percentage'))} %", None),
    ]
    for i, (label, value, sub) in enumerate(metrics):
        row, col = divmod(i, 3)
        x = MARGIN + col * (card_w + gap)
        yy = y - row * (card_h + gap) - card_h
        color = NAVY
        if label == "Processing FPS" and payload.get("fps") is not None and float(payload.get("fps")) >= 20:
            color = GREEN
        _draw_metric_card(c, x, yy, card_w, card_h, label, value, sub, color)
    y -= 2 * (card_h + gap) + 5

    # Secondary performance strip
    rmse = _num(payload.get("rmse_tracking_error_px"))
    avg_proc = _num(payload.get("average_processing_time_ms"), 2)
    loss = _num(payload.get("target_loss_percentage"), 2)
    reacq_count = _text(payload.get("reacquisition_count"), "0")
    strip = [
        ("RMSE", f"{rmse} px"),
        ("Target Loss", f"{loss} %"),
        ("Avg Processing", f"{avg_proc} ms"),
        ("Re-acquisitions", reacq_count),
    ]
    sx = MARGIN
    sw = (inner_w - 3 * 4) / 4
    for label, value in strip:
        c.setFillColor(BLUE_LIGHTER)
        c.setStrokeColor(BORDER)
        c.roundRect(sx, y - 24, sw, 24, 4, fill=1, stroke=1)
        _draw_text(c, sx + 7, y - 10, label, 6.4, MUTED, True)
        _draw_right(c, sx + sw - 7, y - 10, value, 7.0, NAVY, True)
        sx += sw + 4
    y -= 30

    # Requirement check + snapshots side by side
    half = (inner_w - 7) / 2
    left_x = MARGIN
    right_x = MARGIN + half + 7
    y_left = _draw_section_title(c, left_x, y, half, "3", "PS REQUIREMENT CHECK")
    rows = []
    u_status, u_color = _status_for(payload.get("update_rate_hz", 30), 20, ">=")
    p_status, p_color = _status_for(payload.get("fps"), 20, ">=")
    a_status, a_color = _status_for(payload.get("acquisition_time_s"), 2, "<=")
    e_val = payload.get("maximum_tracking_error_px", payload.get("center_error_px"))
    e_status, e_color = _status_for(e_val, 10, "<=")
    l_status, l_color = _status_for(payload.get("target_loss_percentage"), 5, "<")
    r_status, r_color = _status_for(payload.get("average_reacquisition_s"), 1, "<=")
    rows.extend([
        ("Update Rate", f"{_num(payload.get('update_rate_hz', 30),1)} Hz", ">= 20 Hz", u_status),
        ("Processing Speed", f"{_num(payload.get('fps'),1)} FPS", ">= 20 FPS", p_status),
        ("Acquisition Time", f"{_num(payload.get('acquisition_time_s'))} s", "<= 2 s", a_status),
        ("Tracking Error (Max)", f"{_num(e_val)} px", "<= 10 px", e_status),
        ("Target Loss", f"{_num(payload.get('target_loss_percentage'),2)} %", "< 5 %", l_status),
        ("Re-acquisition", f"{_num(payload.get('average_reacquisition_s'))} s", "<= 1 s", r_status),
    ])
    y_table_bottom = _draw_status_table(c, left_x, y_left, half, rows)

    # Snapshot panel on right
    y_right = _draw_section_title(c, right_x, y, half, "4", f"VISUAL SNAPSHOT  (T = {_num(payload.get('simulation_time_s'))} s)")
    snap_gap = 4
    snap_w = (half - snap_gap) / 2
    snap_h = 84
    _draw_radar_snapshot(c, right_x, y_right - snap_h, snap_w, snap_h, payload)
    _draw_camera_snapshot(c, right_x + snap_w + snap_gap, y_right - snap_h, snap_w, snap_h, payload)
    _draw_3d_snapshot(c, right_x, y_right - 2 * snap_h - snap_gap, half, snap_h, payload)
    _draw_text(c, right_x + 7, y_right - 2 * snap_h - snap_gap - 12,
               f"State: {_text(payload.get('state'))}   |   Pan: {_num(payload.get('pan_deg'))} deg   |   Tilt: {_num(payload.get('tilt_deg'))} deg   |   AI: {_num(payload.get('ai_confidence'))}",
               6.3, MUTED)

    # System summary / handoff
    summary_y = min(y_table_bottom, y_right - 2 * snap_h - snap_gap - 20) - 8
    summary_y = _draw_section_title(c, MARGIN, summary_y, inner_w, "5", "SYSTEM SUMMARY")
    summary_h = 42
    ready = bool(payload.get("link_ready"))
    c.setFillColor(GREEN_LIGHT if ready else BLUE_LIGHTER)
    c.setStrokeColor(GREEN if ready else BORDER)
    c.roundRect(MARGIN, summary_y - summary_h, inner_w, summary_h, 5, fill=1, stroke=1)
    headline = "HANDOFF READY - COARSE ALIGNMENT GATE SATISFIED" if ready else "COARSE ALIGNMENT CONTINUES - HANDOFF NOT READY"
    _draw_text(c, MARGIN + 10, summary_y - 16, headline, 8.8, GREEN if ready else RED, True)
    summary = (
        f"Designated target {_text(payload.get('target_id'))} at {_num(payload.get('simulation_time_s'))} s. "
        f"Detection, AI verification, Kalman prediction and pan/tilt control are reported from the recorded telemetry snapshot."
    )
    _draw_text(c, MARGIN + 10, summary_y - 31, summary[:170], 6.5, TEXT)

    _draw_footer(c)
    c.showPage()
    c.save()

    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        raise OSError("PDF generation completed but the PDF file is missing or invalid.")
    return pdf_path, json_path, csv_path


if __name__ == "__main__":
    # Standalone smoke test with realistic placeholder telemetry.
    demo = {
        "simulation_time_s": 35.0,
        "frame": 1050,
        "target_id": "TGT-01",
        "target_signature": "GREEN-SQUARE",
        "target_color": "GREEN",
        "target_shape": "SQUARE",
        "target_count": 3,
        "state": "TRACKING",
        "target_x_px": 318.0,
        "target_y_px": 241.0,
        "center_error_px": 2.1,
        "centroid_error_px": 0.42,
        "fps": 103.6,
        "update_rate_hz": 30.0,
        "acquisition_time_s": 1.24,
        "average_tracking_error_px": 3.42,
        "maximum_tracking_error_px": 8.71,
        "rmse_tracking_error_px": 3.86,
        "lock_retention_percentage": 98.7,
        "target_loss_percentage": 1.3,
        "reacquisition_count": 1,
        "average_reacquisition_s": 0.61,
        "average_processing_time_ms": 9.65,
        "max_processing_time_ms": 12.10,
        "pan_deg": 1.13,
        "tilt_deg": -0.31,
        "detector_confidence": 0.97,
        "ai_confidence": 0.94,
        "signature_score": 1.0,
        "noise": "Gaussian",
        "noise_level": 20,
        "turbulence_enabled": True,
        "turbulence_strength_px": 8,
        "camera_jitter_enabled": True,
        "camera_jitter_px": 10,
        "platform_motion_enabled": True,
        "platform_motion_px": 8,
        "atmosphere": "Clear",
        "atmosphere_level": 0,
        "motion_pattern": "Figure 8",
        "camera_resolution": "640 x 480",
        "fov": "4 deg x 3 deg",
        "link_ready": True,
        "scenario": "Figure 8",
    }
    paths = save_instant_report(demo, output_dir="reports_demo")
    print("PDF:", paths[0])
    print("JSON:", paths[1])
    print("CSV:", paths[2])
