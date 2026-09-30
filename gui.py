from __future__ import annotations

import csv
import glob
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QPointF,
    QSize,
    QThread,
    QTimer,
    Qt,
    QUrl,
    Signal,
)
from PySide6.QtGui import QBrush, QDesktopServices, QFont, QFontMetrics, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsOpacityEffect
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QFileDialog,
    QProgressBar,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import config
from flow_layout import FlowLayout
from benchmark_video import run_video_benchmark
from video_analysis import VideoAnalysisWindow, validate_ground_truth_file
from digital_twin import DigitalTwinWidget, DigitalTwinWindow, TwinPane


# Five-level disturbance intensity scale shared by the THREE disturbance
# families (Atmospheric Turbulence / Camera Jitter / Platform Motion).
# The level names are pure labels; the px values are the exact inputs the
# existing simulator disturbance mathematics already consumed.
DISTURBANCE_LEVEL_NAMES = (
    "Very Low",
    "Low",
    "Medium",
    "High",
    "Very High",
)
DISTURBANCE_LEVEL_PX = (4.0, 8.0, 12.0, 16.0, 20.0)

# Window width at which KALMAN + LINK READINESS can share one row inside the
# Virtual Camera column without squeezing either panel's own content. Below
# it they stack exactly as before (see FSOCWindow._relayout_support_row).
SUPPORT_ROW_MIN_WIDTH = 1400


def disturbance_level_text(level):
    """Return a ``Level 3 — Medium`` style label for a 1..5 selector."""
    level = max(1, min(5, int(level)))
    return f"Level {level} — {DISTURBANCE_LEVEL_NAMES[level - 1]}"


def disturbance_level_to_px(level):
    """Map an intensity level (1..5) onto the simulator's px strength."""
    return DISTURBANCE_LEVEL_PX[max(1, min(5, int(level))) - 1]


def disturbance_px_to_level(strength_px):
    """Nearest intensity level (1..5) for an existing px strength."""
    try:
        strength = max(0.0, float(strength_px))
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, int(round(strength / 4.0)))) or 1


class VideoBenchmarkWorker(QThread):
    """Runs run_video_benchmark off the GUI thread; results via Qt signals.

    In ``live`` mode it also drives real-time playback: the engine hands back
    every analysed frame (``frame_ready``) and the worker paces the video
    timeline, honouring PAUSE (parks, zero CPU, clock frozen) and RESUME
    (continues at exactly the same frame).  There is ONE processing loop for
    the whole run — the same loop that measures the benchmark — so the picture
    on screen can never drift from the analysis.
    """

    finished = Signal(object, list, str, str)
    failed = Signal(str)
    progress = Signal(int, str)
    frame_ready = Signal(object, object)

    def __init__(self, video_path, ground_truth_path=None, preferred_signature=None,
                 supplied_status="NOT_PROVIDED", live=False, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.ground_truth_path = ground_truth_path
        self.preferred_signature = preferred_signature
        # What the USER did: picked a valid CSV (NOT_PROVIDED impossible here) or
        # cancelled / supplied an invalid one. The engine may still auto-detect a
        # sibling <stem>_groundtruth.csv; the finish handler reconciles honestly.
        self.supplied_status = supplied_status
        self.live = bool(live)
        self._cancelled = False
        self._paused = False
        self._play_clock = 0.0
        self._lock = threading.Condition()

    # ---- transport control (called from the GUI thread) ----------------
    def cancel(self):
        """Cooperative cancellation requested from the GUI thread."""
        with self._lock:
            self._cancelled = True
            self._paused = False
            self._lock.notify_all()

    def request_pause(self):
        """Freeze processing: the run parks on the worker thread."""
        with self._lock:
            self._paused = True
            self._lock.notify_all()

    def request_resume(self):
        """Continue from the exact frame that was on screen when paused."""
        with self._lock:
            self._paused = False
            self._lock.notify_all()

    def is_paused(self):
        with self._lock:
            return self._paused

    def is_cancelled(self):
        with self._lock:
            return self._cancelled

    # ---- real-time pacing / pause gate --------------------------------
    def _pace(self, playback_target_s):
        """Block until this frame is due on the video timeline.

        Returns ``True`` to continue and ``False`` when the run was cancelled.
        While paused this parks on the condition variable — no spinning, no
        clock advance — so RESUME continues from the same position instead of
        jumping to catch up.
        """
        try:
            target = float(playback_target_s)
        except (TypeError, ValueError):
            return True
        while True:
            with self._lock:
                if self._cancelled:
                    return False
                if self._paused:
                    self._lock.wait(0.2)
                    continue
                remaining = target - self._play_clock
                if remaining <= 0.0:
                    return True
                started = time.perf_counter()
                self._lock.wait(min(remaining, 0.1))
                if not self._paused and not self._cancelled:
                    self._play_clock += time.perf_counter() - started

    def _emit_live_frame(self, frame_no, overlay_frame, live):
        """Hand one analysed frame + its measurements to the GUI thread."""
        if self._cancelled:
            return
        self.frame_ready.emit(overlay_frame, live)

    def run(self):
        try:
            summary, paths = run_video_benchmark(
                self.video_path,
                output_dir="reports",
                preferred_signature=self.preferred_signature,
                ground_truth_path=self.ground_truth_path,
                progress_callback=lambda pct, text: self.progress.emit(int(pct), str(text)),
                cancel_check=lambda: self._cancelled,
                frame_callback=(self._emit_live_frame if self.live else None),
                pace_callback=(self._pace if self.live else None),
            )
        except (RuntimeError, OSError, ValueError) as exc:
            self.failed.emit(str(exc))
            return
        # Analysis-run PDF (report content matches the JSON/CSV bundle exactly);
        # the window's OPEN REPORT PDF opens precisely this file.
        pdf_path = None
        try:
            pdf_path, _, _ = save_full_performance_report(
                summary,
                output_dir="reports",
                prefix=f"{Path(self.video_path).stem}_benchmark",
                write_json_csv=False,
            )
        except (OSError, ValueError):
            pdf_path = None  # analysis continues honestly without a PDF
        paths = [str(p) for p in paths] + ([str(pdf_path)] if pdf_path else [])
        self.finished.emit(summary, paths, str(self.video_path),
                           self.supplied_status)

from controller import CameraController
from detector import detect_beacon
from performance_report import PerformanceSummary, save_full_performance_report
from simulator import Simulator
from tracker import BeaconTracker
from auth_config import APP_PASSWORD
from instant_report import parse_report_time, save_instant_report
from digital_twin import DigitalTwinWindow


# ---------------------------------------------------------------------------
# Small dependency-free visual widgets
# ---------------------------------------------------------------------------
class RadarWidget(QWidget):
    """World-referenced angular scope for the FSOC coarse-pointing loop.

    The radar is deliberately NOT camera-relative.  It is a stable angular
    mission view whose origin is the physical camera platform and whose
    boresight rotates as the controller slews.  This makes beacon motion,
    camera slewing, trajectory shape and Kalman prediction visible at once.

    Coordinate convention:
      - (0, 0) is the nominal camera platform reference direction.
      - camera_angles are the absolute PAN/TILT pointing direction.
      - target_angles passed to set_data are absolute beacon directions.
      - targets supplied by Simulator are camera-relative and are converted
        to absolute directions using the current camera angles.
      - predicted_angles are absolute predicted beacon directions.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Compact MINIMUM so the whole dashboard fits short work areas
        # (e.g. 1080p at 125% scaling) when maximized; the radar still grows
        # with every pixel of spare height via its stretch factor.
        # Small minimum: the radar is a hero visual that must be able to
        # shrink on a tablet / small laptop instead of forcing the window
        # wider than the screen.
        self.setMinimumSize(240, 170)
        self.target_angles = (0.0, 0.0)
        self.predicted_angles = None
        self.camera_angles = (0.0, 0.0)
        self.state = "SEARCHING"
        self.range_m = 1000.0
        self.history = []
        self.max_angle = 6.0
        self.all_targets = []
        self.designated_id = "TGT-01"

    def set_data(
        self,
        target_angles,
        predicted_angles,
        camera_angles,
        state,
        range_m,
        targets=None,
        designated_id="TGT-01",
    ):
        """Update the radar from one synchronized state snapshot.

        target_angles and camera_angles are ABSOLUTE angular directions.
        ``targets`` are Simulator camera-relative angles and are converted
        exactly once to absolute directions for the stable radar scope.
        """
        self.target_angles = (float(target_angles[0]), float(target_angles[1]))
        self.camera_angles = (float(camera_angles[0]), float(camera_angles[1]))
        self.predicted_angles = None if predicted_angles is None else (
            float(predicted_angles[0]), float(predicted_angles[1])
        )
        self.state = str(state)
        self.range_m = float(range_m)
        self.designated_id = str(designated_id)

        self.all_targets = []
        for item in list(targets or []):
            converted = dict(item)
            # Simulator provides target angle relative to current camera.
            converted["pan"] = float(item.get("pan", 0.0)) + self.camera_angles[0]
            converted["tilt"] = float(item.get("tilt", 0.0)) + self.camera_angles[1]
            self.all_targets.append(converted)

        # Keep the scene comfortably framed while preserving enough scale for
        # the camera motion and beacon trajectory to remain obvious.
        visible = [(0.0, 0.0), self.camera_angles, self.target_angles]
        if self.predicted_angles is not None:
            visible.append(self.predicted_angles)
        visible.extend(
            (float(item.get("pan", 0.0)), float(item.get("tilt", 0.0)))
            for item in self.all_targets
            if item.get("visible", True)
        )
        largest = max(
            (max(abs(pan), abs(tilt)) for pan, tilt in visible),
            default=6.0,
        )
        self.max_angle = max(3.0, min(12.0, largest + 0.8))

        # History is absolute angular target position, so motion trails remain
        # visible even while the camera is following the target.
        self.history.append(self.target_angles)
        self.history = self.history[-240:]
        self.update()

    def _point(self, angles, center, radius):
        pan, tilt = float(angles[0]), float(angles[1])
        limit = max(self.max_angle, 0.1)
        return QPointF(
            center.x() + (pan / limit) * radius,
            center.y() + (tilt / limit) * radius,
        )

    @staticmethod
    def _draw_arrow(painter, start, end, color, width=3):
        painter.setPen(QPen(color, width))
        painter.setBrush(QBrush(color))
        painter.drawLine(start, end)
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.hypot(dx, dy)
        if length < 4.0:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        size = 10.0
        left = QPointF(
            end.x() - ux * size + px * size * 0.55,
            end.y() - uy * size + py * size * 0.55,
        )
        right = QPointF(
            end.x() - ux * size - px * size * 0.55,
            end.y() - uy * size - py * size * 0.55,
        )
        painter.drawPolygon(QPolygonF([end, left, right]))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.black)

        from PySide6.QtGui import QColor
        rect = self.rect()
        center = QPointF(rect.center().x(), rect.center().y() + 8)
        # OLD-UI COMPOSITION: the scope is the dominant element of its card —
        # fill the shorter dimension instead of shrinking to 0.35 of it.
        radius = min(rect.width(), rect.height()) * 0.47

        blue = QColor(65, 155, 255)
        blue_fill = QColor(65, 155, 255, 24)
        green = QColor(50, 255, 95)
        yellow = QColor(255, 220, 55)
        purple = QColor(175, 105, 255)
        decoy = QColor(255, 90, 180)
        white = QColor(235, 240, 245)
        gray = QColor(90, 103, 115)

        painter.setPen(QPen(gray, 1))
        for fraction in (0.25, 0.50, 0.75, 1.00):
            rr = radius * fraction
            painter.drawEllipse(center, rr, rr)

        painter.setPen(QPen(QColor(55, 65, 75), 1))
        painter.drawLine(QPointF(center.x() - radius, center.y()),
                         QPointF(center.x() + radius, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - radius),
                         QPointF(center.x(), center.y() + radius))

        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(white)
        painter.drawText(int(center.x()) - 5, int(center.y() - radius) - 10, "N")
        painter.drawText(int(center.x() + radius) + 8, int(center.y()) + 4, "E")
        painter.drawText(int(center.x()) - 5, int(center.y() + radius) + 20, "S")
        painter.drawText(int(center.x() - radius) - 18, int(center.y()) + 4, "W")

        # Beacon trajectory in the stable angular mission frame.
        if len(self.history) > 1:
            painter.setPen(QPen(purple, 2))
            previous = self._point(self.history[0], center, radius)
            for angles in self.history[1:]:
                current = self._point(angles, center, radius)
                painter.drawLine(previous, current)
                previous = current

        # Camera platform is always at the radar origin. Its current
        # boresight direction is a moving endpoint.
        camera_point = self._point(self.camera_angles, center, radius)
        half_h = max(float(config.FOV_HORIZONTAL) * 0.5, 0.1)
        half_v = max(float(config.FOV_VERTICAL) * 0.5, 0.1)
        fov_center = camera_point
        fov_corners = [
            self._point((self.camera_angles[0] - half_h, self.camera_angles[1] - half_v), center, radius),
            self._point((self.camera_angles[0] + half_h, self.camera_angles[1] - half_v), center, radius),
            self._point((self.camera_angles[0] + half_h, self.camera_angles[1] + half_v), center, radius),
            self._point((self.camera_angles[0] - half_h, self.camera_angles[1] + half_v), center, radius),
        ]
        painter.setBrush(QBrush(blue_fill))
        painter.setPen(QPen(blue, 1))
        painter.drawPolygon(QPolygonF(fov_corners))
        for corner in fov_corners:
            painter.drawLine(fov_center, corner)

        # Camera platform symbol at the origin.
        painter.setBrush(QBrush(blue))
        painter.setPen(QPen(blue, 2))
        painter.drawEllipse(center, 6, 6)
        painter.drawEllipse(center, 12, 12)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(blue)
        painter.drawText(center + QPointF(16, 20), "CAMERA")

        # Moving boresight direction.
        self._draw_arrow(painter, center, camera_point, blue, 3)
        painter.setPen(blue)
        painter.drawText(camera_point + QPointF(10, -10), "BORESIGHT")

        # All targets: designated one in green, decoys in magenta.
        target_map = {str(item.get("id")): item for item in self.all_targets}
        if target_map:
            label_offsets = [(12, -8), (12, 16), (-68, -8), (-68, 16), (12, 30), (-68, 30)]
            for idx, (tid, target) in enumerate(target_map.items()):
                if not target.get("visible", True):
                    continue
                tp = self._point(
                    (target.get("pan", 0.0), target.get("tilt", 0.0)),
                    center,
                    radius,
                )
                is_designated = tid == self.designated_id
                marker_color = green if is_designated else decoy
                marker_size = 10 if is_designated else 7
                painter.setBrush(QBrush(marker_color))
                painter.setPen(QPen(QColor(240, 245, 250), 2 if is_designated else 1))
                painter.drawEllipse(tp, marker_size, marker_size)
                painter.setPen(QPen(marker_color, 1))
                painter.drawEllipse(tp, marker_size + 6, marker_size + 6)
                painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                painter.setPen(marker_color)
                ox, oy = label_offsets[idx % len(label_offsets)]
                suffix = "  DESIGNATED" if is_designated else "  DECOY"
                painter.drawText(tp + QPointF(ox, oy), tid + suffix)

        # Designated beacon marker and pointing relationship.
        target_point = self._point(self.target_angles, center, radius)
        painter.setBrush(QBrush(Qt.transparent))
        painter.setPen(QPen(green, 2))
        painter.drawEllipse(target_point, 13, 13)
        painter.drawLine(camera_point, target_point)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(green)
        painter.drawText(target_point + QPointF(-24, 32), "BEACON")

        # Kalman prediction = short-horizon future beacon direction.
        if self.predicted_angles is not None:
            predicted_point = self._point(self.predicted_angles, center, radius)
            painter.setBrush(QBrush(yellow))
            painter.setPen(QPen(yellow, 2))
            painter.drawEllipse(predicted_point, 7, 7)
            painter.drawEllipse(predicted_point, 13, 13)
            painter.drawLine(target_point, predicted_point)
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(yellow)
            painter.drawText(predicted_point + QPointF(14, 18), "KALMAN PREDICTION")

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(white)
        painter.drawText(12, 18, f"ANGULAR SCOPE  ±{self.max_angle:.1f}°")
        painter.drawText(12, 34, f"CAM PAN {self.camera_angles[0]:+.2f}°  /  TILT {self.camera_angles[1]:+.2f}°")
        painter.drawText(12, rect.height() - 12, f"RANGE {self.range_m:.0f} m")
        painter.drawText(rect.width() - 180, rect.height() - 12, f"STATE {self.state}")


class TrendGraph(QWidget):
    """Dependency-free live line graph for tracking telemetry."""

    def __init__(self, title, unit="", parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.series = []
        # 130 keeps readable axes/legend while letting the whole three-column
        # grid fit a maximized 1080p@125% work area (816 logical px).
        self.setMinimumHeight(130)

    def set_series(self, series):
        self.series = series
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.black)
        r = self.rect().adjusted(38, 28, -10, -30)
        if r.width() < 30 or r.height() < 30:
            return

        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(Qt.lightGray)
        painter.drawText(10, 18, self.title)

        painter.setPen(QPen(Qt.darkGray, 1))
        for i in range(1, 4):
            y = r.top() + r.height() * i / 4.0
            painter.drawLine(r.left(), int(y), r.right(), int(y))
        for i in range(1, 6):
            x = r.left() + r.width() * i / 6.0
            painter.drawLine(int(x), r.top(), int(x), r.bottom())

        values = []
        for _, vals, _ in self.series:
            values.extend(vals)
        if not values:
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(Qt.gray)
            painter.drawText(r.center(), "Waiting for data")
            return

        vmin = min(values)
        vmax = max(values)
        if abs(vmax - vmin) < 1e-6:
            pad = max(1.0, abs(vmax) * 0.15)
        else:
            pad = (vmax - vmin) * 0.12
        vmin -= pad
        vmax += pad

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(Qt.gray)
        painter.drawText(4, r.top() + 5, f"{vmax:.1f}")
        painter.drawText(4, r.bottom(), f"{vmin:.1f}")

        max_len = max((len(vals) for _, vals, _ in self.series), default=0)
        for name, vals, color in self.series:
            if len(vals) < 2:
                continue
            visible = vals[-max_len:]
            painter.setPen(QPen(color, 2))
            previous = None
            for i, value in enumerate(visible):
                x = r.left() + r.width() * i / max(1, len(visible) - 1)
                y = r.bottom() - ((value - vmin) / max(vmax - vmin, 1e-9)) * r.height()
                point = QPointF(x, y)
                if previous is not None:
                    painter.drawLine(previous, point)
                previous = point

        legend_x = r.left()
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        for name, _, color in self.series:
            painter.setPen(QPen(color, 2))
            painter.drawLine(int(legend_x), r.bottom() + 13, int(legend_x + 12), r.bottom() + 13)
            painter.setPen(Qt.lightGray)
            painter.drawText(int(legend_x + 16), r.bottom() + 16, name)
            legend_x += 58 + len(name) * 4

        if self.unit:
            painter.setPen(Qt.gray)
            painter.drawText(r.right() - 35, 18, self.unit)


class CameraFeedWidget(QWidget):
    """Safe, display-only camera feed widget.

    The tracker always operates on the original 640x480 NumPy frame. This
    widget only receives a QPixmap copy for display, so GUI rendering cannot
    modify tracking data.
    """

    # 640x480 feed aspect. The viewport is never allowed to grow wider than
    # this, so the painted image always fills the camera display instead of
    # sitting inside unused black (pillar) bars.
    ASPECT_RATIO = 4.0 / 3.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._status = "NO CAMERA FRAME"
        self._set_frame_count = 0
        # Compact minimum: the right column stacks Camera + Telemetry +
        # Kalman + Readiness (old-UI grid), so every panel must be able to
        # shrink on short work areas; the feed regains size via stretch.
        self.setMinimumSize(200, 90)
        # Expanding both ways so the video viewport consumes the available
        # camera-panel space at every window size (restored AND maximized).
        # The painted pixmap is always KeepAspectRatio, so the 4:3 feed is
        # never distorted; this only governs how much space the viewport gets.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("camera_view")

    def sizeHint(self):
        # A 4:3 hint keeps layout distribution honest for the 640x480 feed,
        # and it tracks the height the layout actually granted so the viewport
        # can use every pixel that the feed's aspect allows.
        height = max(90, self.height())
        return QSize(
            max(200, int(round(height * self.ASPECT_RATIO))),
            420,
        )

    def resizeEvent(self, event):  # noqa: N802 (Qt naming)
        """Keep the camera viewport at the feed's own 4:3 aspect.

        The feed is painted with KeepAspectRatio, so a viewport wider than the
        640x480 feed can only ever add unused black space around the image.
        Capping the viewport width at 4/3 of its height makes the visible
        camera image fill the whole camera display. Nothing is cropped,
        stretched or resampled differently, and the 640x480 camera resolution
        used by tracking/benchmarks is untouched.
        """
        super().resizeEvent(event)
        target = max(self.minimumWidth(),
                     int(round(max(90, self.height()) * self.ASPECT_RATIO)))
        if self.maximumWidth() != target:
            self.setMaximumWidth(target)

    def set_frame(self, pixmap, status="CAMERA FEED"):
        if pixmap is None or pixmap.isNull():
            self._pixmap = QPixmap()
            self._status = "NO CAMERA FRAME"
        else:
            self._pixmap = pixmap
            self._status = str(status)
        self._set_frame_count += 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), Qt.black)

        if not self._pixmap.isNull():
            target = self._pixmap.scaled(
                self.rect().size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
            x = (self.width() - target.width()) // 2
            y = (self.height() - target.height()) // 2
            painter.drawPixmap(x, y, target)

        painter.setPen(QPen(Qt.darkGray, 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.setPen(Qt.lightGray)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(10, self.height() - 10, self._status)
        painter.end()
        super().paintEvent(event)

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class FSOCWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASTRALINK — AI-Based Virtual Camera Tracking System for Mobile FSOC")
        self.resize(1600, 980)
        # Never launch larger than the actual screen work area — neither
        # taller (clipped when maximized, which starved the radar's stretch
        # space) nor WIDER (a window wider than the screen is precisely the
        # "I have to scroll sideways" symptom on a smaller laptop).
        available = self.screen().availableGeometry()
        work_w, work_h = available.width(), available.height()
        width, height = self.width(), self.height()
        if work_h > 100 and height > work_h - 80:
            height = max(600, work_h - 80)
        if work_w > 200 and width > work_w - 40:
            width = max(900, work_w - 40)
        if (width, height) != (self.width(), self.height()):
            self.resize(width, height)

        self.simulator = Simulator()
        self.simulator.set_target_count(1)
        self.simulator.set_designated_target("TGT-01")
        self.controller = CameraController()
        self.tracker = BeaconTracker()

        # Runtime parameters
        self.beacon_speed = 4.0
        self.beacon_range_m = 1000.0
        self.camera_fov_deg = 4.0
        self.camera_range_deg = 5.0
        # Separate, independently editable pan / tilt slew limits (deg/s).
        # camera_slew_rate stays as a convenience alias for the pan axis so
        # the existing helpers keep working unchanged.
        self.camera_pan_speed = float(config.MAX_PAN_SPEED)
        self.camera_tilt_speed = float(config.MAX_TILT_SPEED)
        self.camera_slew_rate = self.camera_pan_speed
        self.prediction_gain = 0.85
        self.beacon_size_px = 10.0
        self.noise_level = 10
        # Disturbance strengths in px, kept in step with the five-level
        # selectors (defaults: turbulence level 3, jitter level 2,
        # platform level 3).
        self.turbulence_strength = disturbance_level_to_px(3)
        self.camera_jitter_strength = disturbance_level_to_px(2)
        self.platform_motion_strength = disturbance_level_to_px(3)
        self.atmosphere_level = 25

        # Refinement 1: multi-target scenario. Keep several targets visible by
        # default so the application immediately demonstrates the PS capability.
        self.target_count = 1
        self.designated_target_id = "TGT-01"

        # Runtime state / metrics
        self.simulation_running = True
        self.previous_time = time.perf_counter()
        self.processing_times = []
        self.fps = 0.0
        self.run_start_time = time.perf_counter()
        self.frames_processed = 0
        self.visible_frames = 0
        self.tracking_frames = 0
        self.detection_misses = 0
        self.evaluation_visible_frames = 0
        self.evaluation_detection_misses = 0
        self.error_sum = 0.0
        self.error_count = 0
        self.average_error = 0.0
        self.maximum_error = 0.0
        self.error_squared_sum = 0.0
        self.centroid_error_sum = 0.0
        self.centroid_error_count = 0
        self.average_centroid_error = 0.0
        self.maximum_centroid_error = 0.0
        self.error_history = []
        self.centroid_error_history = []
        self.pan_history = []
        self.tilt_history = []
        self.metrics_started = False
        self.lock_candidate_frames = 0
        self.lock_required_frames = 5
        self.lock_error_threshold_px = 10.0
        self.acquisition_time_s = None
        self.loss_events = 0
        self.successful_reacquisitions = 0
        self.in_loss = False
        self.loss_start_time = None
        self.reacquisition_times = []
        # Forced target-loss test bookkeeping (STAGE 2). Deliberately separate
        # from the natural loss statistics: armed until the intentional loss
        # event is consumed, then it records the measured re-acquisition time.
        self.forced_loss_active = False
        self.forced_loss_start_frame = None
        self.forced_loss_frame_end = None
        self.forced_loss_reacq_s = None
        self.last_detection = None
        self.report_last_saved = None
        self.last_pdf_path = None  # latest report PDF, reopenable from the UI
        self._video_analysis_worker = None
        self._video_analysis_window = None  # keeps the analysis workspace alive
        self.camera_target_history = []
        self.camera_prediction_history = []

        # Link-readiness stability gate. A link handoff is considered ready
        # only after several consecutive good tracking frames.
        self.link_ready_streak = 0
        self.link_required_frames = config.LINK_READY_STREAK_FRAMES

        # HANDOFF WINDOW (display-only analyzer over the same readiness gate).
        # It records the continuous span during which every coarse-alignment
        # condition holds. No second simulator, no second mission state.
        self._handoff_dialog = None
        self._why_locked_dialog = None
        self._why_locked_evidence = None
        self._reset_handoff_window_state()

        # Display-only camera zoom. This NEVER changes the 640×480 tracking frame.
        self.display_zoom = 1.0

        # Refinement 5: local authentication gate + point-in-time reporting.
        self._authenticated = False
        self.instant_report_history = []
        self._last_report_snapshot = None
        self.digital_twin_window = None

        self._build_ui()
        self._apply_responsive_widths()
        self._refresh_designated_target_selector()
        self._apply_runtime_settings()
        self._reset_metrics()
        self._sync_mission_strip(force=True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._safe_update_simulation)
        self.timer.start(33)  # 30 Hz camera update target

        # Size the live column for the Virtual Camera feed once Qt has run its
        # first layout pass (see _balance_camera_column).
        QTimer.singleShot(0, self._balance_camera_column)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self._central_widget = central
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header
        header = QFrame()
        header.setObjectName("header")
        self.header = header  # twin true-fullscreen hides these chrome regions
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 10, 16, 10)
        title_box = QVBoxLayout()
        title = QLabel("ASTRALINK")
        title.setObjectName("title")
        subtitle = QLabel(
            "AI-Based Virtual Camera Tracking System for Mobile FSOC"
        )
        subtitle.setObjectName("subtitle")
        # Wrap instead of forcing the header (and the whole window) wide.
        subtitle.setWordWrap(True)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hl.addLayout(title_box)
        hl.addStretch()
        for text, obj in [
            ("● AI VERIFIED", "tag_ai"),
            ("● KALMAN ACTIVE", "tag_kalman"),
            ("● 30 Hz", "tag_rate"),
        ]:
            tag = QLabel(text)
            tag.setObjectName(obj)
            hl.addWidget(tag)
        self.system_status = QLabel("● SYSTEM ACTIVE")
        self.system_status.setObjectName("system_status")
        hl.addWidget(self.system_status)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(10)

        # ---- LEFT: single old-style SIMULATION CONTROL column -------------
        # One scrollable engineering panel (old-UI reference): mission buttons,
        # test/output launchers, target/motion, pointing, disturbances.
        self.control_scroll = QScrollArea()
        self.control_scroll.setObjectName("control_scroll")
        self.control_scroll.setWidgetResizable(True)
        # RESPONSIVE CONTROL COLUMN.  The old 300-360 px clamp cut the
        # TEST/OUTPUT launchers off and forced a horizontal scrollbar.  The
        # column now tracks the window width (see _apply_responsive_widths)
        # within a readable range, and sideways scrolling is disabled outright
        # because the wrapping FlowLayouts below guarantee everything fits.
        self.control_scroll.setMinimumWidth(300)
        self.control_scroll.setMaximumWidth(460)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.control_scroll.setFrameShape(QFrame.NoFrame)
        controls = QFrame()
        controls.setObjectName("panel")
        # The scroll container itself is the hideable left column (true
        # fullscreen); hiding it takes the inner frame with it.
        self.sidebar = self.control_scroll
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(6)
        self.control_scroll.setWidget(controls)
        # Compatibility alias: layout/stability checks watch this geometry as
        # the left control column.
        self.control_stack = self.control_scroll
        self.section_buttons = []

        # ---- section: MISSION (primary controls) ---------------------------
        mission_title = QLabel("SIMULATION CONTROL")
        mission_title.setObjectName("panel_title")
        mission_title.setWordWrap(True)
        mission_title.setMinimumWidth(0)
        cl.addWidget(mission_title)

        # Wrapping action row: three across when there is room, fewer when not.
        action = FlowLayout(h_spacing=6, v_spacing=6)
        self.start_button = QPushButton("▶  START")
        self.pause_button = QPushButton("⏸  PAUSE")
        self.reset_button = QPushButton("↻  RESET")
        self.start_button.setObjectName("start_button")
        self.pause_button.setObjectName("pause_button")
        self.reset_button.setObjectName("reset_button")
        for btn in [self.start_button, self.pause_button, self.reset_button]:
            action.addWidget(btn)
        cl.addLayout(action)
        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)

        scenario = QFrame()
        scenario.setObjectName("subpanel")
        sg = QGridLayout(scenario)
        scen_title = QLabel("TARGET & MOTION")
        scen_title.setObjectName("section_title")
        scen_title.setMinimumWidth(0)
        sg.addWidget(scen_title, 0, 0, 1, 2)
        sg.setContentsMargins(8, 6, 8, 6)
        sg.setHorizontalSpacing(8)
        sg.setVerticalSpacing(4)
        sg.addWidget(self._label("Beacon Motion"), 1, 0)
        self.motion_selector = QComboBox()
        self.motion_selector.addItems(["Straight Line", "Circular", "Figure 8", "Random"])
        sg.addWidget(self.motion_selector, 1, 1)
        sg.addWidget(self._label("Moving Targets"), 2, 0)
        self.target_count_selector = QComboBox()
        self.target_count_selector.addItems(["1", "2", "3", "4", "5"])
        self.target_count_selector.setCurrentText(str(self.target_count))
        sg.addWidget(self.target_count_selector, 2, 1)
        sg.addWidget(self._label("Designated Target"), 3, 0)
        self.designated_target_selector = QComboBox()
        self.designated_target_selector.setEditable(False)
        self.designated_target_selector.setCurrentText(self.designated_target_id)
        sg.addWidget(self.designated_target_selector, 3, 1)

        self.target_help = QLabel("Green = designated communication target  •  Pink = decoys")
        self.target_help.setObjectName("caption")
        self.target_help.setWordWrap(True)
        sg.addWidget(self.target_help, 4, 0, 1, 2)

        self.motion_selector.currentTextChanged.connect(self.change_motion_pattern)
        self.target_count_selector.currentTextChanged.connect(self._on_target_count_changed)
        self.designated_target_selector.currentTextChanged.connect(self._on_designated_target_changed)
        cl.addWidget(scenario)

        parameter_box = QFrame()
        parameter_box.setObjectName("subpanel")
        pg = QGridLayout(parameter_box)
        pg.setContentsMargins(8, 6, 8, 6)
        pg.setHorizontalSpacing(8)
        pg.setVerticalSpacing(4)
        param_title = QLabel("POINTING & OPTICS")
        param_title.setObjectName("section_title")
        param_title.setMinimumWidth(0)
        param_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        pg.addWidget(param_title, 0, 0, 1, 2)

        self.speed_slider, self.speed_value = self._make_slider(1, 100, 40, 10, "px/frame")
        self.range_slider, self.range_value = self._make_slider(1, 100, 10, 1, "m×100")
        self.fov_slider, self.fov_value = self._make_slider(10, 100, 40, 10, "°")
        self.cam_range_slider, self.cam_range_value = self._make_slider(10, 100, 50, 10, "±°")
        # Max Pan Speed / Max Tilt Speed: two independent, editable limits.
        # slew_slider / slew_value remain as aliases for the pan axis so the
        # existing single-control helpers keep working.
        self.pan_speed_slider, self.pan_speed_value = self._make_slider(10, 100, 50, 10, "°/s")
        self.tilt_speed_slider, self.tilt_speed_value = self._make_slider(10, 100, 50, 10, "°/s")
        self.slew_slider = self.pan_speed_slider
        self.slew_value = self.pan_speed_value
        self.pred_slider, self.pred_value = self._make_slider(0, 100, 85, 100, "gain")
        self.size_slider, self.size_value = self._make_slider(5, 20, 10, 1, "px")
        self.noise_level_slider, self.noise_level_value = self._make_slider(0, 20, 10, 1, "level")
        # FIVE discrete intensity levels per disturbance category. The
        # slider value IS the level (1..5); _apply_disturbances maps it onto
        # the simulator's px strength so the existing maths is untouched.
        self.turbulence_slider, self.turbulence_value = self._make_slider(1, 5, 3, 1, "level")
        self.jitter_slider, self.jitter_value = self._make_slider(1, 5, 2, 1, "level")
        self.platform_slider, self.platform_value = self._make_slider(1, 5, 3, 1, "level")
        self.atmosphere_slider, self.atmosphere_value = self._make_slider(0, 100, 25, 1, "%")
        _level_tooltip = "5 intensity levels: " + " · ".join(
            f"{index} {name}"
            for index, name in enumerate(DISTURBANCE_LEVEL_NAMES, start=1)
        )
        for _level_slider, _level_value in (
            (self.turbulence_slider, self.turbulence_value),
            (self.jitter_slider, self.jitter_value),
            (self.platform_slider, self.platform_value),
        ):
            _level_slider.setTickPosition(QSlider.TicksBelow)
            _level_slider.setTickInterval(1)
            _level_slider.setToolTip(_level_tooltip)
            # The level caption WRAPS when the control column is narrow, so it
            # is always fully readable and never widens the column.
            _level_value.setWordWrap(True)
            _level_value.setText(disturbance_level_text(_level_slider.value()))
            _level_value.setToolTip(_level_tooltip)

        specs = [
            ("Beacon Speed", self.speed_slider, self.speed_value),
            ("Beacon Range", self.range_slider, self.range_value),
            ("Camera FOV", self.fov_slider, self.fov_value),
            ("Pan/Tilt Range", self.cam_range_slider, self.cam_range_value),
            ("Max Pan Speed", self.pan_speed_slider, self.pan_speed_value),
            ("Max Tilt Speed", self.tilt_speed_slider, self.tilt_speed_value),
            ("Prediction Gain", self.pred_slider, self.pred_value),
            ("Beacon Size", self.size_slider, self.size_value),
        ]
        for index, (label_text, slider, value_label) in enumerate(specs):
            row, col = divmod(index, 2)
            card = self._slider_card(label_text, slider, value_label)
            pg.addWidget(card, row, col)
        cl.addWidget(parameter_box)

        # ---- SIMULATION SPECIFICATION (read only) -------------------------
        # Values are read from the SAME config/settings the live mission uses.
        # Screen Size is the virtual simulation/environment surface only; the
        # application window itself stays responsive and is never forced to
        # 2000x2000.
        spec_box = QFrame()
        spec_box.setObjectName("subpanel")
        spg = QGridLayout(spec_box)
        spg.setContentsMargins(8, 6, 8, 6)
        spg.setHorizontalSpacing(8)
        spg.setVerticalSpacing(3)
        spec_title = QLabel("SIMULATION SPECIFICATION")
        spec_title.setObjectName("section_title")
        spec_title.setWordWrap(True)
        spec_title.setMinimumWidth(0)
        spg.addWidget(spec_title, 0, 0, 1, 2)
        self.spec_screen_value = QLabel()
        self.spec_resolution_value = QLabel()
        self.spec_type_value = QLabel()
        self.spec_interval_value = QLabel()
        self.spec_range_value = QLabel()
        for row, (label_text, value_widget) in enumerate([
            ("Screen Size", self.spec_screen_value),
            ("Camera Resolution", self.spec_resolution_value),
            ("Camera Type", self.spec_type_value),
            ("Update Interval", self.spec_interval_value),
            ("Camera Update Range", self.spec_range_value),
        ], start=1):
            spg.addWidget(self._label(label_text), row, 0)
            value_widget.setObjectName("spec_value")
            value_widget.setWordWrap(True)
            value_widget.setMinimumWidth(0)
            value_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            spg.addWidget(value_widget, row, 1)
        cl.addWidget(spec_box)

        disturbances = QFrame()
        disturbances.setObjectName("subpanel")
        dg = QGridLayout(disturbances)
        dg.setContentsMargins(8, 6, 8, 6)
        dg.setHorizontalSpacing(8)
        dg.setVerticalSpacing(4)
        dt = QLabel("DISTURBANCES")
        dt.setObjectName("section_title")
        dt.setMinimumWidth(0)
        dg.addWidget(dt, 0, 0, 1, 3)
        # Deterministic disturbance profile presets (real simulator feature,
        # previously reachable only through tests).
        dg.addWidget(self._label("Disturbance Preset"), 1, 0)
        self.preset_selector = QComboBox()
        self.preset_selector.addItems(
            ["Clear", "Sensor Noise", "Vibration", "Atmosphere", "Full Combined", "Extreme PAT"]
        )
        self.preset_selector.currentTextChanged.connect(self._apply_disturbance_preset)
        dg.addWidget(self.preset_selector, 1, 1)
        # Disturbance family selectors; the mission strip references them too.
        self.noise_selector = QComboBox()
        self.noise_selector.addItems(["None", "Salt & Pepper", "Gaussian", "Poisson"])
        self.platform_pattern_selector = QComboBox()
        self.platform_pattern_selector.addItems(["Linear", "Circular", "Random", "Figure 8"])
        self.atmosphere_selector = QComboBox()
        self.atmosphere_selector.addItems(["Clear", "Haze", "Fog", "Rain", "Low Light"])
        self.noise_selector.currentTextChanged.connect(self.change_noise_type)
        self.platform_pattern_selector.currentTextChanged.connect(self._apply_disturbances)
        self.atmosphere_selector.currentTextChanged.connect(self._apply_disturbances)
        dg.addWidget(self._label("Noise Type"), 2, 0)
        dg.addWidget(self.noise_selector, 2, 1)
        dg.addWidget(self._label("Atmosphere"), 3, 0)
        dg.addWidget(self.atmosphere_selector, 3, 1)
        dg.addWidget(self._label("Platform Pattern"), 4, 0)
        dg.addWidget(self.platform_pattern_selector, 4, 1)

        self.turbulence_check = QCheckBox("Atmospheric Turbulence")
        self.jitter_check = QCheckBox("Camera Jitter")
        self.platform_check = QCheckBox("Platform Motion")
        self.turbulence_check.setChecked(False)
        self.jitter_check.setChecked(False)
        self.platform_check.setChecked(False)
        # The three toggles wrap: side by side when the column is wide, stacked
        # when it is narrow. Their combined width was the single largest driver
        # of the control column's minimum width (and therefore of the old
        # horizontal scrollbar).
        toggle_row = FlowLayout(h_spacing=8, v_spacing=2)
        for toggle in (self.turbulence_check, self.jitter_check, self.platform_check):
            toggle_row.addWidget(toggle)
        dg.addLayout(toggle_row, 5, 0, 1, 3)

        for row, label_text, slider, value_label in [
            (6, "Atmospheric Turbulence", self.turbulence_slider, self.turbulence_value),
            (7, "Camera Jitter", self.jitter_slider, self.jitter_value),
            (8, "Platform Motion", self.platform_slider, self.platform_value),
            (9, "Atmosphere Intensity", self.atmosphere_slider, self.atmosphere_value),
            (10, "Noise Level", self.noise_level_slider, self.noise_level_value),
        ]:
            dg.addWidget(self._label(label_text), row, 0)
            dg.addWidget(slider, row, 1)
            dg.addWidget(value_label, row, 2)
        # The five intensity levels are named explicitly, once per category,
        # so the scale is unambiguous without duplicating controls.
        level_caption = QLabel(
            "Each disturbance has 5 intensity levels: "
            "Level 1 — Very Low  ·  Level 2 — Low  ·  Level 3 — Medium  ·  "
            "Level 4 — High  ·  Level 5 — Very High"
        )
        level_caption.setObjectName("caption")
        level_caption.setWordWrap(True)
        level_caption.setMinimumWidth(0)
        dg.addWidget(level_caption, 11, 0, 1, 3)
        self.turbulence_check.stateChanged.connect(self._apply_disturbances)
        self.jitter_check.stateChanged.connect(self._apply_disturbances)
        self.platform_check.stateChanged.connect(self._apply_disturbances)

        for slider, handler in [
            (self.speed_slider, self._on_speed_changed),
            (self.range_slider, self._on_range_changed),
            (self.fov_slider, self._on_fov_changed),
            (self.cam_range_slider, self._on_camera_range_changed),
            (self.pan_speed_slider, self._on_pan_speed_changed),
            (self.tilt_speed_slider, self._on_tilt_speed_changed),
            (self.pred_slider, self._on_prediction_changed),
            (self.size_slider, self._on_size_changed),
            (self.noise_level_slider, self._on_noise_level_changed),
            (self.turbulence_slider, self._apply_disturbances),
            (self.jitter_slider, self._apply_disturbances),
            (self.platform_slider, self._apply_disturbances),
            (self.atmosphere_slider, self._apply_disturbances),
        ]:
            slider.valueChanged.connect(handler)

        # FORCE LOSS / RECOVERY group (old-UI TEST LOSS / RECOVER placement).
        loss_box = QFrame()
        loss_box.setObjectName("subpanel")
        lgx = QVBoxLayout(loss_box)
        lgx.setContentsMargins(8, 6, 8, 6)
        lgx.setSpacing(4)
        loss_title = QLabel("FORCE LOSS / RECOVERY")
        loss_title.setObjectName("section_title")
        loss_title.setWordWrap(True)
        loss_title.setMinimumWidth(0)
        lgx.addWidget(loss_title)
        self.force_loss_button = QPushButton("⚠  TEST LOSS / RECOVER")
        self.force_loss_button.setObjectName("force_loss_button")
        self.force_loss_button.clicked.connect(self.force_target_loss)
        lgx.addWidget(self.force_loss_button)
        cl.addWidget(loss_box)

        self.info = QLabel()
        self.info.setObjectName("info_box")
        self.info.setWordWrap(True)
        cl.addWidget(self.info)
        cl.addStretch()

        # TEST/OUTPUT launchers (old-UI second group): secondary screens open
        # as dedicated windows from the control column.
        output_box = QFrame()
        output_box.setObjectName("subpanel")
        og = QVBoxLayout(output_box)
        og.setContentsMargins(8, 6, 8, 6)
        og.setSpacing(4)
        output_title = QLabel("TEST / OUTPUT")
        output_title.setObjectName("section_title")
        output_title.setWordWrap(True)
        output_title.setMinimumWidth(0)
        og.addWidget(output_title)
        self.benchmark_button = QPushButton("🎞  BENCHMARK VIDEO")
        self.report_button = QPushButton("📊  EXPORT REPORT")
        self.instant_report_button = QPushButton("⏱  INSTANT REPORT")
        self.twin_launch_button = QPushButton("★  3D DIGITAL TWIN")
        self.twin_launch_button.setObjectName("twin_launch_button")
        # Launcher row: BENCHMARK VIDEO / EXPORT REPORT / INSTANT REPORT /
        # 3D DIGITAL TWIN flow two-per-row when the panel is wide enough and
        # stack one-per-row on narrow screens — never cut off, never scrolled.
        launch_row = FlowLayout(h_spacing=6, v_spacing=6)
        for launcher in (self.benchmark_button, self.report_button,
                         self.instant_report_button, self.twin_launch_button):
            launch_row.addWidget(launcher)
        og.addLayout(launch_row)
        self.benchmark_button.clicked.connect(self.benchmark_video)
        self.report_button.clicked.connect(self.export_report)
        self.instant_report_button.clicked.connect(self.generate_instant_report)
        self.twin_launch_button.clicked.connect(self.open_digital_twin)

        # 5. BENCHMARK PERFORMANCE GRAPHS (time-series visualizations)
        # Dedicated graph panel in the TEST/OUTPUT section (separate from
        # the benchmark video dialog): loads the latest benchmark telemetry
        # CSV and renders 5 time-series graphs.
        from benchmark_graphs import BenchmarkGraphPanel
        self.graph_panel = BenchmarkGraphPanel()
        self.graph_panel.setMinimumHeight(280)
        self.graph_panel.setSizePolicy(
            __import__('PySide6.QtWidgets', fromlist=['QSizePolicy']).QSizePolicy.Expanding,
            __import__('PySide6.QtWidgets', fromlist=['QSizePolicy']).QSizePolicy.Preferred
        )
        self.graph_panel.setObjectName("panel")
        graph_title = QLabel("BENCHMARK PERFORMANCE GRAPHS")
        graph_title.setObjectName("section_title")
        graph_title.setWordWrap(True)
        graph_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        graph_title.setMinimumWidth(0)
        graph_layout = QHBoxLayout()
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(8)
        graph_layout.addWidget(graph_title)
        graph_layout.addStretch()
        og.addLayout(graph_layout)
        # The panel owns the only status line: one source of truth for
        # "run the benchmark to generate performance data" / "Loaded N frames".
        og.addWidget(self.graph_panel, 1)

        self._benchmark_telemetry_paths = []
        self.benchmark_graphs_button = QPushButton("📈  VIEW BENCHMARK GRAPHS")
        self.benchmark_graphs_button.setObjectName("twin_launch_button")
        self.benchmark_graphs_button.setToolTip(
            "Open the benchmark performance graph panel with loaded telemetry"
        )
        self.benchmark_graphs_button.clicked.connect(self._open_benchmark_graphs)
        og.addWidget(self.benchmark_graphs_button)
        cl.addWidget(output_box)

        # ---- DISTURBANCES subpanel (absorbed TEST LAB) ---------------------
        cl.addWidget(disturbances)

        # ---- main area: shared live-view pages (ONE mission state) --------
        # Page 0: MISSION OPS — radar and virtual camera SIDE BY SIDE with the
        # telemetry / graphs / Kalman / handoff band along the bottom. Both
        # live views always share this single page; the top view switch only
        # changes which panel is visually emphasized.
        self.main_stack = QStackedWidget()
        self.main_stack.setObjectName("main_stack")

        ops_page = QWidget()
        ops = QVBoxLayout(ops_page)
        ops.setContentsMargins(0, 0, 0, 0)
        ops.setSpacing(10)
        views_row = QHBoxLayout()
        views_row.setSpacing(10)

        self.radar_panel = QFrame()
        self.radar_panel.setObjectName("panel")
        rl = QVBoxLayout(self.radar_panel)
        rl.setContentsMargins(10, 10, 10, 8)
        radar_header = QHBoxLayout()
        radar_title = QLabel("VIRTUAL FSOC RADAR")
        radar_title.setObjectName("panel_title")
        radar_header.addWidget(radar_title)
        radar_header.addStretch()
        self.radar_state = QLabel("● SEARCHING")
        self.radar_state.setObjectName("radar_state")
        self.radar_state.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        radar_header.addWidget(self.radar_state)
        rl.addLayout(radar_header)
        radar_caption = QLabel("Target trajectory  •  Kalman prediction  •  Camera boresight  •  FOV")
        radar_caption.setObjectName("caption")
        radar_caption.setWordWrap(True)
        rl.addWidget(radar_caption)
        self.radar = RadarWidget()
        rl.addWidget(self.radar, 1)
        # (the radar panel is placed by the center_column layout below —
        #  radar card on top, graphs card beneath it)

        legend = QFrame()
        legend.setObjectName("legend_panel")
        lg = QGridLayout(legend)
        lg.setContentsMargins(8, 5, 8, 5)
        lg.setHorizontalSpacing(12)
        lg.setVerticalSpacing(2)
        for index, (color, text) in enumerate([
            ("#39ff66", "GREEN  Beacon"),
            ("#ffd83d", "YELLOW  Kalman prediction"),
            ("#4aa3ff", "BLUE  Camera boresight / FOV"),
            ("#a66cff", "PURPLE  Target trail"),
        ]):
            label = QLabel(f'<span style="color:{color}">●</span> {text}')
            label.setObjectName("legend_item")
            label.setWordWrap(True)
            lg.addWidget(label, index // 2, index % 2)
        rl.addWidget(legend)
        # RADAR and VIRTUAL CAMERA sit side by side on ONE page. The top
        # LIVE VIEW switch only moves a visual emphasis outline between them;
        # it never re-pages, hides, or resets either view.
        self._view_emphasis = 0

        graph_panel = QFrame()
        graph_panel.setObjectName("panel")
        gg = QGridLayout(graph_panel)
        gg.setContentsMargins(8, 8, 8, 8)
        gg.setHorizontalSpacing(8)
        self.error_graph = TrendGraph("TRACKING / CENTROID ERROR", "px")
        self.motion_graph = TrendGraph("PAN / TILT CONTROL RESPONSE", "deg")
        gg.addWidget(self.error_graph, 0, 0)
        gg.addWidget(self.motion_graph, 0, 1)
        gg.setColumnStretch(0, 1)
        gg.setColumnStretch(1, 1)
        # OLD-UI STRUCTURE: the graphs live in their OWN clean card directly
        # below the radar card in the center column — never inside the radar
        # card, never on top of it, never in a separate full-width row.
        center_column = QVBoxLayout()
        center_column.setContentsMargins(0, 0, 0, 0)
        center_column.setSpacing(8)
        center_column.addWidget(self.radar_panel, 1)
        center_column.addWidget(graph_panel, 0)
        views_row.addLayout(center_column, 2)
        # The graphs live in the shared BOTTOM BAND (telemetry | error graph |
        # control response | Kalman | handoff), spanning radar + camera.

        # Page 1: 3D DIGITAL TWIN (added below); page 2: REPLAY & REPORTS.
        self.main_stack.addWidget(ops_page)

        # VIRTUAL CAMERA panel — permanently side by side with the radar.
        camera_panel = QFrame()
        camera_panel.setObjectName("panel")
        self.camera_panel = camera_panel
        cpl = QVBoxLayout(camera_panel)
        # Tightest inner padding: the camera feed -- not the padding around it --
        # owns the panel. The ZOOM row stays above the feed, untouched.
        # Final trim: every pixel of pure chrome goes to the feed; nothing
        # visible (title, badge, info line, zoom controls) moves or resizes.
        cpl.setContentsMargins(2, 1, 2, 1)
        cpl.setSpacing(2)
        cam_head = QHBoxLayout()
        cam_title = QLabel("VIRTUAL CAMERA")
        cam_title.setObjectName("panel_title")
        cam_head.addWidget(cam_title)
        cam_head.addStretch()
        self.camera_lock_badge = QLabel("● SEARCHING")
        self.camera_lock_badge.setObjectName("camera_badge")
        self.camera_lock_badge.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        cam_head.addWidget(self.camera_lock_badge)
        cpl.addLayout(cam_head)
        self.camera_info = QLabel()
        self.camera_info.setObjectName("caption")
        # Supplementary status line: may elide at narrow widths but must never
        # force the camera panel (or the window) wider than the screen.
        self.camera_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        cpl.addWidget(self.camera_info)
        self.camera_label = CameraFeedWidget()
        # The aspect-clamped camera display is centred in its panel, so no
        # unused black area is ever painted inside the camera view.
        feed_row = QHBoxLayout()
        feed_row.setContentsMargins(0, 0, 0, 0)
        feed_row.setSpacing(0)
        feed_row.addStretch(1)
        feed_row.addWidget(self.camera_label, 0)
        feed_row.addStretch(1)
        cpl.addLayout(feed_row, 1)

        # Display-only zoom controls. Tracking continues to use the original
        # 640×480 camera frame, so zoom cannot change benchmark accuracy.
        # They live in the TITLE ROW so the feed's stretch owns every pixel
        # below it — the camera is a hero visual, not a strip.
        zoom_label = QLabel("ZOOM")
        zoom_label.setObjectName("small_label")
        cam_head.addWidget(zoom_label)

        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setObjectName("zoom_button")
        self.zoom_out_button.setToolTip("Zoom out")
        self.zoom_out_button.clicked.connect(lambda: self._step_zoom(-0.5))
        cam_head.addWidget(self.zoom_out_button)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 40)
        self.zoom_slider.setValue(10)
        self.zoom_slider.setToolTip("Display zoom: 1× to 4×")
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        cam_head.addWidget(self.zoom_slider, 2)

        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setObjectName("zoom_button")
        self.zoom_in_button.setToolTip("Zoom in")
        self.zoom_in_button.clicked.connect(lambda: self._step_zoom(0.5))
        cam_head.addWidget(self.zoom_in_button)

        self.zoom_reset_button = QPushButton("1×")
        self.zoom_reset_button.setObjectName("zoom_reset_button")
        self.zoom_reset_button.setToolTip("Reset display zoom to 1×")
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        cam_head.addWidget(self.zoom_reset_button)

        self.zoom_value_label = QLabel("1.0×")
        self.zoom_value_label.setObjectName("zoom_value")
        self.zoom_value_label.setMinimumWidth(34)
        cam_head.addWidget(self.zoom_value_label)

        telemetry = QFrame()
        telemetry.setObjectName("panel")
        tl = QGridLayout(telemetry)
        tl.setContentsMargins(6, 6, 6, 6)
        tl.setHorizontalSpacing(5)
        tl.setVerticalSpacing(3)
        tt = QLabel("LIVE TELEMETRY")
        tt.setObjectName("panel_title")
        tl.addWidget(tt, 0, 0, 1, 4)
        self.t_fps = QLabel("0.0")
        self.t_error = QLabel("0.00 px")
        self.t_centroid = QLabel("—")
        self.t_avg = QLabel("0.00 px")
        self.t_max = QLabel("0.00 px")
        self.t_rmse = QLabel("0.00 px")
        self.t_acq = QLabel("—")
        self.t_reacq = QLabel("—")
        self.t_loss = QLabel("0.0 %")
        self.t_lock = QLabel("—")
        self.t_state = QLabel("SEARCHING")
        self.t_pan = QLabel("0.00°")
        self.t_tilt = QLabel("0.00°")
        self.t_velocity = QLabel("0.00 px/s")
        self.t_confidence = QLabel("—")
        self.t_ai = QLabel("—")

        # 4x4 metric grid in the PS review order — every metric gets the
        # same card treatment and equal sizing, with no truncation. The
        # title owns row 0; the sixteen cards occupy rows 1-4. Values keep
        # full precision — nothing is wrapped, shrunk or hidden.
        grid_metrics = [
            ("Processing FPS", self.t_fps, "888.8"),
            ("Tracking Error", self.t_error, "-888.88 px"),
            ("Center Error", self.t_centroid, "-888.88 px"),
            ("Acquisition Time", self.t_acq, "888.88 s"),
            ("Re-acquisition Time", self.t_reacq, "888.88 s"),
            ("Target Loss", self.t_loss, "888.88 %"),
            ("Avg Tracking Error", self.t_avg, "-888.88 px"),
            ("Max Deviation", self.t_max, "-888.88 px"),
            ("RMSE", self.t_rmse, "-888.88 px"),
            ("Lock Retention", self.t_lock, "888.88 %"),
            ("State", self.t_state, "RE-ACQUIRING"),
            ("Pan", self.t_pan, "-888.88°"),
            ("Tilt", self.t_tilt, "-888.88°"),
            ("Velocity", self.t_velocity, "-888.88 px/s"),
            ("Filter Confidence", self.t_confidence, "0.888"),
            ("AI Confidence", self.t_ai, "0.888"),
        ]
        # Responsive metric grid: the cards are kept in a list and re-flowed
        # (4 / 2 / 1 columns) by _relayout_metric_grid() as the window narrows,
        # so the telemetry block never forces a horizontal scrollbar.
        self._telemetry_grid = tl
        self._telemetry_title = tt
        self._metric_cards = []
        self._metric_grid_columns = None
        for index, (label_text, value_label, sample_text) in enumerate(grid_metrics):
            card = QFrame()
            card.setObjectName("metric_card")
            # COMPACT single-line card (label left, value right): the right
            # column stacks Camera + Telemetry + Kalman + Readiness, so the
            # telemetry block must stay a thin supporting strip and leave the
            # vertical space to the two hero visuals (radar, camera).
            ml = QHBoxLayout(card)
            ml.setContentsMargins(5, 1, 5, 1)
            ml.setSpacing(4)
            name = QLabel(label_text)
            name.setObjectName("metric_label")
            name.setToolTip(label_text)
            value_label.setObjectName("metric_value")
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            # Reserve the EXACT styled text widths for both the name and the
            # value, so neither can ever be clipped. The measuring fonts
            # mirror the #metric_label / #metric_value stylesheet sizes.
            label_font = QFont(name.font())
            label_font.setPixelSize(9)
            value_font = QFont(value_label.font())
            value_font.setPixelSize(12)
            value_font.setBold(True)
            value_label.setMinimumWidth(
                int(QFontMetrics(value_font).horizontalAdvance(sample_text) * 1.1 + 10)
            )
            value_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
            # The name WRAPS instead of clipping (or eliding) and, because it
            # keeps an Ignored width policy, it can never inflate the window's
            # minimum width — so the telemetry strip stays readable at every
            # window size without ever introducing a horizontal scrollbar.
            name.setWordWrap(True)
            name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            name.setMinimumWidth(0)
            name.setToolTip(f"{label_text}  •  sample width "
                            f"{QFontMetrics(label_font).horizontalAdvance(label_text)}px")
            ml.addWidget(name, 1)
            ml.addWidget(value_label, 0)
            self._metric_cards.append(card)
        self._relayout_metric_grid(4)

        # KALMAN PREDICTION ENGINE lives in the RIGHT column (old-UI grid).
        kalman_panel = QFrame()
        kalman_panel.setObjectName("kalman_panel")
        kl = QGridLayout(kalman_panel)
        kl.setContentsMargins(8, 6, 8, 6)
        kl.setHorizontalSpacing(12)
        kt = QLabel("KALMAN PREDICTION ENGINE")
        kt.setObjectName("panel_title")
        kl.addWidget(kt, 0, 0, 1, 2)
        self.kalman_state_label = QLabel("Prediction: WAITING")
        self.kalman_state_label.setObjectName("kalman_state")
        self.kalman_pos_label = QLabel("Predicted Position: —")
        self.kalman_vel_label = QLabel("Estimated Velocity: —")
        self.kalman_uncertainty_label = QLabel("Position Uncertainty: —")
        self.kalman_next_label = QLabel("Control Lead: —")
        for widget in [self.kalman_pos_label, self.kalman_vel_label, self.kalman_uncertainty_label, self.kalman_next_label]:
            widget.setObjectName("kalman_text")
        # Live values flow through these labels every frame. With a Preferred
        # width each text change resized the label, reflowing the whole bottom
        # band — the observed Mission Control jitter. Ignored decouples the
        # label from its text entirely: the layout assigns the width (minimum
        # reserved below), so live values can neither move siblings nor grow
        # the window minimum when a value gets longer.
        for widget in [self.kalman_state_label, self.kalman_pos_label,
                       self.kalman_vel_label, self.kalman_uncertainty_label,
                       self.kalman_next_label]:
            widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            widget.setWordWrap(True)
            # No per-label height reservation: at the right column's width the
            # values render on one line, and a zero minimum lets the panel
            # compact on short (maximized 1080p@125%) work areas instead of
            # pushing the whole window minimum past the screen.
        kt.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        kl.addWidget(self.kalman_state_label, 1, 0, 1, 2)
        # 2x2 value grid: all four Kalman readouts stay visible while the
        # panel stays a thin supporting strip under the camera (the hero).
        kl.addWidget(self.kalman_pos_label, 2, 0)
        kl.addWidget(self.kalman_vel_label, 2, 1)
        kl.addWidget(self.kalman_uncertainty_label, 3, 0)
        kl.addWidget(self.kalman_next_label, 3, 1)
        readiness_panel = QFrame()
        readiness_panel.setObjectName("readiness_panel")
        rgl = QGridLayout(readiness_panel)
        rgl.setContentsMargins(8, 6, 8, 6)
        rgl.setHorizontalSpacing(10)
        rgl.setVerticalSpacing(3)
        readiness_title = QLabel("LINK READINESS")
        readiness_title.setObjectName("panel_title")
        rgl.addWidget(readiness_title, 0, 0, 1, 2)
        self.link_readiness_label = QLabel("● NOT READY")
        self.link_readiness_label.setObjectName("link_not_ready")
        # Both states (● HANDOFF READY / ● RECOVERING • LINK NOT READY, etc.)
        # render at the same stylesheet size; reserving the taller of the
        # possible line heights keeps the panel height constant when the
        # objectName (and style) flips between states.
        self.link_readiness_label.setMinimumHeight(
            max(self.link_readiness_label.fontMetrics().height(),
                self.link_readiness_label.fontMetrics().height()) + 6
        )
        self.link_readiness_label.setToolTip(
            "Coarse-alignment readiness only. A physical FSOC communication link may still require fine pointing and hardware verification."
        )
        rgl.addWidget(self.link_readiness_label, 1, 0, 1, 2)
        self.link_conditions_label = QLabel("Tracking: —  |  Error: —  |  FPS: —  |  Confidence: —")
        self.link_conditions_label.setObjectName("caption")
        self.link_conditions_label.setWordWrap(True)
        # Reserve two caption lines: live values change this wrapped text's
        # line count (1 <-> 2), which previously resized the panel every time
        # it flipped — the observed vertical jitter in the readiness panel.
        self.link_conditions_label.setMinimumHeight(
            self.link_conditions_label.fontMetrics().height() * 2 + 4
        )
        rgl.addWidget(self.link_conditions_label, 2, 0, 1, 2)
        self.handoff_window_button = QPushButton("⇄  HANDOFF WINDOW")
        self.handoff_window_button.setToolTip(
            "Open the live handoff-window readout: the continuous span during "
            "which every coarse-alignment handoff condition holds."
        )
        self.handoff_window_button.clicked.connect(self._open_handoff_window)
        # Both evidence buttons share one compact row (readiness is a
        # supporting panel; the vertical space belongs to the heroes).
        self.why_locked_button = QPushButton("❓  WHY LOCKED?")
        self.why_locked_button.setToolTip(
            "Explain the current lock/handoff state from the REAL telemetry: "
            "every condition with its actual value and limit."
        )
        self.why_locked_button.clicked.connect(self._open_why_locked)
        # Equal-size action row: identical width/height/alignment so live
        # status text can never reshape the buttons.
        for b in (self.handoff_window_button, self.why_locked_button):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(28)
        self.link_readiness_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.link_conditions_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        # Both evidence buttons share one compact row (readiness is a
        # supporting panel; the vertical space belongs to the heroes).
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self.handoff_window_button)
        btn_row.addWidget(self.why_locked_button)
        rgl.addLayout(btn_row, 3, 0, 1, 2)
        # OLD-UI GRID (three columns, no global bottom row):
        #   LEFT   control column (built earlier)
        #   CENTER radar card on top + the two wide graphs beneath it
        #   RIGHT  Virtual Camera, then LIVE TELEMETRY / KALMAN /
        #          LINK READINESS stacked underneath it
        right_column = QVBoxLayout()
        right_column.setSpacing(6)
        right_column.addWidget(camera_panel, 1)
        right_column.addWidget(telemetry, 0)
        # KALMAN + LINK READINESS share ONE row whenever the camera column is
        # wide enough to hold both at their full content width (see
        # _relayout_support_row). Sharing a row costs them nothing -- each
        # keeps the minimum height Qt measures for what it already displays --
        # and the height they stop using is handed straight to the Virtual
        # Camera feed. On a narrow window they stack exactly as before, so no
        # value, label or button is ever squeezed.
        self._kalman_panel = kalman_panel
        self._readiness_panel = readiness_panel
        self._support_row = QFrame()
        support_row_layout = QHBoxLayout(self._support_row)
        support_row_layout.setContentsMargins(0, 0, 0, 0)
        support_row_layout.setSpacing(6)
        support_row_layout.addWidget(kalman_panel, 1)
        support_row_layout.addWidget(readiness_panel, 1)
        self._support_row_layout = support_row_layout
        right_column.addWidget(self._support_row, 0)
        self._right_column = right_column
        self._telemetry_panel = telemetry
        self._support_side_by_side = True
        # The Virtual Camera panel is the column's flexible item: whatever the
        # supporting panels below it do not need goes to the camera feed
        # (see _balance_camera_column). Their content is never reduced -- each
        # keeps the minimum height Qt measures for what it already displays.
        self._stacked_panels = (telemetry, self._support_row)
        views_row.addLayout(right_column, 3)

        ops.addLayout(views_row, 1)

        # Page 1: embedded 3D DIGITAL TWIN (the SAME live mission via TwinPane).
        self.twin_pane = TwinPane(show_fullscreen=False)
        self.twin_pane.bind_host(self)
        # Direct 3D manipulation: a twin drag moves the REAL simulator target
        # (image px), so radar/camera/twin render one synchronized mission.
        self.twin_pane.twin.dragDelta.connect(self._apply_twin_drag)
        self.twin_pane.modeChanged.connect(
            lambda mode: self.footer_status.setText(f"3D DIGITAL TWIN  •  VIEW: {mode}")
        )
        self.main_stack.addWidget(self.twin_pane)

        self._apply_view_emphasis(self._view_emphasis)

        # Old-style single dashboard: LEFT = the simulation-control column,
        # CENTER/RIGHT = radar + virtual camera (the main visual area).
        body.addWidget(self.control_scroll)
        body.addWidget(self.main_stack, 1)
        root.addLayout(body, 1)

        # Page 2: REPLAY & REPORTS workspace (uploaded-video launcher + reports).
        # Registered after construction so the reports page is main_stack page 2.
        reports_page = QFrame()
        reports_page.setObjectName("panel")
        rpt = QVBoxLayout(reports_page)
        rpt.setContentsMargins(16, 14, 16, 14)
        rpt.setSpacing(10)
        rpt_title = QLabel("ASTRALINK REPORT CENTER")
        rpt_title.setObjectName("panel_title")
        rpt.addWidget(rpt_title)
        rpt_caption = QLabel(
            "Upload a video for full tracked-video analysis (optional ground-truth CSV), "
            "then export or reopen PDF / JSON / CSV reports."
        )
        rpt_caption.setObjectName("caption")
        rpt_caption.setWordWrap(True)
        rpt.addWidget(rpt_caption)
        self.benchmark_button = QPushButton("🎞  BENCHMARK VIDEO")
        self.analyze_video_button = QPushButton("⬆  ANALYZE UPLOADED VIDEO")
        self.report_button = QPushButton("📊  EXPORT REPORT")
        self.instant_report_button = QPushButton("⏱  INSTANT REPORT")
        self.open_last_report_button = QPushButton("⧉  OPEN LAST REPORT")
        self.open_last_report_button.setObjectName("instant_report_button")
        self.open_last_report_button.setEnabled(False)
        self.open_last_report_button.setToolTip("Reopen the most recently generated report PDF")
        self.report_list = QListWidget()
        self.report_list.setObjectName("report_list")
        self.report_list.setMinimumHeight(180)
        self.report_list.itemDoubleClicked.connect(self._open_selected_report)
        # Wrapping report launchers: four across on a wide screen, stacked
        # when the page is narrow — never clipped, never scrolled sideways.
        reports_actions = FlowLayout(h_spacing=8, v_spacing=6)
        for button in (self.benchmark_button, self.analyze_video_button,
                       self.report_button, self.instant_report_button):
            reports_actions.addWidget(button)
        rpt.addLayout(reports_actions)
        reports_row2 = QHBoxLayout()
        reports_row2.addWidget(self.open_last_report_button)
        reports_row2.addStretch()
        rpt.addLayout(reports_row2)
        rpt.addWidget(QLabel("GENERATED REPORTS  (double-click to open the PDF)"))
        rpt.addWidget(self.report_list, 1)
        reports_buttons_row = QHBoxLayout()
        self.open_report_button = QPushButton("OPEN REPORT")
        self.open_folder_button = QPushButton("OPEN REPORT FOLDER")
        self.open_report_button.clicked.connect(self._open_selected_report)
        self.open_folder_button.clicked.connect(self._open_reports_folder)
        reports_buttons_row.addWidget(self.open_report_button)
        reports_buttons_row.addWidget(self.open_folder_button)
        reports_buttons_row.addStretch()
        rpt.addLayout(reports_buttons_row)
        self.benchmark_button.clicked.connect(self.benchmark_video)
        self.analyze_video_button.clicked.connect(self.analyze_uploaded_video)
        self.report_button.clicked.connect(self.export_report)
        self.instant_report_button.clicked.connect(self.generate_instant_report)
        self.open_last_report_button.clicked.connect(self._open_last_report)
        self.main_stack.addWidget(reports_page)

        # Footer
        footer = QFrame()
        footer.setObjectName("footer")
        self.footer = footer
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 5, 10, 5)
        self.footer_status = QLabel("Ready • Same tracking engine used for simulation and benchmark video")
        self.footer_status.setObjectName("footer_status")
        # Ignored + the trailing stretch keep footer text changes from
        # reflowing the footer row or raising its minimum (Test Lab shares
        # this footer).
        self.footer_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        fl.addWidget(self.footer_status)
        fl.addStretch()
        ps_targets_label = QLabel(
            "PS targets: ≤10 px error  •  <5% loss  •  ≤2 s acquisition  •  ≤1 s re-acquisition  •  ≥20 FPS"
        )
        ps_targets_label.setObjectName("caption")
        # Wrap rather than pin a 1100px minimum on the window footer.
        ps_targets_label.setWordWrap(True)
        fl.addWidget(ps_targets_label)
        root.addWidget(footer)

        self.setStyleSheet("""
            QMainWindow { background-color: #05080d; }
            QWidget { color: #e7edf3; font-family: "Segoe UI"; }
            #header, #panel, #footer {
                background-color: #0d151f;
                border: 1px solid #233545;
                border-radius: 10px;
            }
            #panel[emphasis="true"] {
                border: 2px solid #3f89b5;
            }
            #subpanel, #legend_panel, #kalman_panel {
                background-color: #091119;
                border: 1px solid #1b2b3b;
                border-radius: 8px;
            }
            #title { font-size: 22px; font-weight: 700; color: #f5f7fa; }
            #subtitle { font-size: 10px; color: #8293a5; }
            #system_status { font-size: 12px; font-weight: 700; color: #3bea91; }
            #tag_ai, #tag_kalman, #tag_rate {
                background-color: #101f2b;
                border: 1px solid #264052;
                border-radius: 10px;
                padding: 4px 7px;
                font-size: 8px;
                font-weight: 700;
            }
            #tag_ai { color: #5de89b; }
            #tag_kalman { color: #ffd34c; }
            #tag_rate { color: #61aefc; }
            #panel_title { font-size: 13px; font-weight: 700; color: #f0f4f8; }
            #section_title { font-size: 10px; font-weight: 700; color: #9eb0bf; }
            #small_label, #caption { font-size: 8px; color: #8294a5; }
            #radar_state, #camera_badge { font-size: 10px; font-weight: 700; color: #39e28a; }
            #camera_view { background-color: #020507; border: 1px solid #294050; border-radius: 7px; }
            #info_box {
                background-color: #081019;
                border: 1px solid #1b2a37;
                border-radius: 7px;
                padding: 7px;
                color: #aebac5;
                font-size: 8px;
            }
            #metric_card { background-color: #0b141d; border: 1px solid #1a2a39; border-radius: 6px; }
            #metric_card_primary { background-color: #0b141d; border: 1px solid #24466b; border-radius: 6px; }
            #metric_label { font-size: 9px; color: #8b9dad; }
            #spec_value { font-size: 9px; font-weight: 700; color: #dbe6ef; }
            #metric_value { font-size: 12px; font-weight: 700; color: #eef3f7; }
            #metric_value_primary { font-size: 14px; font-weight: 800; color: #f4f9fd; }
            #slider_value { font-size: 9px; font-weight: 700; color: #f06a6a; }
            #legend_item { font-size: 8px; color: #9baab8; }
            #kalman_state { font-size: 10px; font-weight: 700; color: #ffd34c; }
            #kalman_text { font-size: 8px; color: #9bacbb; }
            #footer_status { font-size: 8px; color: #65d69d; }
            QComboBox {
                background-color: #15232e;
                border: 1px solid #314756;
                border-radius: 6px;
                padding: 4px 6px;
                min-height: 22px;
                font-size: 8px;
            }
            QComboBox QAbstractItemView { background-color: #15232e; color: #e6edf3; }
            QCheckBox { font-size: 8px; color: #b6c2cc; spacing: 5px; }
            /* Column density: every control stays reachable without scrolling */
            #control_scroll QSlider { max-height: 20px; }
            #control_scroll QComboBox { min-height: 0px; max-height: 24px; }
            #control_scroll QPushButton { min-height: 0px; max-height: 28px; padding: 2px 6px; }
            QSlider::groove:horizontal { height: 4px; background: #304252; border-radius: 2px; }
            QSlider::handle:horizontal { width: 11px; margin: -4px 0; border-radius: 6px; background: #e86767; }
            QPushButton {
                min-height: 30px;
                background-color: #16232e;
                border: 1px solid #3a4e5e;
                border-radius: 6px;
                padding: 4px 6px;
                font-size: 9px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #22323e; }
            /* Twin CLOSE: compact secondary action, never a headline control */
            #close_button {
                min-height: 0px;
                padding: 2px 10px;
                font-size: 8px;
                color: #93a5b4;
                border-color: #2a3a48;
            }
            #start_button { background-color: #123622; border-color: #2eaa63; color: #67f0a0; }
            #pause_button { background-color: #3b2d11; border-color: #aa8030; color: #f4c967; }
            #reset_button { background-color: #3b171b; border-color: #a54850; color: #ff8a92; }
            #force_loss_button { background-color: #2f2110; border-color: #b17a2f; color: #f2c765; }
            #force_loss_button:hover { background-color: #493216; }
            #zoom_button, #zoom_reset_button { min-width: 32px; min-height: 24px; padding: 2px 6px; font-size: 11px; }
            #zoom_button { background-color: #10243a; border-color: #31587a; color: #72b8ff; }
            #zoom_reset_button { background-color: #1a2028; border-color: #40515f; color: #c7d1d9; }
            #zoom_value { font-size: 9px; font-weight: 700; color: #72b8ff; }
            #readiness_panel { background-color: #091119; border: 1px solid #2c4a3c; border-radius: 8px; }
            #link_ready { font-size: 13px; font-weight: 800; color: #43f08f; }
            #link_not_ready { font-size: 13px; font-weight: 800; color: #ff8a92; }
            #link_recover { font-size: 13px; font-weight: 800; color: #f4c967; }
            #camera_view { background-color: #010305; border: 1px solid #274257; border-radius: 7px; }
            QScrollArea { background: transparent; }
            #auth_overlay { background-color: rgba(2, 7, 12, 252); }
            #auth_panel { background-color: #0c1721; border: 1px solid #2b4558; border-radius: 14px; }
            #auth_kicker { font-size: 10px; font-weight: 700; color: #6aa8d9; letter-spacing: 2px; }
            #auth_title { font-size: 28px; font-weight: 800; color: #f4f7fa; }
            #auth_subtitle { font-size: 10px; color: #8195a8; }
            #auth_status { font-size: 9px; font-weight: 700; color: #6f879a; }
            #auth_input { background-color: #08111a; border: 1px solid #345163; border-radius: 7px; padding: 10px 12px; color: #eef4f8; font-size: 11px; }
            #auth_input:focus { border-color: #4d93c8; }
            #auth_button { min-height: 38px; background-color: #12334a; border: 1px solid #4b8fba; color: #e9f6ff; border-radius: 7px; font-size: 10px; font-weight: 800; }
            #auth_button:hover { background-color: #184561; }
            #auth_error { font-size: 9px; color: #ff7a7a; font-weight: 700; }
            #instant_report_button { background-color: #102b38; border-color: #41758d; color: #9dd8ff; }
            #view_switch {
                background-color: #0d151f;
                border: 1px solid #233545;
                border-radius: 10px;
            }
            #view_button, #twin_view_button {
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 10px;
                font-weight: 800;
                min-height: 30px;
            }
            #view_button { background-color: #101d29; border: 1px solid #2c4557; color: #b9c8d4; }
            #view_button:hover { background-color: #1a2c3c; }
            #view_button:checked { background-color: #14324a; border-color: #3f89b5; color: #9fdcff; }
            #twin_view_button {
                background-color: #0e2b1e;
                border: 1px solid #2eaa63;
                color: #67f0a0;
                font-size: 11px;
                padding: 6px 18px;
            }
            #twin_view_button:hover { background-color: #13402c; }
            #twin_view_button:checked { background-color: #155a36; border-color: #4ef09a; color: #c9ffe2; }
            #sidebar {
                background-color: #0d151f;
                border: 1px solid #233545;
                border-radius: 10px;
            }
            #section_button {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                text-align: left;
                padding: 9px 12px;
                font-size: 10px;
                font-weight: 800;
                color: #8ba0b0;
                min-height: 22px;
            }
            #section_button:hover { background-color: #14212e; color: #d5e2ec; }
            #section_button:checked {
                background-color: #14324a;
                border-color: #3f89b5;
                color: #bfe6ff;
            }
            #report_list {
                background-color: #081019;
                border: 1px solid #1b2a37;
                border-radius: 7px;
                font-size: 9px;
                color: #cfd9e2;
            }
            #report_list::item { padding: 4px 6px; }
            #report_list::item:selected { background-color: #14324a; color: #d8efff; }
        """)

        self._build_auth_overlay(central)

    def _apply_view_emphasis(self, index):
        """Draw the emphasis outline on the selected live view (visual only).

        Radar and camera are always both visible side by side; the top LIVE
        VIEW switch merely highlights the selected one. No state is reset.
        """
        for panel, selected in ((self.radar_panel, index == 0),
                                (self.camera_panel, index == 1)):
            panel.setProperty("emphasis", bool(selected))
            panel.style().unpolish(panel)
            panel.style().polish(panel)

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setObjectName("small_label")
        # Responsive caption: it WRAPS onto a second line instead of pinning
        # the control column's minimum width (the reason the column could not
        # shrink before). The widget keeps its natural size whenever there is
        # room, so the wide layout is visually unchanged.
        label.setWordWrap(True)
        label.setToolTip(text)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        label.setMinimumWidth(0)
        return label

    # ------------------------------------------------------------------
    # Section + live-view navigation
    # ------------------------------------------------------------------
    def _switch_section(self, index):
        """Single-dashboard navigation shim (kept for harness compatibility).

        The old-UI dashboard has no 3-section control stack anymore. Index 2
        (legacy REPLAY & REPORTS) still swaps the main area to the reports
        workspace; indexes 0/1 return to the live mission view.
        """
        if int(index) == 2:
            # REPLAY & REPORTS owns main_stack page 2.
            self.main_stack.setCurrentIndex(2)
            self.footer_status.setText("REPLAY & REPORTS  •  upload video analysis and report artifacts")
        else:
            if self.main_stack.currentIndex() == 2:
                self._switch_live_view(0)

    def _switch_live_view(self, index):
        """Switch the shared live visualization (radar / camera / twin).

        All three pages visualize the SAME simulation; switching never resets
        or duplicates mission state.
        """
        index = max(0, min(int(index), 2))
        if index in (0, 1):
            # RADAR and VIRTUAL CAMERA share the MISSION OPS page side by
            # side; the switch only moves the visual emphasis between them.
            self._view_emphasis = index
            self.main_stack.setCurrentIndex(0)
            self._apply_view_emphasis(index)
        else:
            # 3D DIGITAL TWIN: distinct primary view (main_stack page 1).
            self.main_stack.setCurrentIndex(1)
        if index == 2:
            self.footer_status.setText(
                "3D DIGITAL TWIN  •  same live mission  •  FOCUS buttons for terminal close-ups"
            )
        elif index == 0:
            self.footer_status.setText("LIVE VIEW: RADAR  •  trajectory, prediction and boresight")
        else:
            self.footer_status.setText("LIVE VIEW: VIRTUAL CAMERA  •  640 × 480 optical feed")

    @staticmethod
    def _make_slider(minimum, maximum, initial, scale, suffix):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(initial)
        value = QLabel()
        value.setObjectName("slider_value")
        if scale == 100:
            number = initial / 100.0
            value.setText(f"{number:.2f} {suffix}" if suffix != "gain" else f"{number:.2f}")
        elif scale == 10:
            value.setText(f"{initial / 10.0:.1f} {suffix}")
        elif suffix == "m×100":
            value.setText(f"{initial * 100:.0f} m")
        else:
            value.setText(f"{initial / scale:.0f} {suffix}")
        return slider, value

    @staticmethod
    def _slider_card(label_text, slider, value_label):
        card = QFrame()
        card.setObjectName("metric_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(1)
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setObjectName("small_label")
        lbl.setToolTip(label_text)
        # The caption must never widen the column: the fixed VALUE keeps its
        # own width, the name yields first (same rule as the telemetry cards).
        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lbl.setMinimumWidth(0)
        row.addWidget(lbl, 1)
        row.addWidget(value_label, 0)
        layout.addLayout(row)
        layout.addWidget(slider)
        return card

    # ------------------------------------------------------------------
    # Runtime settings
    # ------------------------------------------------------------------
    def _refresh_specification_panel(self):
        """Mirror the live simulation settings onto the read-only spec card.

        Every value is read from the same config/settings the mission uses, so
        this card can never disagree with the simulation.
        """
        self.spec_screen_value.setText(
            f"{config.SCREEN_WIDTH} \u00d7 {config.SCREEN_HEIGHT} px"
        )
        self.spec_resolution_value.setText(
            f"{config.FRAME_WIDTH} \u00d7 {config.FRAME_HEIGHT} px"
        )
        self.spec_type_value.setText(config.CAMERA_TYPE)
        self.spec_interval_value.setText(
            f"{1000.0 / max(config.UPDATE_RATE, 1):.0f} ms  ({config.UPDATE_RATE} Hz)"
        )
        self.spec_range_value.setText(self._camera_update_range_text())

    def _camera_update_range_text(self):
        """The live pointing range the controller is allowed to update over."""
        return (
            f"\u00b1{self.camera_range_deg:.1f}\u00b0 pan  \u00b7  "
            f"\u00b1{self.camera_range_deg * 0.75:.1f}\u00b0 tilt"
        )

    def _apply_runtime_settings(self):
        self.simulator.set_beacon_speed(self.beacon_speed)
        self.simulator.set_beacon_range(self.beacon_range_m)
        self.simulator.set_beacon_size(self.beacon_size_px)
        self.simulator.set_fov(self.camera_fov_deg)
        config.MAX_PAN_SPEED = self.camera_pan_speed
        config.MAX_TILT_SPEED = self.camera_tilt_speed
        self.controller.set_angle_limits(self.camera_range_deg, self.camera_range_deg * 0.75)
        self.controller.set_gain(2.0)
        self.simulator.noise_level = self.noise_level
        self._apply_disturbances()

        self.speed_value.setText(f"{self.beacon_speed:.1f} px/frame")
        self.range_value.setText(f"{self.beacon_range_m:.0f} m")
        self.fov_value.setText(f"{self.camera_fov_deg:.1f}°")
        self.cam_range_value.setText(f"±{self.camera_range_deg:.1f}°")
        self.pan_speed_value.setText(f"{self.camera_pan_speed:.1f}°/s")
        self.tilt_speed_value.setText(f"{self.camera_tilt_speed:.1f}°/s")
        self.pred_value.setText(f"{self.prediction_gain:.2f}")
        self.size_value.setText(f"{self.beacon_size_px:.0f} px")
        self.noise_level_value.setText(str(self.noise_level))
        self.turbulence_value.setText(disturbance_level_text(self.turbulence_slider.value()))
        self.jitter_value.setText(disturbance_level_text(self.jitter_slider.value()))
        self.platform_value.setText(disturbance_level_text(self.platform_slider.value()))
        self.atmosphere_value.setText(f"{self.atmosphere_level}%")
        self._refresh_specification_panel()
        self.camera_info.setText(
            f"{config.FRAME_WIDTH} × {config.FRAME_HEIGHT}  |  FOV {config.FOV_HORIZONTAL:.1f}° × {config.FOV_VERTICAL:.1f}°  |  {config.UPDATE_RATE} Hz"
        )

    def _apply_disturbance_preset(self, preset):
        """Apply a deterministic simulator disturbance profile to the live
        mission and mirror its values back onto this dashboard's controls."""
        self.simulator.set_disturbance_preset(preset)
        sim = self.simulator
        # Mirror the applied profile onto the controls with signals BLOCKED:
        # unblocked setters would fire _apply_disturbances mid-mirror and
        # re-apply stale sibling values over the fresh preset.
        mirrors = [
            (self.noise_selector, "setCurrentText", sim.noise_type),
            (self.noise_level_slider, "setValue", int(sim.noise_level)),
            (self.turbulence_check, "setChecked", bool(sim.turbulence_enabled)),
            (self.turbulence_slider, "setValue", disturbance_px_to_level(sim.turbulence_strength_px)),
            (self.jitter_check, "setChecked", bool(sim.camera_jitter_enabled)),
            (self.jitter_slider, "setValue", disturbance_px_to_level(sim.camera_jitter_px)),
            (self.platform_check, "setChecked", bool(sim.platform_motion_enabled)),
            (self.platform_slider, "setValue", disturbance_px_to_level(sim.platform_motion_px)),
            (self.platform_pattern_selector, "setCurrentText", sim.platform_motion_pattern),
            (self.atmosphere_selector, "setCurrentText", sim.atmosphere),
            (self.atmosphere_slider, "setValue", int(sim.atmosphere_level)),
        ]
        for widget, method, value in mirrors:
            widget.blockSignals(True)
            try:
                getattr(widget, method)(value)
            finally:
                widget.blockSignals(False)
        # The preset has already written its EXACT px strengths into the
        # simulator. Re-applying from the quantized 5-level widgets would
        # overwrite them, so the dashboard state is mirrored FROM the
        # simulator instead — the disturbance mathematics stays untouched.
        self.turbulence_strength = float(sim.turbulence_strength_px)
        self.camera_jitter_strength = float(sim.camera_jitter_px)
        self.platform_motion_strength = float(sim.platform_motion_px)
        self.atmosphere_level = int(sim.atmosphere_level)
        self.turbulence_value.setText(disturbance_level_text(self.turbulence_slider.value()))
        self.jitter_value.setText(disturbance_level_text(self.jitter_slider.value()))
        self.platform_value.setText(disturbance_level_text(self.platform_slider.value()))
        self.atmosphere_value.setText(f"{self.atmosphere_level}%")
        self.noise_level_value.setText(str(self.noise_level_slider.value()))
        self.noise_level = int(self.noise_level_slider.value())
        self.footer_status.setText(f"DISTURBANCE PRESET: {preset} applied to the live mission")

    def _apply_disturbances(self, _value=None):
        # Slider value = intensity level (1..5); the simulator still receives
        # the px strength it has always used.
        self.turbulence_strength = disturbance_level_to_px(self.turbulence_slider.value())
        self.camera_jitter_strength = disturbance_level_to_px(self.jitter_slider.value())
        self.platform_motion_strength = disturbance_level_to_px(self.platform_slider.value())
        self.atmosphere_level = int(self.atmosphere_slider.value())
        self.turbulence_value.setText(disturbance_level_text(self.turbulence_slider.value()))
        self.jitter_value.setText(disturbance_level_text(self.jitter_slider.value()))
        self.platform_value.setText(disturbance_level_text(self.platform_slider.value()))
        self.atmosphere_value.setText(f"{self.atmosphere_level}%")
        self.simulator.set_disturbances(
            turbulence_enabled=self.turbulence_check.isChecked(),
            turbulence_strength_px=self.turbulence_strength,
            camera_jitter_enabled=self.jitter_check.isChecked(),
            camera_jitter_px=self.camera_jitter_strength,
            platform_motion_enabled=self.platform_check.isChecked(),
            platform_motion_px=self.platform_motion_strength,
            platform_motion_pattern=self.platform_pattern_selector.currentText(),
            atmosphere=self.atmosphere_selector.currentText(),
            atmosphere_level=self.atmosphere_level,
        )

    def _on_speed_changed(self, value):
        self.beacon_speed = value / 10.0
        self.speed_value.setText(f"{self.beacon_speed:.1f} px/frame")
        self.simulator.set_beacon_speed(self.beacon_speed)

    def _on_range_changed(self, value):
        self.beacon_range_m = float(value * 100)
        self.range_value.setText(f"{self.beacon_range_m:.0f} m")
        self.simulator.set_beacon_range(self.beacon_range_m)
        # Keep every twin surface's SEPARATION control on the real value.
        for pane in self._twin_panes():
            pane._sync_separation_display(self.beacon_range_m)

    def _twin_panes(self):
        """Every live twin surface (embedded pane + standalone window)."""
        panes = []
        if getattr(self, "twin_pane", None) is not None:
            panes.append(self.twin_pane)
        standalone = getattr(self, "digital_twin_window", None)
        if standalone is not None and getattr(standalone, "pane", None) is not None:
            panes.append(standalone.pane)
        return panes

    def _on_fov_changed(self, value):
        self.camera_fov_deg = value / 10.0
        self.fov_value.setText(f"{self.camera_fov_deg:.1f}°")
        self.simulator.set_fov(self.camera_fov_deg)
        self.radar.max_angle = max(3.0, self.camera_fov_deg * 0.9)
        self.camera_info.setText(
            f"{config.FRAME_WIDTH} × {config.FRAME_HEIGHT}  |  FOV {config.FOV_HORIZONTAL:.1f}° × {config.FOV_VERTICAL:.1f}°  |  {config.UPDATE_RATE} Hz"
        )

    def _on_camera_range_changed(self, value):
        self.camera_range_deg = value / 10.0
        self.cam_range_value.setText(f"±{self.camera_range_deg:.1f}°")
        self.controller.set_angle_limits(self.camera_range_deg, self.camera_range_deg * 0.75)
        if getattr(self, "spec_range_value", None) is not None:
            self.spec_range_value.setText(self._camera_update_range_text())

    def _set_pan_tilt_speeds(self, pan_speed, tilt_speed):
        """Apply the real Max Pan / Max Tilt speed limits to the controller."""
        self.camera_pan_speed = max(0.1, float(pan_speed))
        self.camera_tilt_speed = max(0.1, float(tilt_speed))
        self.camera_slew_rate = self.camera_pan_speed
        config.MAX_PAN_SPEED = self.camera_pan_speed
        config.MAX_TILT_SPEED = self.camera_tilt_speed
        self.pan_speed_value.setText(f"{self.camera_pan_speed:.1f}°/s")
        self.tilt_speed_value.setText(f"{self.camera_tilt_speed:.1f}°/s")

    def _on_pan_speed_changed(self, value):
        self._set_pan_tilt_speeds(value / 10.0, self.camera_tilt_speed)

    def _on_tilt_speed_changed(self, value):
        self._set_pan_tilt_speeds(self.camera_pan_speed, value / 10.0)

    def _on_slew_changed(self, value):
        # Legacy single-control entry point: writes BOTH axes.
        self._set_pan_tilt_speeds(value / 10.0, value / 10.0)

    def _on_prediction_changed(self, value):
        self.prediction_gain = value / 100.0
        self.pred_value.setText(f"{self.prediction_gain:.2f}")

    def _on_size_changed(self, value):
        self.beacon_size_px = float(value)
        self.size_value.setText(f"{value} px")
        self.simulator.set_beacon_size(self.beacon_size_px)

    def _on_noise_level_changed(self, value):
        self.noise_level = int(value)
        self.noise_level_value.setText(str(value))
        self.simulator.noise_level = self.noise_level
        self._reset_metrics()

    # ------------------------------------------------------------------
    # Metrics / lifecycle
    # ------------------------------------------------------------------
    def _reset_metrics(self):
        self.previous_time = time.perf_counter()
        self.run_start_time = self.previous_time
        self.processing_times.clear()
        self.frames_processed = 0
        self.visible_frames = 0
        self.tracking_frames = 0
        self.detection_misses = 0
        self.evaluation_visible_frames = 0
        self.evaluation_detection_misses = 0
        self.error_sum = 0.0
        self.error_count = 0
        self.average_error = 0.0
        self.maximum_error = 0.0
        self.error_squared_sum = 0.0
        self.centroid_error_sum = 0.0
        self.centroid_error_count = 0
        self.average_centroid_error = 0.0
        self.maximum_centroid_error = 0.0
        self.error_history.clear()
        self.centroid_error_history.clear()
        self.pan_history.clear()
        self.tilt_history.clear()
        self.metrics_started = False
        self.lock_candidate_frames = 0
        self.acquisition_time_s = None
        self.loss_events = 0
        self.successful_reacquisitions = 0
        self.in_loss = False
        self.loss_start_time = None
        self.reacquisition_times.clear()
        # Reset the forced-loss test bookkeeping too.
        self.forced_loss_active = False
        self.forced_loss_start_frame = None
        self.forced_loss_frame_end = None
        self.forced_loss_reacq_s = None
        self.last_detection = None
        self.camera_target_history.clear()
        self.camera_prediction_history.clear()
        self.link_ready_streak = 0
        self._reset_handoff_window_state()
        self.fps = 0.0
        self.instant_report_history.clear()
        self._last_report_snapshot = None

        self.link_readiness_label.setText("● NOT READY")
        self.link_readiness_label.setObjectName("link_not_ready")
        self.link_readiness_label.style().unpolish(self.link_readiness_label)
        self.link_readiness_label.style().polish(self.link_readiness_label)
        self.link_conditions_label.setText("Tracking: —  |  Error: —  |  FPS: —  |  AI: —  |  Stable: 0/10")

        self.t_fps.setText("0.0")
        self.t_error.setText("0.00 px")
        self.t_centroid.setText("—")
        self.t_avg.setText("0.00 px")
        self.t_max.setText("0.00 px")
        self.t_rmse.setText("0.00 px")
        self.t_acq.setText("—")
        self.t_reacq.setText("—")
        self.t_loss.setText("0.0 %")
        self.t_lock.setText("—")
        self.t_state.setText("SEARCHING")
        self.t_ai.setText("—")
        self.error_graph.set_series([])
        self.motion_graph.set_series([])
        self.kalman_state_label.setText("Prediction: WAITING")

    def change_motion_pattern(self, pattern):
        self.simulator.set_motion_pattern(pattern)
        # Programmatic resets (harness, presets) must keep the selector in
        # step with the authoritative simulator pattern.
        if self.motion_selector.currentText() != pattern:
            self.motion_selector.blockSignals(True)
            self.motion_selector.setCurrentText(pattern)
            self.motion_selector.blockSignals(False)
        self.controller = CameraController()
        self._apply_runtime_settings()
        self.tracker.reset()
        self.radar.history.clear()
        self._reset_metrics()

    def change_noise_type(self, noise_type):
        self.simulator.set_noise_type(noise_type)
        self._reset_metrics()

    def start_simulation(self):
        if not self._authenticated:
            self.footer_status.setText(
                "Not authenticated • enter the access key to start the simulation"
            )
            return
        self.simulation_running = True
        # Reports must measure MISSION duration, not wall-clock since launch:
        # re-anchoring here keeps idle pre-START time out of Processing FPS.
        self.run_start_time = time.perf_counter()
        self.timer.start(33)
        self.system_status.setText("● SYSTEM ACTIVE")
        self.footer_status.setText("Simulation running • disturbances are applied before beacon detection")

    def pause_simulation(self):
        self.simulation_running = False
        self.timer.stop()
        self.system_status.setText("● SYSTEM PAUSED")
        if self.frames_processed > 0:
            try:
                paths = self._save_current_report(prefix="simulation_pause")
                self.footer_status.setText(f"Paused • report saved: {paths[0].name}")
            except OSError as exc:
                self.footer_status.setText(f"Paused • report save failed: {exc}")

    def _refresh_designated_target_selector(self):
        ids = self.simulator.get_target_ids()
        self.designated_target_selector.blockSignals(True)
        self.designated_target_selector.clear()
        self.designated_target_selector.addItems(ids)
        if self.designated_target_id in ids:
            self.designated_target_selector.setCurrentText(self.designated_target_id)
        elif ids:
            self.designated_target_id = ids[0]
            self.simulator.set_designated_target(ids[0])
            self.designated_target_selector.setCurrentText(ids[0])
        self.designated_target_selector.blockSignals(False)

    def _on_target_count_changed(self, text):
        try:
            count = int(text)
        except ValueError:
            count = 1
        self.target_count = max(1, min(5, count))
        self.simulator.set_target_count(self.target_count)
        self._refresh_designated_target_selector()
        self.controller = CameraController()
        self.tracker.reset()
        self.radar.history.clear()
        self._reset_metrics()
        self.footer_status.setText(f"Multi-target scenario • {self.target_count} moving targets")

    def _on_designated_target_changed(self, target_id):
        target_id = str(target_id)
        if target_id not in self.simulator.get_target_ids():
            return
        self.designated_target_id = target_id
        self.simulator.set_designated_target(target_id)
        self.controller = CameraController()
        self.tracker.reset()
        self.radar.history.clear()
        self._reset_metrics()
        self.footer_status.setText(f"Designated target changed to {target_id} • tracking reset")

    def reset_simulation(self):
        self.timer.stop()
        motion = self.motion_selector.currentText()
        noise = self.noise_selector.currentText()
        pattern = self.platform_pattern_selector.currentText()
        atmosphere = self.atmosphere_selector.currentText()
        self.simulator = Simulator()
        self.simulator.set_target_count(self.target_count)
        self.simulator.set_designated_target(self.designated_target_id)
        self.simulator.set_motion_pattern(motion)
        self.simulator.set_noise_type(noise)
        self.platform_pattern_selector.setCurrentText(pattern)
        self.atmosphere_selector.setCurrentText(atmosphere)
        self.controller = CameraController()
        self.tracker.reset()
        self._refresh_designated_target_selector()
        self._apply_runtime_settings()
        self.radar.history.clear()
        self._reset_metrics()
        # A new run starts the performance graphs from empty: no curve from the
        # previous run survives into the next one (REPLAY does the same for the
        # video benchmark). A mid-run parameter tweak does NOT clear them, so
        # the graph still shows how those changed conditions affected tracking.
        panel = getattr(self, "graph_panel", None)
        if panel is not None:
            try:
                panel.reset("Run benchmark to generate performance data.")
            except Exception:
                pass  # a display reset must never stop the simulation
        self.simulation_running = True
        if self._authenticated:
            self.timer.start(33)
        self.system_status.setText("● SYSTEM ACTIVE")
        self.footer_status.setText("Reset • ready for a new scenario")

    # ------------------------------------------------------------------
    # Refinement 6: live-synchronized perspective-projected digital twin
    # ------------------------------------------------------------------
    def open_digital_twin(self):
        """Open the standalone twin window (also used by the harness). The
        in-app ★ 3D DIGITAL TWIN view switch shows the same synchronized
        TwinPane embedded in the main window."""
        if not self._authenticated:
            self.footer_status.setText(
                "Not authenticated • enter the access key to open the Digital Twin"
            )
            return
        if self.digital_twin_window is None:
            self.digital_twin_window = DigitalTwinWindow(self)
            self.digital_twin_window.destroyed.connect(self._clear_digital_twin_window)
            # Direct 3D manipulation for the standalone pane too (connected
            # ONCE at creation — never per tick).
            self.digital_twin_window.pane.twin.dragDelta.connect(self._apply_twin_drag)
        self.digital_twin_window.show()
        self.digital_twin_window.raise_()
        self.digital_twin_window.activateWindow()

    # ------------------------------------------------------------------
    # Digital-twin mission strip (ONE source of truth with the main GUI)
    # ------------------------------------------------------------------
    def _apply_twin_mission(self, section, key, value):
        """Apply a mission-strip edit to the REAL main-GUI control.

        The strip edit lands on the main GUI widget for that parameter, whose
        normal signal handlers push it into the simulator — so there is no
        second simulation and Mission Control, Test Lab, the twin strip and
        the standalone twin window can never disagree.
        """
        if value is None:
            return
        if key == "count" and str(value) != self.target_count_selector.currentText():
            self.target_count_selector.setCurrentText(str(value))
        elif key == "designated" and str(value) != self.designated_target_selector.currentText():
            self.designated_target_selector.setCurrentText(str(value))
        elif key == "motion" and str(value) != self.motion_selector.currentText():
            self.motion_selector.setCurrentText(str(value))
        elif key == "noise" and str(value) != self.noise_selector.currentText():
            self.noise_selector.setCurrentText(str(value))
        elif key == "atmosphere" and str(value) != self.atmosphere_selector.currentText():
            self.atmosphere_selector.setCurrentText(str(value))
        elif key == "turbulence":
            if self.turbulence_check.isChecked() != bool(value):
                self.turbulence_check.setChecked(bool(value))
        elif key == "jitter":
            if self.jitter_check.isChecked() != bool(value):
                self.jitter_check.setChecked(bool(value))
        elif key == "platform":
            if self.platform_check.isChecked() != bool(value):
                self.platform_check.setChecked(bool(value))

    def _apply_twin_drag(self, target_id, dx_px, dy_px):
        """Twin drag -> REAL simulator position (Stage 3 direct manipulation).

        The simulator IS the single source of truth; radar, virtual camera
        and both twin surfaces render the moved object on the next frame.
        """
        try:
            self.simulator.move_target(str(target_id), float(dx_px), float(dy_px))
        except AttributeError:
            pass

    def _twin_strip_state(self):
        """Live mission parameters for the twin strip (main-GUI values)."""
        return {
            "count": self.target_count_selector.currentText(),
            "designated": self.designated_target_id,
            "designated_options": self.simulator.get_target_ids(),
            # Single source of truth: the LIVE simulator motion, never the
            # selector widget text (which can lag the actual simulation).
            "motion": self.simulator.motion_pattern,
            "noise": self.noise_selector.currentText(),
            "atmosphere": self.atmosphere_selector.currentText(),
            "turbulence": self.turbulence_check.isChecked(),
            "jitter": self.jitter_check.isChecked(),
            "platform": self.platform_check.isChecked(),
        }

    def _twin_strip_values(self, center="—", error="—"):
        """ONE authoritative strip payload for every twin surface.

        Every value comes from the live mission objects (simulator, tracker,
        controller) — never from selector widgets and never hard-coded.
        """
        designated = next(
            (t for t in self.simulator.targets if t["id"] == self.designated_target_id),
            None,
        )
        controller = getattr(self, "controller", None)
        return {
            "designated": self.designated_target_id,
            "motion": self.simulator.motion_pattern,
            "center": center,
            "error": error,
            "state": self.tracker.state if hasattr(self, "tracker") else "SEARCHING",
            "handoff": "READY" if self.link_ready_streak >= self.link_required_frames else "STANDBY",
            "pan": f"{controller.pan:+.2f}\u00b0" if controller is not None else "N/A",
            "tilt": f"{controller.tilt:+.2f}\u00b0" if controller is not None else "N/A",
            "separation": f"{self.beacon_range_m:.0f} m" if getattr(self, "beacon_range_m", None) else "N/A",
        }

    def _sync_mission_strip(self, force=False, center="—", error="—"):
        """Refresh the embedded twin strip's read-only status values."""
        pane = getattr(self, "twin_pane", None)
        if pane is None:
            return
        if force:
            pane._refresh_strip_controls()
        # Hero strip shows only the essentials: TARGET / MOTION / POINTING /
        # LINK. Disturbance and count mirrors live on the main dashboard.
        pane.update_strip(**self._twin_strip_values(center=center, error=error))

    def _push_strip_to_twin_surfaces(self, center="—", error="—"):
        """Feed the SAME strip values to every other twin surface (standalone
        window). One mission state, one set of strip numbers, no divergence."""
        standalone = getattr(self, "digital_twin_window", None)
        if standalone is None:
            return
        pane = getattr(standalone, "pane", None)
        if pane is None:
            return
        pane.update_strip(**self._twin_strip_values(center=center, error=error))

    def _open_selected_report(self, *_args):
        selection = self.report_list.currentItem()
        if selection is None:
            self.footer_status.setText("Select a report in the list first")
            return
        if self._open_path(selection.text()):
            self.footer_status.setText(f"Opened report: {Path(selection.text()).name}")

    def _open_reports_folder(self):
        folder = Path("reports").resolve()
        folder.mkdir(exist_ok=True)
        if self._open_path(str(folder)):
            self.footer_status.setText("Opened report folder")

    def _register_report_artifact(self, path):
        """List a freshly generated PDF report in the Replay & Reports page."""
        if path is None:
            return
        self.report_list.addItem(str(path))
        self.report_list.setCurrentRow(self.report_list.count() - 1)

    def _clear_digital_twin_window(self, *_args):
        self.digital_twin_window = None

    def _update_digital_twin(self, state, predicted_world=None, target_states=None,
                             center_error_px=None, offset_px=(0.0, 0.0),
                             link_ready=False):
        # The embedded twin page shows the SAME synchronized state as the
        # standalone window — feed both from the identical snapshot.
        # Page 0 = mission ops (radar + camera), page 1 = 3D DIGITAL TWIN.
        # link_ready is passed in directly from the computed boolean so the
        # twin's beam-impact state is correct even during LOST/SEARCHING.
        if self.main_stack.currentIndex() == 1 and self.twin_pane is not None:
            self._push_twin_state(self.twin_pane.set_state, state, predicted_world,
                                  target_states, link_ready=link_ready)
            self.twin_pane.set_beam_impact_data(center_error_px, offset_px, link_ready)
        if self.digital_twin_window is None:
            return

        # IMPORTANT: the twin receives the simulator's raw target positions,
        # not camera-relative angles. This gives the 3D view an independent
        # inertial/local-space viewpoint while preserving exact synchronization
        # with the same live target motion used by radar and virtual camera.
        self._push_twin_state(self.digital_twin_window.set_state, state, predicted_world,
                              target_states, link_ready=link_ready)
        self.digital_twin_window.set_beam_impact_data(center_error_px, offset_px, link_ready)

    def _push_twin_state(self, sink, state, predicted_world, target_states, link_ready=False):
        """Push one identical synchronized snapshot into a twin surface
        (embedded pane or standalone window) — one mission, many views."""
        try:
            twin_state = self.simulator.get_digital_twin_state()
        except AttributeError:
            # Backward-compatible fallback for an older simulator copy.
            twin_state = {
                "targets": [dict(item) for item in getattr(self.simulator, "targets", [])],
                "camera_position": (0.0, 0.0, 0.0),
                "platform_offset_px": getattr(self.simulator, "last_platform_offset", (0.0, 0.0)),
                "jitter_px": getattr(self.simulator, "last_camera_jitter", (0.0, 0.0)),
                "beam_wander_px": (0.0, 0.0),
                "motion_pattern": self.motion_selector.currentText(),
            }

        sink(
            camera_angles=(self.simulator.camera_pan, self.simulator.camera_tilt),
            target_angles=self.simulator.get_target_angles(),
            predicted_angles=predicted_world,
            range_m=self.beacon_range_m,
            targets=twin_state.get("targets", target_states or []),
            world_targets=twin_state.get("targets", target_states or []),
            camera_position=twin_state.get("camera_position", (0.0, 0.0, 0.0)),
            designated_id=self.designated_target_id,
            state=state,
            frame_count=self.simulator.frame_count,
            sim_time=self.frames_processed / max(config.UPDATE_RATE, 1),
            fps=self.fps,
            motion_pattern=twin_state.get("motion_pattern", self.motion_selector.currentText()),
            turbulence_enabled=bool(self.turbulence_check.isChecked()),
            turbulence_strength=self.turbulence_strength,
            jitter_enabled=bool(self.jitter_check.isChecked()),
            jitter_strength=self.camera_jitter_strength,
            platform_enabled=bool(self.platform_check.isChecked()),
            platform_strength=self.platform_motion_strength,
            noise_name=self.noise_selector.currentText(),
            atmosphere_name=self.atmosphere_selector.currentText(),
            atmosphere_level=self.atmosphere_level,
            platform_offset_px=twin_state.get("platform_offset_px", (0.0, 0.0)),
            jitter_px=twin_state.get("jitter_px", (0.0, 0.0)),
            beam_wander_px=twin_state.get("beam_wander_px", (0.0, 0.0)),
            link_ready=link_ready,
        )

    # ------------------------------------------------------------------
    # Manual re-acquisition test control
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # HANDOFF WINDOW — display-only analyzer over the existing readiness gate
    # ------------------------------------------------------------------
    def _reset_handoff_window_state(self):
        """Reset the handoff-window bookkeeping (kept with the other run
        metrics so RESET starts a fresh window)."""
        self._handoff_qualified_start_s = None   # mission time of the first ready frame
        self._handoff_stable_frames = 0          # consecutive ready frames in the open window
        self._handoff_open = False               # window currently open?
        self._handoff_ready = False              # streak reached the required length?
        self._handoff_last = None                # latest readout dict for the dialog

    def _update_handoff_window(self, ready_frame, center_error, signature_score, ai_conf):
        """Track the continuous span in which every handoff condition holds.

        The verdict comes from the SAME ready_frame computation used by the
        LINK READINESS panel — this method only records it. When any condition
        fails, the active window closes; the next qualifying frame opens a new
        one (the previous window's totals stay visible in the dialog).
        """
        now_s = self.frames_processed / max(config.UPDATE_RATE, 1)
        if ready_frame:
            if self._handoff_qualified_start_s is None:
                self._handoff_qualified_start_s = now_s
            self._handoff_stable_frames += 1
            self._handoff_open = True
            if self._handoff_stable_frames >= self.link_required_frames:
                self._handoff_ready = True
        else:
            # Conditions broke: the active window ends. A subsequent run of
            # ready frames opens a fresh window from that moment.
            self._handoff_qualified_start_s = None
            self._handoff_stable_frames = 0
            self._handoff_open = False
            self._handoff_ready = False

        duration_s = (
            (now_s - self._handoff_qualified_start_s)
            if self._handoff_open and self._handoff_qualified_start_s is not None
            else 0.0
        )
        self._handoff_last = {
            "open": self._handoff_open,
            "ready": self._handoff_ready,
            "first_qualified_s": self._handoff_qualified_start_s,
            "stable_duration_s": duration_s,
            "stable_frames": self._handoff_stable_frames,
            "center_error_px": center_error,
            "max_tracking_error_px": float(self.maximum_error) if self.error_count else None,
            "loss_percent": self._target_loss_rate(),
            "reacquisition_s": (
                self.reacquisition_times[-1] if self.reacquisition_times else None
            ),
            "stable_frames_required": self.link_required_frames,
            "signature_score": signature_score,
            "ai_confidence": ai_conf,
        }
        if self._handoff_dialog is not None:
            try:
                self._handoff_dialog.update_values(self._handoff_last)
            except RuntimeError:
                # Dialog was closed and deleted; drop the reference.
                self._handoff_dialog = None

    def _open_handoff_window(self):
        """Open (or focus) the live HANDOFF WINDOW readout dialog."""
        if self._handoff_dialog is not None:
            try:
                self._handoff_dialog.raise_()
                self._handoff_dialog.activateWindow()
                return
            except RuntimeError:
                self._handoff_dialog = None
        self._handoff_dialog = HandoffWindowDialog(self)
        if self._handoff_last is not None:
            self._handoff_dialog.update_values(self._handoff_last)
        else:
            self._handoff_dialog.update_values({
                "open": False,
                "ready": False,
                "first_qualified_s": None,
                "stable_duration_s": 0.0,
                "stable_frames": 0,
                "center_error_px": None,
                "max_tracking_error_px": None,
                "loss_percent": 0.0,
                "reacquisition_s": None,
                "stable_frames_required": self.link_required_frames,
                "signature_score": 0.0,
                "ai_confidence": 0.0,
            })
        self._handoff_dialog.show()

    # ------------------------------------------------------------------
    # WHY LOCKED? — explainability panel built from the REAL gate conditions
    # ------------------------------------------------------------------
    def _build_why_locked_evidence(self, ready, state, center_error,
                                   detector_conf, ai_conf, signature_score,
                                   beacon_detected):
        """Itemize every handoff condition with its real value.

        Each row is (label, value_text, passed) where ``passed`` is True/False
        for checked conditions and None when not yet measurable (rendered as
        a neutral dash — never a fake check). Values come from the live run.
        """
        loss = self._target_loss_rate()
        reacq = self.reacquisition_times[-1] if self.reacquisition_times else None
        beacon_ok = (
            beacon_detected
            and detector_conf >= 0.55
            and signature_score >= 0.75
            and ai_conf >= config.LINK_READY_AI_THRESHOLD
        )
        conditions = [
            ("BEACON VERIFIED",
             f"detector {detector_conf:.2f} • signature {signature_score:.2f} • AI {ai_conf:.2f}",
             beacon_ok),
            ("TRACKING STABLE", str(state), state == "TRACKING" and self.metrics_started),
            ("CENTER ERROR",
             f"{center_error:.2f} px  (limit 10 px)" if center_error is not None else "—  (no lock)",
             center_error is not None and center_error <= 10.0),
            ("MAX TRACKING ERROR",
             f"{self.maximum_error:.2f} px" if self.error_count else "—",
             self.error_count > 0 and self.maximum_error <= 10.0),
            ("TARGET LOSS", f"{loss:.2f} %  (limit 5 %)", loss < 5.0),
            ("ACQUISITION",
             f"{self.acquisition_time_s:.2f} s  (limit 2 s)" if self.acquisition_time_s is not None else "—",
             self.acquisition_time_s is not None and self.acquisition_time_s <= 2.0),
            ("RE-ACQUISITION",
             f"{reacq:.2f} s  (limit 1 s)" if reacq is not None else "—  (none yet)",
             None if reacq is None else reacq <= 1.0),
            ("PROCESSING FPS", f"{self.fps:.1f}  (min 20)", self.fps >= config.PS_PROCESSING_MIN_FPS),
            ("STABLE FRAMES",
             f"{self.link_ready_streak}/{self.link_required_frames}",
             self.link_ready_streak >= self.link_required_frames),
        ]
        return {"ready": bool(ready), "state": str(state), "conditions": conditions}

    def _open_why_locked(self):
        """Open (or focus) the WHY LOCKED? explanation dialog."""
        if self._why_locked_dialog is not None:
            try:
                self._why_locked_dialog.raise_()
                self._why_locked_dialog.activateWindow()
                if self._why_locked_evidence is not None:
                    self._why_locked_dialog.update_evidence(self._why_locked_evidence)
                return
            except RuntimeError:
                self._why_locked_dialog = None
        self._why_locked_dialog = WhyLockedDialog(self)
        if self._why_locked_evidence is None:
            self._why_locked_evidence = self._build_why_locked_evidence(
                False, self.tracker.state, None, 0.0, 0.0, 0.0, False
            )
        self._why_locked_dialog.update_evidence(self._why_locked_evidence)
        self._why_locked_dialog.show()

    def force_target_loss(self):
        """Hide the designated beacon briefly so re-acquisition can be demonstrated.

        The intentional test is armed here; when the resulting detection gap
        appears, the loss event is consumed by the armed test and the measured
        re-acquisition time is kept for display and reporting.
        """
        if not self._authenticated:
            return
        if not self.simulation_running:
            self.footer_status.setText("Target-loss test unavailable while simulation is paused")
            return
        if self.tracker.state != "TRACKING" or not self.metrics_started:
            self.footer_status.setText("Target-loss test: wait until beacon status is TRACKING")
            return

        # 0.30 s at the configured 30 Hz update rate, safely below the 1 s
        # re-acquisition requirement and long enough to make the state change visible.
        duration_frames = max(6, int(round(config.UPDATE_RATE * 0.30)))
        start_frame = int(self.simulator.frame_count) + 1
        self.simulator.set_beacon_loss(start_frame, duration_frames)
        duration_s = duration_frames / max(config.UPDATE_RATE, 1)
        self.forced_loss_active = True
        self.forced_loss_start_frame = None
        self.forced_loss_frame_end = None
        self.forced_loss_reacq_s = None
        self.footer_status.setText(
            f"TEST: FORCE TARGET LOSS armed — beacon hidden {duration_s:.2f}s • "
            f"expect TRACKING → LOSS → RE-ACQUIRING → TRACKING"
        )

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    def _safe_update_simulation(self):
        """Run one simulation tick without letting a Qt slot failure look like a frozen GUI."""
        try:
            self.update_simulation()
        except Exception as exc:
            traceback.print_exc()
            self.timer.stop()
            message = f"SIMULATION ERROR: {type(exc).__name__}: {exc}"
            if hasattr(self, "footer_status"):
                self.footer_status.setText(message)
            if hasattr(self, "camera_label"):
                self.camera_label.set_frame(None, message)
            try:
                QMessageBox.critical(self, "Simulation Error", message + "\n\nSee the terminal for the full traceback.")
            except Exception:
                pass

    def update_simulation(self):
        if not self._authenticated or not self.simulation_running:
            return

        processing_start = time.perf_counter()
        frame = self.simulator.get_frame()
        ground_truth = self.simulator.get_ground_truth_image_position()
        target_states = self.simulator.get_target_states()
        all_target_angles = self.simulator.get_all_target_angles()

        # IMPORTANT: put the raw simulator image on screen immediately.
        # This keeps the virtual camera visible even if a later tracking/UI
        # calculation has a runtime problem. The tracking pipeline still uses
        # the original NumPy frame unchanged.
        self._display_camera_frame(frame, "CAMERA ACQUIRING…")

        expected_position = None
        if (
            self.tracker.state_vector is not None
            and self.tracker.x is not None
            and self.tracker.state != "LOST"
        ):
            expected_position = (float(self.tracker.x), float(self.tracker.y))
        # A LOST track keeps publishing its dead prediction, which drifts far
        # from the real beacon. Feeding that stale prediction as the expected
        # position makes the detector's spatial gate reject the still-visible
        # beacon forever (observed: mid-flight Atmosphere switch -> permanent
        # LOST at 500+ px error with the beacon fully visible). When the track
        # is already dead there is no gate to protect, so search the whole
        # frame with the signature gate — exactly the acquisition behaviour
        # the system already uses at mission start. Live tracks (TRACKING /
        # PREDICTING / RE-ACQUIRING / ACQUIRING) keep the gate unchanged.

        detection, detection_info = detect_beacon(
            frame,
            return_details=True,
            expected_position=expected_position,
            preferred_signature=self.simulator.get_designated_target_signature(),
        )
        tracked = self.tracker.update(detection)
        self.frames_processed += 1

        center_x = config.FRAME_WIDTH // 2
        center_y = config.FRAME_HEIGHT // 2

        centroid_error = None
        if ground_truth is not None:
            self.visible_frames += 1
            if detection is None:
                self.detection_misses += 1
            else:
                centroid_error = math.hypot(
                    float(detection[0]) - ground_truth[0],
                    float(detection[1]) - ground_truth[1],
                )
                self.centroid_error_sum += centroid_error
                self.centroid_error_count += 1
                self.average_centroid_error = self.centroid_error_sum / self.centroid_error_count
                self.maximum_centroid_error = max(self.maximum_centroid_error, centroid_error)
                self.centroid_error_history.append(float(centroid_error))

        if tracked is not None:
            x, y, state = tracked
            measured_error = math.hypot(x - center_x, y - center_y)
            self.camera_target_history.append((int(x), int(y)))
            self.camera_target_history = self.camera_target_history[-36:]

            # Predict one frame ahead from the Kalman velocity estimate.
            lead_dt = 1.0 / config.UPDATE_RATE
            control_x = float(x) + self.tracker.velocity_x * lead_dt * self.prediction_gain
            control_y = float(y) + self.tracker.velocity_y * lead_dt * self.prediction_gain
            control_x = max(0.0, min(config.FRAME_WIDTH - 1.0, control_x))
            control_y = max(0.0, min(config.FRAME_HEIGHT - 1.0, control_y))

            if state in {"TRACKING", "RE-ACQUIRING", "PREDICTING"}:
                pan, tilt = self.controller.update(
                    control_x - center_x,
                    control_y - center_y,
                )
                self.simulator.update_camera(pan, tilt)
            else:
                pan, tilt = self.controller.pan, self.controller.tilt

            # Acquisition window: do not pollute tracking accuracy with the
            # initial corner-to-center acquisition manoeuvre.
            if state == "TRACKING" and measured_error <= self.lock_error_threshold_px:
                self.lock_candidate_frames += 1
            else:
                self.lock_candidate_frames = 0

            if not self.metrics_started and self.lock_candidate_frames >= self.lock_required_frames:
                self.metrics_started = True
                self.acquisition_time_s = self.frames_processed / config.UPDATE_RATE

            # Evaluation begins only after acquisition.
            if self.metrics_started and ground_truth is not None:
                self.evaluation_visible_frames += 1
                if detection is None:
                    self.evaluation_detection_misses += 1

            if self.metrics_started and state == "TRACKING":
                self.tracking_frames += 1
                self.error_sum += measured_error
                self.error_squared_sum += measured_error * measured_error
                self.error_count += 1
                self.average_error = self.error_sum / self.error_count
                self.maximum_error = max(self.maximum_error, measured_error)
                self.error_history.append(float(measured_error))

            # Loss / re-acquisition events are based on detector availability.
            # Forced beacon loss makes ground truth unavailable by design, so
            # this must not depend on ground_truth being present.
            if self.metrics_started and detection is None:
                if not self.in_loss:
                    self.loss_events += 1
                    self.in_loss = True
                    self.loss_start_time = self.frames_processed / config.UPDATE_RATE
                    if self.forced_loss_active:
                        # The armed forced-loss test consumed exactly this one
                        # intentional loss event.
                        self.forced_loss_active = False
                        self.forced_loss_start_frame = int(self.frames_processed)
                        self.forced_loss_frame_end = None
            elif self.in_loss and state in {"RE-ACQUIRING", "TRACKING"}:
                if self.loss_start_time is not None:
                    reacq = self.frames_processed / config.UPDATE_RATE - self.loss_start_time
                    self.reacquisition_times.append(reacq)
                    self.successful_reacquisitions += 1
                    if self.forced_loss_start_frame is not None and self.forced_loss_frame_end is None:
                        self.forced_loss_frame_end = int(self.frames_processed)
                        self.forced_loss_reacq_s = float(reacq)
                self.in_loss = False
                self.loss_start_time = None

            # -----------------------------------------------------------
            # Camera-view overlay: make the closed-loop follow action
            # immediately visible without changing any measurements.
            # -----------------------------------------------------------
            pred_x, pred_y = int(round(control_x)), int(round(control_y))
            self.camera_prediction_history.append((pred_x, pred_y))
            self.camera_prediction_history = self.camera_prediction_history[-18:]

            # Short trail of the measured target position. Because the camera
            # is actively following the target, the trail stays close to the
            # optical centre instead of the beacon disappearing from view.
            overlay = frame.copy()
            history_len = len(self.camera_target_history)
            for index, (hx, hy) in enumerate(self.camera_target_history):
                alpha = (index + 1) / max(1, history_len)
                radius = 2 if index < history_len - 5 else 3
                intensity = int(55 + 125 * alpha)
                cv2.circle(overlay, (hx, hy), radius, (0, intensity, 120), -1)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0)

            # Optical-centre reticle and small acquisition corridor.
            zone = 9
            cv2.rectangle(
                frame,
                (center_x - zone, center_y - zone),
                (center_x + zone, center_y + zone),
                (70, 120, 90),
                1,
            )
            cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 1)
            cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 1)
            cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

            # Detected beacon: green square + acquisition ring.
            half = max(5, int(round(self.beacon_size_px / 2.0)))
            cv2.rectangle(
                frame,
                (x - half, y - half),
                (x + half, y + half),
                (0, 255, 0),
                2,
            )
            cv2.circle(frame, (x, y), 11, (0, 255, 0), 1)

            # Kalman predicted position: amber ring + vector.
            cv2.circle(frame, (pred_x, pred_y), 6, (0, 215, 255), 2)
            cv2.line(frame, (x, y), (pred_x, pred_y), (0, 215, 255), 1)

            # Current camera-centre-to-target error vector.
            cv2.line(frame, (center_x, center_y), (x, y), (255, 255, 0), 1)

            # Mark all detected candidates lightly, while emphasizing the
            # designated target selected by the optical signature.
            for candidate in detection_info.get("candidates", []):
                cbx, cby, cbw, cbh = candidate["bbox"]
                cv2.rectangle(frame, (cbx, cby), (cbx + cbw, cby + cbh), (120, 120, 120), 1)
                cv2.putText(
                    frame,
                    f"CAND {candidate['score']:.2f}",
                    (cbx, max(12, cby - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (170, 170, 170),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"TARGET {self.designated_target_id}",
                (config.FRAME_WIDTH - 170, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 80),
                2,
                cv2.LINE_AA,
            )

            # Status text directly on the camera feed.
            if state == "TRACKING":
                cv2.putText(
                    frame,
                    "BEACON LOCKED",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (80, 255, 120),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "CAMERA FOLLOWING TARGET",
                    (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (80, 210, 255),
                    1,
                    cv2.LINE_AA,
                )
            elif state == "PREDICTING":
                cv2.putText(
                    frame,
                    "BEACON PREDICTING",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Forced-loss test banners (display-only). They make the phases of
            # the intentional test obvious on the virtual camera without any
            # effect on detection, tracking, or the measured statistics.
            if self.in_loss:
                cv2.rectangle(frame, (0, 0), (config.FRAME_WIDTH, 34), (40, 40, 210), -1)
                cv2.putText(
                    frame,
                    "TARGET LOSS TEST - BEACON SIGNAL HIDDEN",
                    (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif state == "RE-ACQUIRING":
                cv2.rectangle(frame, (0, 0), (config.FRAME_WIDTH, 34), (30, 170, 250), -1)
                cv2.putText(
                    frame,
                    "RE-ACQUISITION IN PROGRESS - KALMAN COASTING",
                    (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (20, 30, 40),
                    2,
                    cv2.LINE_AA,
                )
            elif self.forced_loss_reacq_s is not None and state == "TRACKING":
                cv2.rectangle(frame, (0, 0), (config.FRAME_WIDTH, 34), (40, 170, 60), -1)
                cv2.putText(
                    frame,
                    f"RE-ACQUIRED IN {self.forced_loss_reacq_s:.2f} s - LINK RESTORED",
                    (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Camera telemetry directly on the frame.
            cv2.putText(
                frame,
                f"PAN {pan:+.2f} deg",
                (12, config.FRAME_HEIGHT - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"TILT {tilt:+.2f} deg",
                (12, config.FRAME_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"ERR {measured_error:.2f} px",
                (config.FRAME_WIDTH - 115, config.FRAME_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (80, 255, 120),
                1,
                cv2.LINE_AA,
            )

            # Small zoomed beacon inset: gives judges a clear view of the
            # detection/prediction relationship without changing the source frame.
            zoom_size = 96
            zoom_scale = 2
            zx0 = max(0, x - zoom_size // 2)
            zy0 = max(0, y - zoom_size // 2)
            zx1 = min(config.FRAME_WIDTH, zx0 + zoom_size)
            zy1 = min(config.FRAME_HEIGHT, zy0 + zoom_size)
            crop = frame[zy0:zy1, zx0:zx1]
            if crop.size:
                zoom = cv2.resize(crop, (zoom_size * zoom_scale, zoom_size * zoom_scale), interpolation=cv2.INTER_NEAREST)
                zh, zw = zoom.shape[:2]
                x1 = config.FRAME_WIDTH - zw - 10
                y1 = config.FRAME_HEIGHT - zh - 35
                if x1 >= 0 and y1 >= 0:
                    cv2.rectangle(frame, (x1 - 2, y1 - 22), (x1 + zw + 2, y1 + zh + 2), (34, 48, 60), -1)
                    frame[y1:y1 + zh, x1:x1 + zw] = zoom
                    cv2.rectangle(frame, (x1, y1), (x1 + zw - 1, y1 + zh - 1), (100, 150, 120), 1)
                    cv2.putText(frame, "BEACON ZOOM 2x", (x1 + 6, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 220), 1, cv2.LINE_AA)

            velocity = math.hypot(self.tracker.velocity_x, self.tracker.velocity_y)
            confidence = float(getattr(self.tracker, "confidence", 0.0))
            uncertainty = getattr(self.tracker, "position_uncertainty", None)
            ai_conf = float(detection_info.get("ai_score", 0.0))

            self._update_kalman_panel(state, control_x, control_y, lead_dt, uncertainty)

            # Radar prediction uses camera-relative predicted position.
            ppd_x = config.FRAME_WIDTH / config.FOV_HORIZONTAL
            ppd_y = config.FRAME_HEIGHT / config.FOV_VERTICAL
            platform_x, platform_y = getattr(self.simulator, "last_platform_offset", (0.0, 0.0))
            jitter_x, jitter_y = getattr(self.simulator, "last_camera_jitter", (0.0, 0.0))
            predicted_world = (
                self.simulator.camera_pan
                + (control_x - center_x + platform_x + jitter_x) / ppd_x,
                self.simulator.camera_tilt
                + (control_y - center_y + platform_y + jitter_y) / ppd_y,
            )
            target_relative = self.simulator.get_target_angles()
            target_world = (
                self.simulator.camera_pan + target_relative[0],
                self.simulator.camera_tilt + target_relative[1],
            )
            self.radar.set_data(
                target_world,
                predicted_world,
                (self.simulator.camera_pan, self.simulator.camera_tilt),
                state,
                self.beacon_range_m,
                targets=all_target_angles,
                designated_id=self.designated_target_id,
            )

            self.radar_state.setText(f"● {state}")
            self.camera_lock_badge.setText(f"● {state}")
            self.t_fps.setText(f"{self.fps:.1f}")
            self.t_error.setText(f"{measured_error:.2f} px")
            self.t_centroid.setText(f"{centroid_error:.2f} px" if centroid_error is not None else "—")
            self.t_avg.setText(f"{self.average_error:.2f} px")
            self.t_max.setText(f"{self.maximum_error:.2f} px")
            rmse = math.sqrt(self.error_squared_sum / self.error_count) if self.error_count else 0.0
            self.t_rmse.setText(f"{rmse:.2f} px")
            self.t_acq.setText(f"{self.acquisition_time_s:.2f} s" if self.acquisition_time_s is not None else "ACQUIRING")
            self.t_reacq.setText(
                f"{self.reacquisition_times[-1]:.2f} s"
                if len(self.reacquisition_times) == 1
                else (
                    f"{self.reacquisition_times[-1]:.2f} s  (avg {(sum(self.reacquisition_times) / len(self.reacquisition_times)):.2f})"
                    if self.reacquisition_times else "—"
                )
            )
            self.t_loss.setText(f"{self._target_loss_rate():.2f} %")
            self.t_lock.setText(f"{self._lock_retention():.2f} %")
            self.t_state.setText(state)
            self.t_pan.setText(f"{pan:.2f}°")
            self.t_tilt.setText(f"{tilt:.2f}°")
            self.t_velocity.setText(f"{velocity:.2f} px/s")
            self.t_confidence.setText(f"{confidence:.3f}")
            self.t_ai.setText(f"{ai_conf:.3f}")
            frame_error_text = f"{measured_error:.2f} px"
            frame_center_text = (
                f"{math.hypot(x - config.FRAME_WIDTH / 2.0, y - config.FRAME_HEIGHT / 2.0):.2f} px"
                if tracked is not None else "—"
            )
            self._sync_mission_strip(center=frame_center_text, error=frame_error_text)
            self._push_strip_to_twin_surfaces(
                center=frame_center_text,
                error=frame_error_text,
            )
        else:
            state = self.tracker.state
            if self.metrics_started and detection is None and not self.in_loss:
                self.loss_events += 1
                self.in_loss = True
                self.loss_start_time = self.frames_processed / config.UPDATE_RATE
                if self.forced_loss_active:
                    self.forced_loss_active = False
                    self.forced_loss_start_frame = int(self.frames_processed)
                    self.forced_loss_frame_end = None
            target_relative = self.simulator.get_target_angles()
            target_world = (
                self.simulator.camera_pan + target_relative[0],
                self.simulator.camera_tilt + target_relative[1],
            )
            self.radar.set_data(
                target_world,
                None,
                (self.simulator.camera_pan, self.simulator.camera_tilt),
                state,
                self.beacon_range_m,
                targets=all_target_angles,
                designated_id=self.designated_target_id,
            )
            self.radar_state.setText(f"● {state}")
            self.camera_lock_badge.setText(f"● {state}")
            self.t_state.setText(state)
            self.t_centroid.setText("—")
            self.t_ai.setText(f"{float(detection_info.get('ai_score', 0.0)):.3f}")
            frame_error_text = "—"
            frame_center_text = "—"
            self._sync_mission_strip(center=frame_center_text, error=frame_error_text)
            self._push_strip_to_twin_surfaces(
                center=frame_center_text,
                error=frame_error_text,
            )
            self.t_pan.setText(f"{self.controller.pan:.2f}°")
            self.t_tilt.setText(f"{self.controller.tilt:.2f}°")
            self._update_kalman_panel(
                state,
                None,
                None,
                1.0 / config.UPDATE_RATE,
                getattr(self.tracker, "position_uncertainty", None),
            )
            # Fix 1: keep accumulation-based and always-live labels current
            # during LOST / SEARCHING so the telemetry panel never goes stale.
            self.t_error.setText("— px")
            self.t_avg.setText(f"{self.average_error:.2f} px" if self.error_count else "—")
            self.t_max.setText(f"{self.maximum_error:.2f} px" if self.error_count else "—")
            rmse_else = math.sqrt(self.error_squared_sum / self.error_count) if self.error_count else 0.0
            self.t_rmse.setText(f"{rmse_else:.2f} px" if self.error_count else "—")
            self.t_acq.setText(f"{self.acquisition_time_s:.2f} s" if self.acquisition_time_s is not None else "ACQUIRING")
            self.t_reacq.setText(
                f"{self.reacquisition_times[-1]:.2f} s"
                if len(self.reacquisition_times) == 1
                else (
                    f"{self.reacquisition_times[-1]:.2f} s  (avg {(sum(self.reacquisition_times) / len(self.reacquisition_times)):.2f})"
                    if self.reacquisition_times else "—"
                )
            )
            self.t_loss.setText(f"{self._target_loss_rate():.2f} %")
            self.t_lock.setText(f"{self._lock_retention():.2f} %" if hasattr(self, '_lock_retention') else "—")
            self.t_velocity.setText("— px/s")
            self.t_confidence.setText(f"{self.tracker.confidence:.3f}" if self.tracker.state != "SEARCHING" else "—")

        self.error_history = self.error_history[-160:]
        self.centroid_error_history = self.centroid_error_history[-160:]
        self.pan_history.append(float(self.controller.pan))
        self.tilt_history.append(float(self.controller.tilt))
        self.pan_history = self.pan_history[-160:]
        self.tilt_history = self.tilt_history[-160:]
        self.error_graph.set_series([
            ("Tracking", self.error_history, Qt.green),
            ("Centroid", self.centroid_error_history, Qt.yellow),
            ("10 px limit", [10.0] * max(1, len(self.error_history)), Qt.red),
        ])
        self.motion_graph.set_series([
            ("Pan", self.pan_history, Qt.blue),
            ("Tilt", self.tilt_history, Qt.yellow),
        ])

        # --- Live data feed to the TEST/OUTPUT graph panel ---
        # Every processed frame APPENDS one real measurement taken from this
        # very iteration, so the benchmark performance graphs are a genuine
        # time series that changes with the live simulation conditions
        # (noise, turbulence, jitter, platform motion, beacon speed, FOV,
        # prediction gain, …). This is the same per-frame append path the
        # video benchmark uses: no value is invented, nothing is interpolated
        # after the fact, and a fresh scenario starts from an empty panel.
        live_sample = {
            "frame": self.frames_processed,
            "timestamp_s": self.frames_processed / max(config.UPDATE_RATE, 1),
            "state": self.tracker.state,
            # Tracking error = detected target position vs the optical centre
            # (the same offset the aiming loop corrects). A gap before lock is
            # honest: there is no tracking error to plot until there is a track.
            "center_error": (float(measured_error) if tracked is not None else None),
            "centroid_error": (float(self.centroid_error_history[-1]) if self.centroid_error_history else None),
            "target_x": (float(detection[0]) if detection is not None else None),
            "target_y": (float(detection[1]) if detection is not None else None),
            "predicted_x": (float(control_x) if tracked is not None else None),
            "predicted_y": (float(control_y) if tracked is not None else None),
            "pan": float(getattr(self.controller, "pan", 0.0)),
            "tilt": float(getattr(self.controller, "tilt", 0.0)),
            "fps": float(self.fps) if getattr(self, "fps", None) is not None else None,
            "filter_confidence": float(getattr(self.tracker, "confidence", 0.0)),
            "detector_confidence": float(detection_info.get("confidence", 0.0)),
            "ai_confidence": float(detection_info.get("ai_score", 0.0)),
        }
        try:
            self.graph_panel.append_live_sample(
                live_sample["timestamp_s"], self.frames_processed, live_sample
            )
        except Exception:
            # Never let a graph-panel update crash the simulation loop.
            pass

        # Processing speed excludes Qt image display overhead.
        processing_elapsed = max(time.perf_counter() - processing_start, 1e-6)
        self.processing_times.append(processing_elapsed)
        self.processing_times = self.processing_times[-30:]
        self.fps = len(self.processing_times) / max(sum(self.processing_times), 1e-6)
        self.t_fps.setText(f"{self.fps:.1f}")

        self.info.setText(
            "CURRENT SCENARIO\n"
            f"Camera {config.FRAME_WIDTH} × {config.FRAME_HEIGHT} | FOV {config.FOV_HORIZONTAL:.1f}° × {config.FOV_VERTICAL:.1f}°\n"
            f"Targets {self.target_count} | Designated {self.designated_target_id} | Signature {self.simulator.get_designated_target_signature()}\n"
            f"Beacon {self.beacon_size_px:.0f} px | Range {self.beacon_range_m:.0f} m | Speed {self.beacon_speed:.1f} px/frame\n"
            f"Visible objects {sum(1 for t in target_states if t.get('visible'))} | Detector candidates {len(detection_info.get('candidates', []))}\n"
            f"Noise {self.noise_selector.currentText()} level {self.noise_level}\n"
            f"Turbulence {'ON' if self.turbulence_check.isChecked() else 'OFF'} | Jitter {'ON' if self.jitter_check.isChecked() else 'OFF'} | Platform {'ON' if self.platform_check.isChecked() else 'OFF'}\n"
            f"Atmosphere {self.atmosphere_selector.currentText()} {self.atmosphere_level}%\n"
            "AI Beacon Verification: ON  •  Kalman Prediction: ON"
        )

        self.camera_info.setText(
            f"{config.FRAME_WIDTH} × {config.FRAME_HEIGHT}  |  FOV {config.FOV_HORIZONTAL:.1f}° × {config.FOV_VERTICAL:.1f}°  |  UPDATE {config.UPDATE_RATE} Hz  |  TARGETS {self.target_count}"
        )

        designated_signature = self.simulator.get_designated_target_signature()
        designated_color, designated_shape = designated_signature.split("-", 1) if "-" in designated_signature else (designated_signature, "")
        self.footer_status.setText(
            f"{self.motion_selector.currentText()} • {self.noise_selector.currentText()} • "
            f"Tracking: {self.designated_target_id} | COLOR: {designated_color} | SHAPE: {designated_shape} • "
            f"Targets {self.target_count} • "
            f"Status {'LOCKED' if self.metrics_started else 'ACQUIRING'} • "
            f"Loss {self._target_loss_rate():.2f}%"
        )

        # Show final camera image. Tracking always used the original frame.
        # `tracked` is the actual variable returned by the tracker.
        # (The old code accidentally used an undefined `tracked_position`,
        # which stopped execution before the frame reached the widget.)
        if tracked is not None:
            display_focus = (control_x, control_y)
        else:
            display_focus = (center_x, center_y)
        camera_status = f"{state}  •  {self.display_zoom:.1f}× DISPLAY"
        # Display-only disturbance indication. The values are the SAME live
        # simulator offsets the tracking loop is subject to, so the picture and
        # the telemetry can never disagree.
        self._draw_disturbance_overlay(frame)
        self._display_camera_frame(frame, camera_status, display_focus)

        # Link-readiness indicator: this is a coarse-alignment gate, not a
        # physical communication-link guarantee. The gate deliberately uses
        # the LIVE detector/tracker state plus the same PS-style coarse
        # alignment limits, and requires a short stable run so the badge does
        # not flicker READY on a single lucky frame.
        current_state = self.tracker.state
        current_error = measured_error if tracked is not None else None
        # Real tracker-vs-frame-center offset (640x480 px) for the twin's
        # BEAM IMPACT visualization — the same measurement, display only.
        current_offset_px = (
            (float(tracked[0]) - center_x, float(tracked[1]) - center_y)
            if tracked is not None
            else (0.0, 0.0)
        )
        detector_conf = float(detection_info.get("confidence", 0.0))
        ai_conf = float(detection_info.get("ai_score", 0.0))

        ready_frame = (
            current_state == "TRACKING"
            and self.metrics_started
            and self.acquisition_time_s is not None
            and self.acquisition_time_s <= 2.0
            and detection is not None
            and current_error is not None
            and current_error <= 10.0
            and self.fps >= 20.0
            and detector_conf >= 0.55
            and float(detection_info.get("signature_score", 0.0)) >= 0.75
            and ai_conf >= config.LINK_READY_AI_THRESHOLD
        )

        if ready_frame:
            self.link_ready_streak += 1
        else:
            self.link_ready_streak = 0

        ready = self.link_ready_streak >= self.link_required_frames
        # HANDOFF WINDOW bookkeeping rides on the exact same gate verdict —
        # pure display-state, never fed back into tracking or the gate.
        self._update_handoff_window(
            ready_frame,
            current_error,
            float(detection_info.get("signature_score", 0.0)),
            ai_conf,
        )
        # WHY LOCKED? evidence: the same gate conditions, itemized with their
        # real values so the dialog can explain the verdict truthfully.
        self._why_locked_evidence = self._build_why_locked_evidence(
            ready,
            current_state,
            current_error,
            detector_conf,
            ai_conf,
            float(detection_info.get("signature_score", 0.0)),
            detection is not None,
        )
        if self._why_locked_dialog is not None:
            try:
                self._why_locked_dialog.update_evidence(self._why_locked_evidence)
            except RuntimeError:
                self._why_locked_dialog = None

        if ready:
            self.link_readiness_label.setText("● HANDOFF READY  •  LINK READY")
            self.link_readiness_label.setObjectName("link_ready")
        elif current_state in {"PREDICTING", "RE-ACQUIRING"}:
            self.link_readiness_label.setText("● RECOVERING  •  LINK NOT READY")
            self.link_readiness_label.setObjectName("link_recover")
        elif current_state == "TRACKING":
            self.link_readiness_label.setText("● ALIGNING  •  LINK NOT READY")
            self.link_readiness_label.setObjectName("link_not_ready")
        else:
            self.link_readiness_label.setText("● NOT READY")
            self.link_readiness_label.setObjectName("link_not_ready")

        self.link_readiness_label.style().unpolish(self.link_readiness_label)
        self.link_readiness_label.style().polish(self.link_readiness_label)

        self._update_digital_twin(
            state,
            predicted_world if tracked is not None else None,
            target_states,
            center_error_px=current_error,
            offset_px=current_offset_px,
            link_ready=ready,
        )

        error_text = f"{current_error:.2f} px" if current_error is not None else "—"
        self.link_conditions_label.setText(
            f"Tracking: {current_state}  |  Error: {error_text}  |  "
            f"FPS: {self.fps:.1f}  |  Detector: {detector_conf:.2f}  |  "
            f"AI: {ai_conf:.2f}  |  Signature: {float(detection_info.get('signature_score', 0.0)):.2f}  |  Stable: {self.link_ready_streak}/{self.link_required_frames}"
        )

        # Refinement 5: record a complete point-in-time telemetry snapshot on
        # every simulation frame.  This buffer is what the Instant Report
        # dialog queries for times such as 35s, 2m, or the current moment.
        # Keep the history bounded so long demonstrations do not grow
        # memory indefinitely (18,000 frames ~= 10 minutes at 30 Hz).
        simulation_time_s = self.frames_processed / max(config.UPDATE_RATE, 1)
        designated_signature = self.simulator.get_designated_target_signature()
        if "-" in designated_signature:
            target_color, target_shape = designated_signature.split("-", 1)
        else:
            target_color, target_shape = designated_signature, ""

        snapshot = {
            "simulation_time_s": float(simulation_time_s),
            "frame": int(self.frames_processed),
            "target_id": self.designated_target_id,
            "target_signature": designated_signature,
            "target_color": target_color,
            "target_shape": target_shape,
            "target_count": int(self.target_count),
            "state": str(current_state),
            "target_x_px": float(x) if tracked is not None else None,
            "target_y_px": float(y) if tracked is not None else None,
            "predicted_x_px": float(pred_x) if tracked is not None else None,
            "predicted_y_px": float(pred_y) if tracked is not None else None,
            "pan_deg": float(self.simulator.camera_pan),
            "tilt_deg": float(self.simulator.camera_tilt),
            "center_error_px": float(current_error) if current_error is not None else None,
            "centroid_error_px": float(centroid_error) if centroid_error is not None else None,
            "fps": float(self.fps),
            "detector_confidence": float(detector_conf),
            "ai_confidence": float(ai_conf),
            "signature_score": float(detection_info.get("signature_score", 0.0)),
            "acquisition_time_s": self.acquisition_time_s,
            # Real deviation statistics from the same accumulators that feed
            # the live telemetry and the normal performance report, so the
            # instant report cannot drift from them (Max Deviation included).
            "average_tracking_error_px": float(self.average_error) if self.error_count else None,
            "maximum_tracking_error_px": float(self.maximum_error) if self.error_count else None,
            "rmse_tracking_error_px": (
                float(math.sqrt(self.error_squared_sum / self.error_count))
                if self.error_count else None
            ),
            "lock_retention_percentage": float(self._lock_retention()),
            "target_loss_percentage": float(self._target_loss_rate()),
            "loss_events": int(self.loss_events),
            "reacquisition_count": int(self.successful_reacquisitions),
            # Forced-loss test facts, kept honest: None until the intentional
            # loss/re-acquisition cycle has actually completed.
            "forced_loss_reacq_s": (
                float(self.forced_loss_reacq_s)
                if self.forced_loss_reacq_s is not None else None
            ),
            "forced_loss_last_duration_s": (
                (self.forced_loss_frame_end - self.forced_loss_start_frame)
                / max(config.UPDATE_RATE, 1)
                if (self.forced_loss_frame_end is not None and self.forced_loss_start_frame is not None)
                else None
            ),
            "average_reacquisition_s": (
                float(sum(self.reacquisition_times) / len(self.reacquisition_times))
                if self.reacquisition_times else None
            ),
            "noise": self.noise_selector.currentText(),
            "noise_level": int(self.noise_level),
            "turbulence_enabled": bool(self.turbulence_check.isChecked()),
            "turbulence_strength_px": float(self.turbulence_strength),
            "camera_jitter_enabled": bool(self.jitter_check.isChecked()),
            "camera_jitter_px": float(self.camera_jitter_strength),
            "platform_motion_enabled": bool(self.platform_check.isChecked()),
            "platform_motion_px": float(self.platform_motion_strength),
            "atmosphere": self.atmosphere_selector.currentText(),
            "atmosphere_level": int(self.atmosphere_level),
            "link_ready": bool(ready),
        }
        self.instant_report_history.append(snapshot)
        self.instant_report_history = self.instant_report_history[-18000:]
        self._last_report_snapshot = snapshot

    def _on_zoom_changed(self, value):
        self.display_zoom = max(1.0, min(4.0, float(value) / 10.0))
        self.zoom_value_label.setText(f"{self.display_zoom:.1f}×")

    def _step_zoom(self, delta):
        new_value = int(round((self.display_zoom + delta) * 10.0))
        new_value = max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), new_value))
        self.zoom_slider.setValue(new_value)

    def _reset_zoom(self):
        self.zoom_slider.setValue(10)

    @staticmethod
    def _ascii_level_text(level):
        """OpenCV's Hershey fonts are ASCII-only, so the em dash is swapped."""
        return disturbance_level_text(level).replace("\u2014", "-")

    def _disturbance_overlay_lines(self):
        """Active disturbance categories with their REAL live magnitudes."""
        sim = self.simulator
        lines = []
        if self.turbulence_check.isChecked() and self.turbulence_strength > 0:
            lines.append(
                f"ATMOS TURBULENCE  {self._ascii_level_text(self.turbulence_slider.value())}"
                f"  ({float(getattr(sim, 'turbulence_strength_px', 0.0)):.0f} px)"
            )
        if self.jitter_check.isChecked() and self.camera_jitter_strength > 0:
            jx, jy = getattr(sim, "last_camera_jitter", (0.0, 0.0))
            lines.append(
                f"CAMERA JITTER  {self._ascii_level_text(self.jitter_slider.value())}"
                f"  ({math.hypot(float(jx), float(jy)):.1f} px shake)"
            )
        if self.platform_check.isChecked() and self.platform_motion_strength > 0:
            px, py = getattr(sim, "last_platform_offset", (0.0, 0.0))
            lines.append(
                f"PLATFORM MOTION  {self._ascii_level_text(self.platform_slider.value())}"
                f"  ({math.hypot(float(px), float(py)):.1f} px offset)"
            )
        return lines

    def _draw_disturbance_overlay(self, frame):
        """Announce the ACTIVE disturbances on the virtual camera image.

        The banner lists the enabled categories with their real magnitudes,
        and the orange arrow shows the instantaneous platform + jitter
        displacement that is actually being applied to the scene.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return
        lines = self._disturbance_overlay_lines()
        if not lines:
            return

        # Instantaneous displacement indicator (real offsets, display only).
        px, py = getattr(self.simulator, "last_platform_offset", (0.0, 0.0))
        jx, jy = getattr(self.simulator, "last_camera_jitter", (0.0, 0.0))
        dx, dy = float(px) + float(jx), float(py) + float(jy)
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            cx, cy = config.FRAME_WIDTH // 2, config.FRAME_HEIGHT // 2
            ex = int(max(10, min(config.FRAME_WIDTH - 10, cx + dx * 1.4)))
            ey = int(max(10, min(config.FRAME_HEIGHT - 10, cy + dy * 1.4)))
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 165, 255), 2, tipLength=0.28)
            cv2.circle(frame, (ex, ey), 4, (0, 165, 255), -1)

        line_h = 15
        panel_h = line_h * (len(lines) + 1) + 8
        cv2.rectangle(frame, (8, 52), (398, 52 + panel_h), (16, 22, 32), -1)
        cv2.rectangle(frame, (8, 52), (398, 52 + panel_h), (60, 105, 150), 1)
        cv2.putText(
            frame, "DISTURBANCE ACTIVE", (16, 52 + line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 200, 255), 1, cv2.LINE_AA,
        )
        for index, line in enumerate(lines, start=1):
            cv2.putText(
                frame, line, (16, 52 + line_h * (index + 1)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (215, 228, 238), 1, cv2.LINE_AA,
            )

    def _display_camera_frame(self, frame, status="CAMERA FEED", focus_point=None):
        """Convert a simulator BGR frame to a Qt pixmap and display it safely.

        This is display-only. No tracking measurement or benchmark value is
        changed by zooming or drawing overlays. A defensive copy is made before
        QImage is constructed so the NumPy buffer can never become invalid.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.camera_label.set_frame(None, "NO CAMERA FRAME")
            return

        display_frame = self._prepare_display_frame(frame, focus_point)
        display_frame = np.ascontiguousarray(display_frame)

        if display_frame.ndim != 3 or display_frame.shape[2] != 3:
            self.camera_label.set_frame(None, "INVALID CAMERA FRAME")
            return

        try:
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            height, width = rgb.shape[:2]
            image = QImage(
                rgb.data,
                width,
                height,
                rgb.strides[0],
                QImage.Format_RGB888,
            ).copy()
            if image.isNull():
                self.camera_label.set_frame(None, "INVALID CAMERA FRAME")
                return
            self.camera_label.set_frame(QPixmap.fromImage(image), str(status))
        except Exception as exc:
            # Keep the rest of the simulation alive and make the failure visible
            # in the camera panel instead of leaving an unexplained black box.
            self.camera_label.set_frame(None, f"CAMERA DISPLAY ERROR: {type(exc).__name__}")

    def _prepare_display_frame(self, frame, focus_point=None):
        """Apply display-only zoom around the camera/beacon; tracking is unchanged."""
        zoom = max(1.0, min(4.0, float(self.display_zoom)))
        if zoom <= 1.001:
            return frame.copy()

        height, width = frame.shape[:2]
        if focus_point is None:
            cx, cy = width // 2, height // 2
        else:
            cx = int(round(focus_point[0]))
            cy = int(round(focus_point[1]))

        crop_w = max(8, int(round(width / zoom)))
        crop_h = max(8, int(round(height / zoom)))
        x0 = max(0, min(width - crop_w, cx - crop_w // 2))
        y0 = max(0, min(height - crop_h, cy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w].copy()
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_NEAREST)

    def _update_kalman_panel(self, state, control_x, control_y, lead_dt, uncertainty):
        mode = "PREDICTION ONLY" if state == "PREDICTING" else "MEASUREMENT + MODEL"
        self.kalman_state_label.setText(f"Prediction: {state}  |  {mode}")
        if control_x is None or control_y is None:
            self.kalman_pos_label.setText("Predicted Position: —")
            self.kalman_vel_label.setText("Estimated Velocity: —")
        else:
            self.kalman_pos_label.setText(f"Predicted Position: ({control_x:.1f}, {control_y:.1f}) px")
            self.kalman_vel_label.setText(
                f"Estimated Velocity: ({self.tracker.velocity_x:.1f}, {self.tracker.velocity_y:.1f}) px/s"
            )
        self.kalman_uncertainty_label.setText(
            f"Position Uncertainty: ±{float(uncertainty):.2f} px" if uncertainty is not None else "Position Uncertainty: —"
        )
        self.kalman_next_label.setText(f"Control Lead: {lead_dt * 1000.0:.0f} ms")

    def _target_loss_rate(self):
        if self.evaluation_visible_frames <= 0:
            return 0.0
        return 100.0 * self.evaluation_detection_misses / self.evaluation_visible_frames

    def _lock_retention(self):
        if self.evaluation_visible_frames <= 0:
            return 0.0
        return 100.0 * self.tracking_frames / self.evaluation_visible_frames

    # ------------------------------------------------------------------
    # Report opening helpers (PDF / folder), shared by every report path
    # ------------------------------------------------------------------
    def _open_path(self, path) -> bool:
        """Open a file or folder with the OS default handler.

        QDesktopServices first, os.startfile as the Windows fallback, and a
        clear error dialog instead of a silent no-op when both fail.
        """
        target = str(path) if path is not None else ""
        if not target:
            return False
        try:
            resolved = Path(target)
            if not resolved.exists():
                QMessageBox.warning(self, "Cannot Open", f"File not found:\n{resolved.resolve()}")
                return False
            if QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved.resolve()))):
                return True
            if hasattr(os, "startfile"):
                try:
                    os.startfile(str(resolved.resolve()))
                    return True
                except OSError:
                    pass
            QMessageBox.warning(
                self, "Cannot Open",
                f"No application is associated with:\n{resolved.resolve()}",
            )
            return False
        except Exception as exc:
            QMessageBox.warning(self, "Cannot Open", f"Could not open:\n{target}\n\n{exc}")
            return False

    def _show_report_dialog(self, title, paths):
        """Polished completion dialog: OPEN PDF / OPEN REPORT FOLDER / CLOSE."""
        pdf_path = Path(paths[0])
        artifact_lines = [
            f"{extension}:  {artifact}"
            for extension, artifact in zip(("PDF", "JSON", "CSV"), paths)
        ]
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Information)
        box.setText("Report generated successfully.")
        box.setInformativeText("\n".join(artifact_lines))
        open_pdf_button = box.addButton("OPEN PDF", QMessageBox.AcceptRole)
        open_folder_button = box.addButton("OPEN REPORT FOLDER", QMessageBox.ActionRole)
        close_button = box.addButton("CLOSE", QMessageBox.RejectRole)
        close_button.setDefault(True)
        box.exec()
        clicked = box.clickedButton()
        if clicked is open_pdf_button:
            self._open_path(pdf_path)
        elif clicked is open_folder_button:
            self._open_path(pdf_path.parent)

    def _open_last_report(self):
        if self.last_pdf_path is None or not Path(self.last_pdf_path).exists():
            QMessageBox.information(
                self, "No Report Yet",
                "Generate a report first — the latest PDF will be reopenable here.",
            )
            return
        self._open_path(self.last_pdf_path)

    # ------------------------------------------------------------------
    # Refinement 5: local authentication gate
    # ------------------------------------------------------------------
    def _build_auth_overlay(self, parent):
        overlay = QFrame(parent)
        overlay.setObjectName("auth_overlay")
        overlay.setGeometry(parent.rect())
        overlay.raise_()
        self.auth_overlay = overlay
        # The overlay is an unmanaged child (deliberately outside the layout),
        # so its geometry must be re-synced AFTER Qt finalizes the window
        # layout, otherwise a stale rect misplaces the card.
        QTimer.singleShot(0, self._sync_auth_overlay_geometry)

        outer = QVBoxLayout(overlay)
        outer.setContentsMargins(30, 30, 30, 30)
        outer.addStretch(1)

        panel = QFrame()
        panel.setObjectName("auth_panel")
        panel.setMaximumWidth(560)
        panel.setMinimumHeight(330)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(36, 32, 36, 30)
        panel_layout.setSpacing(10)
        # TRUE VISUAL CENTER: the card content floats mid-card (equal space
        # above and below) instead of packing against the card's top edge.
        panel_layout.addStretch(1)

        kicker = QLabel("MISSION CONTROL  /  SECURE ACCESS")
        kicker.setObjectName("auth_kicker")
        panel_layout.addWidget(kicker)

        title = QLabel("ASTRALINK")
        title.setObjectName("auth_title")
        panel_layout.addWidget(title)
        tagline = QLabel("AI-BASED VIRTUAL CAMERA TRACKING SYSTEM FOR MOBILE FSOC")
        tagline.setObjectName("auth_kicker")
        panel_layout.addWidget(tagline)
        panel_layout.addSpacing(14)

        label = QLabel("ACCESS KEY")
        label.setObjectName("auth_status")
        panel_layout.addWidget(label)

        self.auth_input = QLineEdit()
        self.auth_input.setObjectName("auth_input")
        self.auth_input.setPlaceholderText("Enter authorization key")
        self.auth_input.setEchoMode(QLineEdit.Password)
        self.auth_input.returnPressed.connect(self._authenticate)
        panel_layout.addWidget(self.auth_input)

        self.auth_button = QPushButton("AUTHENTICATE  →")
        self.auth_button.setObjectName("auth_button")
        self.auth_button.clicked.connect(self._authenticate)
        panel_layout.addWidget(self.auth_button)

        self.auth_error = QLabel("")
        self.auth_error.setObjectName("auth_error")
        self.auth_error.setMinimumHeight(18)
        panel_layout.addWidget(self.auth_error)

        panel_layout.addStretch(1)

        outer.addWidget(panel, 0, Qt.AlignHCenter)
        outer.addStretch(1)
        self.auth_input.setFocus()

    # ------------------------------------------------------------------
    # Responsive layout (no horizontal overflow at any window width)
    # ------------------------------------------------------------------
    def _balance_camera_column(self):
        """Hand the Virtual Camera panel every pixel the panels below it do not need.

        The camera feed is the primary element of its column, so LIVE
        TELEMETRY / KALMAN PREDICTION ENGINE / LINK READINESS are each capped at
        the minimum height Qt measures for their OWN existing content: every
        field, value and button they already show stays visible (Qt's minimum
        accounts for all of it) and none of their values, logic or order is
        touched. The Virtual Camera panel is the column's only flexible item,
        so it absorbs all of the remaining height at every window size.
        """
        for panel in getattr(self, "_stacked_panels", ()):
            if panel is None:
                continue
            hint = int(panel.minimumSizeHint().height())
            if hint < 60:
                # Layout has not settled yet -- try again on the next pass.
                continue
            if panel.maximumHeight() != hint:
                panel.setMaximumHeight(hint)

    def _relayout_support_row(self, side_by_side):
        """Put KALMAN + LINK READINESS side by side on a wide camera column.

        The Virtual Camera column is more than twice as wide as either of
        those two supporting panels needs, so sharing one row costs them
        nothing: each keeps the exact minimum height Qt measures for the
        fields it already shows (nothing is wrapped, clipped or hidden). The
        height the column no longer needs for them is then claimed by the
        Virtual Camera feed through _balance_camera_column.

        Below ``SUPPORT_ROW_MIN_WIDTH`` the two panels stack vertically again
        -- byte-for-byte the previous arrangement -- so a narrow window can
        never squeeze their labels or buttons.
        """
        if side_by_side == getattr(self, "_support_side_by_side", None):
            return
        self._support_side_by_side = side_by_side
        kalman = self._kalman_panel
        readiness = self._readiness_panel
        row = self._support_row
        column = self._right_column
        telemetry = self._telemetry_panel
        if side_by_side:
            for panel in (kalman, readiness):
                column.removeWidget(panel)
                self._support_row_layout.addWidget(panel, 1)
            row.setVisible(True)
            # The row (not the two panels) is what the column balances now.
            self._stacked_panels = (telemetry, row)
        else:
            for panel in (kalman, readiness):
                self._support_row_layout.removeWidget(panel)
                column.addWidget(panel, 0)
            row.setVisible(False)
            self._stacked_panels = (telemetry, kalman, readiness)
        # Clear the caps applied to the panels that just changed role, so
        # _balance_camera_column can re-cap exactly what is on screen now.
        for panel in (kalman, readiness, row):
            panel.setMaximumHeight(16777215)
        self._balance_camera_column()

    def _apply_responsive_widths(self):
        """Size the control column and metric grid to the current window.

        Called from resizeEvent only, and every change is guarded so a
        layout pass can never trigger a second, redundant one.
        """
        width = max(self.width(), 320)
        # Control column: wider than the old 300-360 clamp on desktop, and
        # still comfortable on a narrow window.
        panel_width = int(max(300, min(460, round(width * 0.26))))
        if self.control_scroll.minimumWidth() != panel_width:
            self.control_scroll.setMinimumWidth(panel_width)
            self.control_scroll.setMaximumWidth(panel_width)

        # LIVE TELEMETRY metric cards: 4 / 3 / 2 / 1 columns. The thresholds
        # keep every label and value at full width (no clipping) at each step.
        if width >= 1700:
            self._relayout_metric_grid(4)
        elif width >= 1200:
            self._relayout_metric_grid(3)
        elif width >= 860:
            self._relayout_metric_grid(2)
        else:
            self._relayout_metric_grid(1)

        # VIRTUAL CAMERA: re-balance the live column so the camera feed keeps
        # the largest 4:3 area the panel can give it at this window size.
        # KALMAN + LINK READINESS first share a row while the camera column is
        # wide enough for both at full content width: the height that frees
        # goes to the camera feed, and nothing else about them changes.
        self._relayout_support_row(width >= SUPPORT_ROW_MIN_WIDTH)
        self._balance_camera_column()

        # BENCHMARK PERFORMANCE GRAPHS: the embedded panel re-flows its own
        # grid (3 / 2 / 1 columns) from its own resizeEvent.

    def _relayout_metric_grid(self, columns):
        """Re-flow the telemetry metric cards into ``columns`` columns.

        The cards are removed from the grid first (a widget may not be added
        twice) and re-inserted.  The per-frame telemetry path is untouched:
        this only runs when the column count actually changes.
        """
        columns = max(1, min(4, int(columns)))
        if getattr(self, "_metric_grid_columns", None) == columns:
            return
        self._metric_grid_columns = columns
        grid = self._telemetry_grid
        for card in self._metric_cards:
            grid.removeWidget(card)
        grid.addWidget(self._telemetry_title, 0, 0, 1, columns)
        for index, card in enumerate(self._metric_cards):
            grid.addWidget(card, 1 + index // columns, index % columns)
        for col in range(4):
            grid.setColumnStretch(col, 1 if col < columns else 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self._apply_responsive_widths()
        except Exception:
            pass  # a layout pass must never break the mission loop
        if hasattr(self, "auth_overlay"):
            # Immediate sync plus a deferred sync that runs after Qt has
            # relaid out the central widget (covers maximize/restore and the
            # twin fullscreen enter/exit transitions).
            self._sync_auth_overlay_geometry()
            QTimer.singleShot(0, self._sync_auth_overlay_geometry)

    # ------------------------------------------------------------------
    # Stage 9: TRUE Digital-Twin full screen (whole window, ESC to exit)
    # ------------------------------------------------------------------
    def enter_twin_fullscreen(self):
        """Give the Digital Twin the entire application surface.

        Hides every other UI region (sidebar, header, view switch, footer and
        the QStackedWidget pages) without touching the mission: simulator,
        tracker, controller, frame clock and metrics keep running untouched,
        and the previous interface is restored verbatim on exit.
        """
        if getattr(self, "_twin_fullscreen", False):
            return
        self._twin_fullscreen = True
        # Remember the exact pre-fullscreen window state so ESC can put the
        # window back verbatim (normal size or maximized, mission untouched).
        self._twin_fs_state = self.windowState()
        self._twin_fs_geometry = self.saveGeometry()
        # Hide every chrome region around the view stack; the stack itself
        # stays visible and locked to the twin page, which then occupies the
        # whole central area (its own strip/scene rows are hidden by the
        # pane's full-screen chrome). Mission state is never touched.
        self._twin_fs_hidden = [
            w for w in (self.sidebar, self.header, self.footer)
            if w is not None and w.isVisible()
        ]
        for widget in self._twin_fs_hidden:
            widget.setVisible(False)
        self.twin_pane.set_fullscreen_chrome(True)
        self._switch_live_view(2)
        # TRUE OS fullscreen — not merely hidden chrome or a maximized window.
        self.showFullScreen()

    def exit_twin_fullscreen(self):
        """Restore the exact pre-fullscreen interface (mission state untouched)."""
        if not getattr(self, "_twin_fullscreen", False):
            return
        self._twin_fullscreen = False
        self.twin_pane.set_fullscreen_chrome(False)
        for widget in getattr(self, "_twin_fs_hidden", []):
            widget.setVisible(True)
        # Restore the exact pre-fullscreen window state (normal or maximized).
        geometry = getattr(self, "_twin_fs_geometry", None)
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = getattr(self, "_twin_fs_state", None)
        if state is not None:
            self.setWindowState(state & ~Qt.WindowFullScreen)
        # The view stays where the user was (twin page) — the interface is
        # restored exactly as it was before entering full screen.

    def keyPressEvent(self, event):
        # Stage 9: F11 toggles the true twin full screen while the twin page
        # is up; ESC always leaves it. Mission state is never touched.
        if getattr(self, "_twin_fullscreen", False) and event.key() == Qt.Key_F11:
            self.exit_twin_fullscreen()
            event.accept()
            return
        if (
            event.key() == Qt.Key_F11
            and self.main_stack.currentIndex() == 1
            and not getattr(self, "_twin_fullscreen", False)
        ):
            self.enter_twin_fullscreen()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and getattr(self, "_twin_fullscreen", False):
            self.exit_twin_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "auth_overlay"):
            QTimer.singleShot(0, self._sync_auth_overlay_geometry)
        # The first real layout only exists once the window is shown, so the
        # Virtual Camera panel is balanced to the true panel minimums here.
        QTimer.singleShot(0, self._balance_camera_column)

    def _sync_auth_overlay_geometry(self):
        """Track the central widget's live geometry so the authorization card
        is always horizontally and vertically centered, fully visible and
        independent of the mission-control layout."""
        overlay = getattr(self, "auth_overlay", None)
        central = self.centralWidget()
        if overlay is None or central is None:
            return
        if getattr(self, "_authenticated", False):
            return  # overlay is permanently hidden after authentication
        overlay.setGeometry(central.rect())
        overlay.raise_()

    def _authenticate(self):
        if self.auth_input.text() != APP_PASSWORD:
            self.auth_error.setText("ACCESS DENIED  •  INVALID AUTHORIZATION KEY")
            self.auth_input.selectAll()
            self.auth_input.setFocus()
            return

        self._authenticated = True
        self.auth_error.setStyleSheet("color:#67e9a0;")
        self.auth_error.setText("AUTHENTICATED  •  INITIALIZING OPTICAL TRACKING CORE…")
        self.auth_button.setEnabled(False)
        self.auth_input.setEnabled(False)

        effect = QGraphicsOpacityEffect(self.auth_overlay)
        self.auth_overlay.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(650)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.InOutCubic)
        self._auth_animation = animation

        def finish():
            self.auth_overlay.hide()
            self.auth_overlay.setGraphicsEffect(None)
            self.timer.start(33)
            self.system_status.setText("● SYSTEM ACTIVE")
            self.footer_status.setText("Authenticated • simulation running")

        animation.finished.connect(finish)
        animation.start()

    # ------------------------------------------------------------------
    # Reports / benchmark video
    # ------------------------------------------------------------------
    def _make_summary(self, source="live simulation"):
        duration = max(time.perf_counter() - self.run_start_time, 1e-6)
        # ONE FPS definition everywhere: the same sustained processing rate
        # shown on the dashboard card and gated by the readiness check —
        # never a second, diverging wall-clock calculation.
        fps = float(self.fps) if self.frames_processed else 0.0
        avg_processing_ms = (
            sum(self.processing_times) / len(self.processing_times) * 1000.0
            if self.processing_times else 0.0
        )
        max_processing_ms = max(self.processing_times) * 1000.0 if self.processing_times else 0.0
        rmse = math.sqrt(self.error_squared_sum / self.error_count) if self.error_count else 0.0
        avg_reacq = (
            sum(self.reacquisition_times) / len(self.reacquisition_times)
            if self.reacquisition_times else None
        )
        return PerformanceSummary(
            frames_processed=self.frames_processed,
            duration_s=duration,
            fps=fps,
            acquisition_time_s=self.acquisition_time_s,
            average_tracking_error_px=self.average_error,
            maximum_tracking_error_px=self.maximum_error,
            rmse_tracking_error_px=rmse,
            average_centroid_error_px=self.average_centroid_error,
            maximum_centroid_error_px=self.maximum_centroid_error,
            lock_retention_percentage=self._lock_retention(),
            target_loss_percentage=self._target_loss_rate(),
            target_loss_events=self.loss_events,
            successful_reacquisitions=self.successful_reacquisitions,
            average_reacquisition_time_s=avg_reacq,
            average_processing_time_ms=avg_processing_ms,
            max_processing_time_ms=max_processing_ms,
            ground_truth_available=True,
            source=source,
        )

    def _latest_benchmark_summary(self):
        """The latest COMPLETED benchmark result, or None.

        One accessor for the whole application: graphs, instant report and
        export report never keep their own copy of the numbers.
        """
        return getattr(self, "_last_benchmark_summary", None)

    def _save_summary_report(self, summary, output_dir, prefix):
        """Persist an explicit PerformanceSummary (report bookkeeping shared
        by the live-simulation and benchmark export paths)."""
        if summary is None:
            raise ValueError("no summary to report")
        paths = save_full_performance_report(summary, output_dir=output_dir, prefix=prefix)
        self.report_last_saved = paths
        self.last_pdf_path = Path(paths[0])
        self.open_last_report_button.setEnabled(True)
        return paths

    def _save_current_report_into(self, output_dir, prefix="performance"):
        return self._save_summary_report(self._make_summary(), output_dir, prefix)

    def _save_current_report(self, prefix="performance"):
        return self._save_current_report_into("reports", prefix=prefix)

    def _make_export_summary(self):
        """Summary for EXPORT REPORT: the latest completed benchmark result
        when one exists (with its runs' parameters), otherwise the live
        simulation's own measured values. Never a fabricated placeholder."""
        summary = self._latest_benchmark_summary()
        return summary if summary is not None else self._make_summary()

    def export_report(self):
        benchmark_summary = self._latest_benchmark_summary()
        prefix = "benchmark_export" if benchmark_summary is not None else "simulation_performance"
        title = "Benchmark Report" if benchmark_summary is not None else "Performance Report"
        try:
            paths = self._save_summary_report(
                self._make_export_summary(), "reports", prefix
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Report Error", str(exc))
            return
        self._show_report_dialog(title, paths)
        self.footer_status.setText(
            f"Report saved ({'latest benchmark' if benchmark_summary is not None else 'live simulation'}) "
            f"• {paths[0].name}"
        )
        self._register_report_artifact(paths[0])

    def _benchmark_snapshot(self, summary):
        """Point-in-time snapshot built from the latest completed benchmark.

        Every value comes straight from the recorded benchmark result and the
        live simulator settings — nothing is invented. Fields the benchmark
        cannot measure (ground-truth-dependent error terms without ground
        truth) stay None and are printed as “—” by the report writer.
        """
        metadata = dict(getattr(summary, "metadata", None) or {})
        video_path = metadata.get("video_path") or self._last_benchmark_video_path
        scenario = Path(video_path).name if video_path else "Benchmark Video"
        return {
            "simulation_time_s": float(summary.duration_s or 0.0),
            "frame": int(summary.frames_processed or 0),
            "target_id": self.designated_target_id,
            "target_signature": self.simulator.get_designated_target_signature(),
            "target_count": int(self.target_count),
            "state": str(metadata.get("state", "BENCHMARK")),
            "fps": float(summary.fps or 0.0),
            "update_rate_hz": float(config.UPDATE_RATE),
            "acquisition_time_s": summary.acquisition_time_s,
            "average_tracking_error_px": summary.average_tracking_error_px,
            "maximum_tracking_error_px": summary.maximum_tracking_error_px,
            "rmse_tracking_error_px": summary.rmse_tracking_error_px,
            "centroid_error_px": summary.average_centroid_error_px,
            "lock_retention_percentage": summary.lock_retention_percentage,
            "target_loss_percentage": summary.target_loss_percentage,
            "loss_events": int(summary.target_loss_events or 0),
            "reacquisition_count": int(summary.successful_reacquisitions or 0),
            "average_reacquisition_s": summary.average_reacquisition_time_s,
            "average_processing_time_ms": summary.average_processing_time_ms,
            "max_processing_time_ms": summary.max_processing_time_ms,
            "pan_deg": float(self.simulator.camera_pan),
            "tilt_deg": float(self.simulator.camera_tilt),
            "noise": self.noise_selector.currentText(),
            "noise_level": int(self.noise_level),
            "turbulence_enabled": bool(self.turbulence_check.isChecked()),
            "turbulence_strength_px": float(self.turbulence_strength),
            "camera_jitter_enabled": bool(self.jitter_check.isChecked()),
            "camera_jitter_px": float(self.camera_jitter_strength),
            "platform_motion_enabled": bool(self.platform_check.isChecked()),
            "platform_motion_px": float(self.platform_motion_strength),
            "atmosphere": self.atmosphere_selector.currentText(),
            "atmosphere_level": int(self.atmosphere_level),
            "motion_pattern": scenario,
            "scenario": scenario,
            "camera_resolution": f"{config.FRAME_WIDTH} x {config.FRAME_HEIGHT}",
            "fov": f"{config.FOV_HORIZONTAL:.1f} deg x {config.FOV_VERTICAL:.1f} deg",
            "link_ready": bool(self.link_readiness_is_ready()),
            "ground_truth_available": bool(summary.ground_truth_available),
            "source": str(summary.source),
        }

    def link_readiness_is_ready(self):
        """Current coarse-alignment readiness verdict (display/report only).

        Reads the same objectName the readiness badge itself is styled from,
        so the report can never disagree with what the operator sees.
        """
        label = getattr(self, "link_readiness_label", None)
        return bool(label is not None and label.objectName() == "link_ready")

    def generate_instant_report(self):
        if not self._authenticated:
            self.footer_status.setText(
                "Not authenticated • enter the access key to generate reports"
            )
            return

        # Resolve the data source BEFORE asking for a time: asking "which
        # moment?" when nothing has been recorded is the confusing, broken-
        # looking behaviour this method used to have.
        if not self.instant_report_history:
            self._instant_report_from_benchmark()
            return

        current_time = self.frames_processed / max(config.UPDATE_RATE, 1)
        entered, ok = QInputDialog.getText(
            self,
            "Point-in-Time Report",
            "Recorded time (e.g. 35s, 2m, 1:30). Leave blank for current moment:",
            QLineEdit.Normal,
            "",
        )
        if not ok:
            return

        try:
            requested_time = parse_report_time(entered) if entered.strip() else None
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Time", str(exc))
            return

        if requested_time is None:
            snapshot = self.instant_report_history[-1]
        else:
            if requested_time > current_time + (1.0 / max(config.UPDATE_RATE, 1)):
                QMessageBox.warning(
                    self,
                    "Time Not Reached",
                    f"The requested time is {requested_time:.2f} s, but the simulation has only recorded {current_time:.2f} s.",
                )
                return
            snapshot = min(
                self.instant_report_history,
                key=lambda item: abs(item["simulation_time_s"] - requested_time),
            )

        try:
            paths = save_instant_report(snapshot, output_dir="reports")
        except OSError as exc:
            QMessageBox.critical(self, "Instant Report Error", str(exc))
            return

        actual_time = snapshot["simulation_time_s"]
        self.footer_status.setText(f"Instant report saved • t={actual_time:.2f}s • {paths[0].name}")
        self.last_pdf_path = Path(paths[0])
        self.open_last_report_button.setEnabled(True)
        self._register_report_artifact(str(paths[0]))
        self._show_report_dialog("Point-in-Time Report Generated", paths)

    def _instant_report_from_benchmark(self):
        """INSTANT REPORT with no live telemetry recorded yet.

        Uses the latest completed benchmark result when one exists (real
        measured values); otherwise tells the user plainly what to do first
        instead of failing silently or crashing.
        """
        benchmark_summary = self._latest_benchmark_summary()
        if benchmark_summary is None:
            QMessageBox.warning(
                self,
                "No Benchmark Data",
                "No benchmark data available. Run Benchmark Video first.",
            )
            self.footer_status.setText("Instant report: no benchmark data available")
            return
        try:
            paths = save_instant_report(
                self._benchmark_snapshot(benchmark_summary), output_dir="reports"
            )
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Instant Report Error", str(exc))
            return
        self.last_pdf_path = Path(paths[0])
        self.open_last_report_button.setEnabled(True)
        self._register_report_artifact(str(paths[0]))
        self.footer_status.setText(
            f"Instant report saved from latest benchmark • {paths[0].name}"
        )
        self._show_report_dialog("Point-in-Time Report Generated", paths)

    def analyze_uploaded_video(self):
        """Upload an external video and run the SAME benchmark pipeline on it.

        Signature gate defaults to AUTO (external footage is not assumed to
        carry the simulator's synthetic target signature). Optional ground
        truth is pre-validated: an empty/invalid CSV is treated as unavailable
        rather than silently producing zero-error "perfection".
        """
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video To Analyze",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not video_path:
            return

        gt_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ground-Truth CSV (frame,x,y) — Cancel if none",
            str(Path(video_path).parent),
            "CSV Files (*.csv);;All Files (*)",
        )
        if gt_path:
            status = validate_ground_truth_file(gt_path)
            if status == "INVALID":
                QMessageBox.warning(
                    self,
                    "Ground Truth Invalid",
                    "The ground-truth CSV contains no usable frame,x,y rows.\n\n"
                    "It will be treated as NOT PROVIDED: GT-dependent metrics "
                    "(tracking error, centroid error, RMSE, Max Deviation) will "
                    "not be calculated rather than fabricated.",
                )
                gt_path = None  # engine sees no truth; status stays INVALID for the banner
        else:
            status = "NOT_PROVIDED"

        self.pause_simulation()
        self.analyze_video_button.setEnabled(False)
        self.footer_status.setText(
            "Analyzing uploaded video through the same detector/tracker pipeline…"
        )
        QApplication.processEvents()

        # Pre-check: fail fast with a clear message instead of a crashed worker.
        probe = cv2.VideoCapture(video_path)
        if not probe.isOpened():
            self.analyze_video_button.setEnabled(True)
            QMessageBox.critical(
                self, "Video Not Readable",
                "OpenCV could not open this file. The codec may be unsupported "
                "by this system's OpenCV build.",
            )
            self.footer_status.setText("Video analysis failed: unreadable video")
            return
        probe.release()

        self._video_analysis_worker = VideoBenchmarkWorker(
            video_path,
            ground_truth_path=(gt_path if status == "AVAILABLE" else None),
            preferred_signature=None,  # AUTO/neutral for external footage
            supplied_status=status,
        )
        self._video_analysis_worker.finished.connect(self._on_video_analysis_finished)
        self._video_analysis_worker.failed.connect(self._on_video_analysis_failed)
        self._video_analysis_worker.start()

    def _on_video_analysis_failed(self, message):
        self.analyze_video_button.setEnabled(True)
        self.footer_status.setText("Video analysis failed")
        QMessageBox.critical(self, "Video Analysis Error", message)

    def _on_video_analysis_finished(self, summary, paths, video_path, supplied_status):
        self.analyze_video_button.setEnabled(True)
        self.footer_status.setText(f"Video analysis complete • {Path(video_path).name}")
        # Uploaded-video analysis runs the SAME engine, so its result is also
        # a first-class benchmark result for the graphs and the reports.
        self._last_benchmark_summary = summary
        self._last_benchmark_paths = list(paths)
        self._last_benchmark_video_path = str(video_path)

        # Reconcile honestly: the engine may auto-detect a sibling
        # <stem>_groundtruth.csv even when the user cancelled the dialog.
        if summary.ground_truth_available:
            gt_status = "AVAILABLE"
        else:
            gt_status = supplied_status  # NOT_PROVIDED or INVALID

        # Honesty guard: the benchmark engine cannot know whether "zero" came
        # from real measurements or from an empty error list, so the GUI layer
        # nulls GT-dependent metrics whenever truth was absent/invalid.
        if not summary.ground_truth_available:
            summary.average_tracking_error_px = None
            summary.maximum_tracking_error_px = None
            summary.rmse_tracking_error_px = None
            summary.average_centroid_error_px = None
            summary.maximum_centroid_error_px = None

        pdf_path = Path(paths[4]) if len(paths) > 4 and paths[4] else None
        self.last_pdf_path = str(pdf_path) if pdf_path is not None else None
        self._video_analysis_window = VideoAnalysisWindow(
            summary, paths, video_path, gt_status=gt_status, parent=self
        )
        self._video_analysis_window.show()
        self._video_analysis_window.raise_()
        self._video_analysis_window.activateWindow()

    def _preferred_video_dir(self):
        """Folder the benchmark file dialog should open in.

        Remembers the last benchmark video (so repeat runs are one click),
        and otherwise opens where the bundled synthetic FSOC beacon clip
        lives so a working benchmark is always reachable.
        """
        last = getattr(self, "_last_benchmark_video_path", None)
        if last:
            parent = Path(last).parent
            if parent.exists():
                return str(parent)
        bundled = Path(__file__).resolve().parent
        return str(bundled if bundled.exists() else Path.cwd())

    @staticmethod
    def _video_is_readable(video_path):
        """True when OpenCV can actually decode the first frame.

        A missing file, an unsupported codec or a truncated video must give a
        clear message instead of a failed worker or an empty benchmark.
        """
        try:
            if not Path(video_path).exists():
                return False
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    return False
                ok, frame = capture.read()
                return bool(ok and frame is not None)
            finally:
                capture.release()
        except (OSError, ValueError, cv2.error):
            return False

    def _new_benchmark_screen(self):
        """Open a fresh benchmark screen, disposing of the previous one.

        Only one benchmark screen is ever live: the old instance stops its
        run, releases its OpenCV decoder and is then freed, so running the
        benchmark repeatedly cannot accumulate screens, open files, timers
        or worker threads.
        """
        previous = getattr(self, "_benchmark_screen", None)
        if previous is not None:
            try:
                # close() runs closeEvent() -> cleanup: stop the run and
                # release the decoder.
                previous.close()
                # Then actually free the widget tree (graph panels, scroll
                # areas, labels). Without this every benchmark screen ever
                # opened stayed alive for the lifetime of the application.
                previous.setParent(None)
                previous.deleteLater()
            except RuntimeError:
                pass  # already destroyed by Qt — nothing to release
        return BenchmarkScreen(self)

    def _reset_benchmark_graph_panel(self, screen=None):
        """Clear previous benchmark metrics before a new run starts.

        Clears BOTH graph panels that can show benchmark telemetry — the one in
        the TEST/OUTPUT section and the one inside the benchmark screen — so no
        curve from a previous run can survive into the next one (REPLAY starts
        from a genuinely empty state).
        """
        panels = []
        main_panel = getattr(self, "graph_panel", None)
        if main_panel is not None:
            panels.append(main_panel)
        target = screen if screen is not None else getattr(self, "_benchmark_screen", None)
        screen_panel = getattr(target, "graph_panel", None)
        if screen_panel is not None and screen_panel is not main_panel:
            panels.append(screen_panel)
        for panel in panels:
            try:
                panel.reset("Benchmark running — collecting performance data…")
            except Exception:
                pass  # a display reset must never stop the benchmark

    def benchmark_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Benchmark MP4",
            self._preferred_video_dir(),
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
        )
        if not video_path:
            self.footer_status.setText("Benchmark cancelled — no video selected")
            return

        self.pause_simulation()
        # Ground truth is only worth asking for once the video itself decodes;
        # an unreadable file is reported ON the benchmark screen (with RUN
        # BENCHMARK still available to retry) instead of leaving the user with
        # a closed screen and a dismissal-only dialog.
        gt_path = ""
        if self._video_is_readable(video_path):
            gt_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Ground-Truth CSV (Cancel if none)",
                str(Path(video_path).parent),
                "CSV Files (*.csv);;All Files (*)",
            )
        self._prepare_benchmark_screen(video_path, gt_path or None)

    def _prepare_benchmark_screen(self, video_path, ground_truth_path=None):
        """Open the benchmark screen with a video loaded and PLAY armed.

        No processing starts here: the transport (PLAY) is what runs the ONE
        engine loop, so the user always sees the video and the analysis start
        together.
        """
        self._last_benchmark_video_path = video_path
        self._pending_benchmark_ground_truth = ground_truth_path

        screen = self._new_benchmark_screen()
        self._benchmark_screen = screen
        screen.show()
        screen.raise_()
        screen.activateWindow()
        # A brand-new screen starts with no stale curve from any previous run.
        self._reset_benchmark_graph_panel(screen)

        if not self._video_is_readable(video_path):
            screen.info_value.setText(
                f"{Path(video_path).name}\n"
                "This file could not be opened as a video."
            )
            screen.set_load_error(
                "Benchmark video could not be loaded. "
                "Press RUN BENCHMARK to choose another file."
            )
            self.footer_status.setText("Benchmark video could not be loaded")
            return screen

        screen.set_video_preview(video_path)
        screen.load_benchmark_source(video_path, self._video_frame_count(video_path))
        screen.info_value.setText(
            f"{Path(video_path).name}\n"
            f"Ground truth: {'provided' if ground_truth_path else 'NOT PROVIDED — error metrics shown as —'}\n"
            f"Pipeline: benchmark_video.run_video_benchmark (unchanged)"
        )
        screen.set_progress(0, "Ready — press PLAY to start the benchmark video.")
        screen.run_button.setEnabled(True)
        screen.cancel_button.setEnabled(False)
        self.footer_status.setText(
            "Benchmark video loaded — press PLAY to run video + analysis together"
        )
        return screen

    @staticmethod
    def _video_frame_count(video_path):
        """Frame count of a video, or 0 when it cannot be read."""
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return 0
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except (cv2.error, ValueError):
            return 0
        finally:
            capture.release()

    def _start_benchmark_screen_run(self, screen=None):
        """Start the ONE benchmark processing loop (PLAY / REPLAY).

        Guarded so a double click, a stray signal or a second PLAY can never
        create a second analysis loop: if a worker is still running this is a
        no-op.
        """
        screen = screen if screen is not None else getattr(self, "_benchmark_screen", None)
        if screen is None:
            return
        worker = getattr(self, "_benchmark_worker", None)
        if worker is not None and worker.isRunning():
            return  # only ONE active benchmark loop, ever
        if getattr(screen, "_playback_state", "idle") in ("running", "paused"):
            return  # the transport already owns a live run

        video_path = getattr(screen, "_source_path", None) or self._last_benchmark_video_path
        if not video_path or not Path(video_path).exists():
            screen.set_load_error("Benchmark video could not be loaded.")
            self.footer_status.setText("Benchmark video not found")
            return
        if not self._video_is_readable(video_path):
            screen.set_load_error("Benchmark video could not be loaded.")
            self.footer_status.setText("Benchmark video could not be loaded")
            return

        # A fresh run (including REPLAY) starts from an empty state: no curves,
        # no results and no counters from the previous run survive.
        self._reset_benchmark_graph_panel(screen)
        for label in screen.result_labels.values():
            label.setText("—")
        screen.video_title.setText("TEST VIEW — LIVE ANALYSIS")
        screen.set_progress(0, "Starting standardized benchmark…")
        screen.run_button.setEnabled(True)
        screen.cancel_button.setEnabled(True)
        screen.reset_playback_position()
        screen._set_playback_state(screen.PLAYBACK_RUNNING)
        self.footer_status.setText(
            "Benchmark running on the same detector/tracker pipeline…"
        )

        self._benchmark_worker = VideoBenchmarkWorker(
            video_path,
            ground_truth_path=getattr(self, "_pending_benchmark_ground_truth", None),
            # AUTO/neutral for external footage: the uploaded video is not
            # assumed to carry the live simulator's synthetic signature (same
            # choice as analyze_uploaded_video). An external beacon of any
            # colour is still detectable; the signature gate would otherwise
            # honestly report no detection for a mismatched video.
            preferred_signature=None,
            supplied_status="NOT_PROVIDED",
            live=True,
        )
        self._benchmark_worker.finished.connect(self._on_benchmark_screen_finished)
        self._benchmark_worker.failed.connect(self._on_benchmark_screen_failed)
        self._benchmark_worker.progress.connect(self._on_benchmark_screen_progress)
        self._benchmark_worker.frame_ready.connect(screen._on_live_frame)
        self._benchmark_worker.start()

    def _request_benchmark_pause(self):
        """PAUSE: freeze the video, the analysis and every metric."""
        worker = getattr(self, "_benchmark_worker", None)
        if worker is None or not worker.isRunning():
            return
        worker.request_pause()
        screen = getattr(self, "_benchmark_screen", None)
        if screen is not None:
            screen.set_progress(screen.progress_bar.value(),
                                "Paused — video and analysis are frozen.")
        self.footer_status.setText("Benchmark paused")

    def _request_benchmark_resume(self):
        """RESUME: continue from the same frame, same metrics. No restart."""
        worker = getattr(self, "_benchmark_worker", None)
        if worker is None or not worker.isRunning():
            return
        worker.request_resume()
        screen = getattr(self, "_benchmark_screen", None)
        if screen is not None:
            screen.set_progress(screen.progress_bar.value(),
                                "Resumed — continuing from the same frame.")
        self.footer_status.setText("Benchmark running on the same detector/tracker pipeline…")

    def _on_benchmark_screen_progress(self, percent, text):
        """Live per-frame progress from the benchmark worker thread."""
        screen = getattr(self, "_benchmark_screen", None)
        if screen is not None:
            screen.set_progress(percent, text)

    def _run_benchmark_with_video(self, video_path):
        """Load a specific video and start its benchmark run immediately."""
        if not video_path or not Path(video_path).exists():
            self.footer_status.setText("Benchmark video file not found")
            QMessageBox.warning(
                self, "Video Not Found",
                f"The benchmark video is no longer available:\n{video_path}\n\n"
                "Use BENCHMARK VIDEO to pick another file.",
            )
            return

        self.pause_simulation()
        previous = getattr(self, "_benchmark_worker", None)
        if previous is not None and previous.isRunning():
            return  # one benchmark loop at a time

        # For re-run we do not ask for ground truth again - the engine still
        # auto-detects the sibling <stem>_groundtruth.csv when it exists.
        screen = self._prepare_benchmark_screen(video_path, None)
        self._start_benchmark_screen_run(screen)

    def _on_benchmark_screen_cancel(self):
        """User asked to cancel: stop the worker cooperatively at the next frame.

        ``cancel()`` also un-parks a PAUSED run, so a paused benchmark can
        always be stopped without resuming it first.
        """
        screen = getattr(self, "_benchmark_screen", None)
        worker = getattr(self, "_benchmark_worker", None)
        if worker is not None and worker.isRunning():
            worker.cancel()
            if screen is not None:
                screen.set_progress(
                    screen.progress_bar.value(), "Cancelling…")
                screen.cancel_button.setEnabled(False)

    def _on_benchmark_screen_failed(self, message):
        if self.sender() is not None and self.sender() is not getattr(self, "_benchmark_worker", None):
            return  # a superseded run must never touch the live screen
        screen = getattr(self, "_benchmark_screen", None)
        if screen is not None:
            screen.run_button.setEnabled(True)
            screen.cancel_button.setEnabled(False)
            screen.set_progress(0, f"Failed: {message}")
            screen.mark_run_stopped()
        self.footer_status.setText("Benchmark failed")
        QMessageBox.critical(self, "Benchmark Error", message)

    def _on_benchmark_screen_finished(self, summary, paths, video_path, supplied_status):
        if self.sender() is not None and self.sender() is not getattr(self, "_benchmark_worker", None):
            return  # a superseded run must never overwrite the live results
        screen = getattr(self, "_benchmark_screen", None)
        if screen is None:
            return
        screen.run_button.setEnabled(True)
        screen.cancel_button.setEnabled(False)
        if dict(getattr(summary, "metadata", None) or {}).get("cancelled"):
            # A stopped run is a stopped run: show honest state, never results.
            screen.set_progress(0, "Cancelled by user — partial artifacts kept on disk.")
            screen.video_title.setText("TEST VIEW — CANCELLED")
            screen.mark_run_stopped()
            self.footer_status.setText("Benchmark cancelled")
            return
        screen.set_progress(100, "Complete.")
        # SINGLE SOURCE OF TRUTH: the dashboard, the graph panel, EXPORT REPORT
        # and INSTANT REPORT all read this one result object. A new run simply
        # replaces it, so no stale value from a previous benchmark can leak
        # into the graphs or the reports.
        self._last_benchmark_summary = summary
        self._last_benchmark_paths = list(paths)
        self._last_benchmark_video_path = str(video_path)
        # Honesty guard (same as uploaded-video analysis): the engine cannot
        # distinguish "measured zero" from "no ground truth", so GT-dependent
        # metrics are nulled when truth was absent — shown as —, never fake.
        if not summary.ground_truth_available:
            summary.average_tracking_error_px = None
            summary.maximum_tracking_error_px = None
            summary.rmse_tracking_error_px = None
        screen.set_results(summary)
        # Video and analysis reached the end of the clip: the transport offers
        # REPLAY (a genuinely fresh run), which is the next sensible action.
        screen.video_title.setText("TEST VIEW — BENCHMARK COMPLETE")
        screen.mark_run_finished()
        # Fallback only: if nothing was ever painted live, put the finished
        # tracked-overlay frame on screen. In the normal live path the last
        # analysed frame (same overlays) is already the correct final image.
        tracked_video_path = dict(getattr(summary, "metadata", None) or {}).get("output_video")
        if getattr(screen, "_live_frame", None) is None and tracked_video_path \
                and Path(tracked_video_path).exists():
            screen.show_tracked_video(tracked_video_path)
        self.footer_status.setText(f"Benchmark complete • {Path(video_path).name}")
        self.report_last_saved = paths
        # paths follows the benchmark engine's artifact order (json, csv, …, pdf)
        pdf_path = Path(paths[4]) if len(paths) > 4 and paths[4] else None
        if pdf_path is not None and pdf_path.exists():
            self.last_pdf_path = pdf_path
            self.open_last_report_button.setEnabled(True)

        # Push new benchmark telemetry into the TEST/OUTPUT graph panel
        # (the live timer will then keep it updating every second)
        self._load_correlation_telemetry_for_panel(summary)

    def _load_correlation_telemetry_for_panel(self, summary=None):
        """Load the freshly produced benchmark telemetry into the graph panel.

        The EXACT telemetry CSV this run wrote is used first (one dict lookup);
        only when the summary does not name one does it fall back to scanning
        the reports folder.  The panel's built-in 1-second timer then keeps
        the display live.
        """
        if getattr(self, "graph_panel", None) is None:
            return
        try:
            target_path = None
            if summary is not None:
                named = dict(getattr(summary, "metadata", None) or {}).get("telemetry_csv")
                if named and Path(named).exists():
                    target_path = str(named)
            if target_path is None:
                telemetry_paths = sorted(
                    Path("reports").glob("*_tracked_telemetry_*.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not telemetry_paths:
                    self.graph_panel.status_label.setText(
                        "Run Benchmark Video to generate performance data."
                    )
                    return
                target_path = str(telemetry_paths[0])
            loaded = self.graph_panel.load_telemetry(target_path)
            if loaded:
                self.graph_panel.status_label.setText(
                    f"Loaded {len(self.graph_panel.timestamps)} frames from {Path(target_path).name}"
                )
            else:
                self.graph_panel.status_label.setText(
                    f"Failed to load telemetry from {Path(target_path).name}"
                )
        except Exception as exc:
            self.graph_panel.status_label.setText(f"Error: {exc}")

    def _open_benchmark_graphs(self):
        """Open the benchmark performance graph panel in the TEST/OUTPUT section.

        Loads the latest benchmark telemetry CSV (if any) and displays the five
        time-series graphs: tracking error, target vs Kalman position, pan/tilt
        control response, FPS/performance, and tracking confidence.
        """
        # Prefer the most recent benchmark telemetry CSV from reports
        telemetry_paths = sorted(
            Path("reports").glob("*_tracked_telemetry_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        target_path = None
        if telemetry_paths:
            target_path = str(telemetry_paths[0])
        elif self._last_benchmark_video_path and Path(self._last_benchmark_video_path).exists():
            target_path = self._last_benchmark_video_path

        if target_path is None:
            QMessageBox.information(
                self,
                "No Telelemetry",
                "No benchmark telemetry CSV found. Run a benchmark first, then open these graphs.",
            )
            return

        try:
            loaded = self.graph_panel.load_telemetry(target_path)
            if loaded:
                self.graph_panel.status_label.setText(
                    f"Loaded {len(self.graph_panel.timestamps)} frames from {Path(target_path).name}"
                )
            else:
                self.graph_panel.status_label.setText(
                    f"Failed to load telemetry from {Path(target_path).name}"
                )
        except Exception as exc:
            self.graph_panel.status_label.setText(f"Error: {exc}")
            QMessageBox.critical(
                self, "Graph Panel Error", f"Failed to load telemetry: {exc}"
            )


class BenchmarkScreen(QDialog):
    """Dedicated STANDARDIZED PERFORMANCE VALIDATION screen.

    Hierarchy per the final-polish spec: (1) large video/test view,
    (2) test information, (3) processing progress, (4) final performance
    results in large readable type. Runs the EXISTING benchmark engine on a
    worker thread — no new calculation path, no simulated values.

    The transport drives that ONE engine loop: PLAY starts the video and the
    analysis together, PAUSE parks both, RESUME continues from the same frame,
    and REPLAY starts a completely fresh run.
    """

    PLAYBACK_IDLE = "idle"
    PLAYBACK_RUNNING = "running"
    PLAYBACK_PAUSED = "paused"
    PLAYBACK_DONE = "done"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ASTRALINK — BENCHMARK VIDEO | STANDARDIZED VALIDATION")
        self.setModal(False)
        self.resize(1240, 820)
        self.setMinimumSize(980, 680)

        central = QWidget()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(14)

        # ---- 1. LARGE VIDEO / TEST VIEW (dominant) -------------------------
        video_panel = QFrame()
        video_panel.setObjectName("panel")
        self.video_panel = video_panel  # Store reference for video player controls
        vp = QVBoxLayout(video_panel)
        vp.setContentsMargins(12, 10, 12, 10)
        vp.setSpacing(6)
        self.video_title = QLabel("TEST VIEW")
        self.video_title.setObjectName("panel_title")
        vp.addWidget(self.video_title)
        self.video_view = QLabel("Select a benchmark video to begin")
        self.video_view.setObjectName("camera_view")
        self.video_view.setAlignment(Qt.AlignCenter)
        self.video_view.setMinimumSize(560, 420)
        self.video_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vp.addWidget(self.video_view, 1)
        root.addWidget(video_panel, 3)

        # ---- right column: info / progress / results -----------------------
        side = QVBoxLayout()
        side.setSpacing(10)

        # 2. TEST INFORMATION
        info_panel = QFrame()
        info_panel.setObjectName("panel")
        ip = QVBoxLayout(info_panel)
        ip.setContentsMargins(10, 8, 10, 8)
        ip.setSpacing(4)
        info_title = QLabel("TEST INFORMATION")
        info_title.setObjectName("panel_title")
        ip.addWidget(info_title)
        self.info_value = QLabel("No test loaded.")
        self.info_value.setObjectName("caption")
        self.info_value.setWordWrap(True)
        ip.addWidget(self.info_value)
        side.addWidget(info_panel)

        # 3. PROCESSING PROGRESS
        prog_panel = QFrame()
        prog_panel.setObjectName("panel")
        pp = QVBoxLayout(prog_panel)
        pp.setContentsMargins(10, 8, 10, 8)
        pp.setSpacing(4)
        prog_title = QLabel("PROCESSING PROGRESS")
        prog_title.setObjectName("panel_title")
        pp.addWidget(prog_title)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        pp.addWidget(self.progress_bar)
        self.progress_label = QLabel("Idle.")
        self.progress_label.setObjectName("caption")
        pp.addWidget(self.progress_label)
        side.addWidget(prog_panel)

        # 4. FINAL PERFORMANCE RESULTS (large, readable)
        res_panel = QFrame()
        res_panel.setObjectName("panel")
        rp = QVBoxLayout(res_panel)
        rp.setContentsMargins(10, 8, 10, 8)
        rp.setSpacing(6)
        res_title = QLabel("FINAL PERFORMANCE RESULTS")
        res_title.setObjectName("panel_title")
        rp.addWidget(res_title)
        self.result_labels = {}
        for name in ("FPS", "RMSE", "MAX ERROR", "TARGET LOSS",
                     "ACQUISITION", "RE-ACQUISITION"):
            row = QHBoxLayout()
            lab = QLabel(name)
            lab.setObjectName("section_title")
            val = QLabel("—")
            val.setObjectName("metric_value_primary")
            row.addWidget(lab)
            row.addStretch()
            row.addWidget(val)
            rp.addLayout(row)
            self.result_labels[name] = val
        rp.addStretch()
        side.addWidget(res_panel, 1)

        # 5. BENCHMARK PERFORMANCE GRAPHS (time-series visualizations)
        from benchmark_graphs import BenchmarkGraphPanel
        self.graph_panel = BenchmarkGraphPanel()
        self.graph_panel.setMinimumHeight(250)
        self.graph_panel.setSizePolicy(
            __import__('PySide6.QtWidgets', fromlist=['QSizePolicy']).QSizePolicy.Expanding,
            __import__('PySide6.QtWidgets', fromlist=['QSizePolicy']).QSizePolicy.Preferred
        )
        side.addWidget(self.graph_panel, 1)

        # actions
        actions = QHBoxLayout()
        self.run_button = QPushButton("▶  RUN BENCHMARK")
        self.run_button.setObjectName("start_button")
        self.run_button.clicked.connect(self._request_run_benchmark)
        self.cancel_button = QPushButton("■  CANCEL")
        self.cancel_button.setObjectName("reset_button")
        self.cancel_button.setEnabled(False)  # enabled only while a run is live
        self.cancel_button.clicked.connect(self._request_cancel)
        self.close_button = QPushButton("CLOSE")
        self.close_button.setObjectName("close_button")
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.run_button, 1)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.close_button)
        side.addLayout(actions)
        root.addLayout(side, 2)

        # Setup video player controls (initially disabled until video loaded)
        self._setup_video_player()

    def _request_cancel(self):
        """Ask the parent window to cancel the running benchmark worker."""
        self._parent_call("_on_benchmark_screen_cancel")
        self.mark_run_stopped()

    def _request_run_benchmark(self):
        """Start (or REPLAY) the benchmark on the screen's own video file.

        The owning window owns the worker, so this is a single delegated call:
        it starts ONE run, or does nothing at all when a run is already live.
        """
        parent = self.parent()
        if parent is None:
            return
        if hasattr(parent, "_start_benchmark_screen_run"):
            parent._start_benchmark_screen_run(self)
            return
        if getattr(parent, "_last_benchmark_video_path", None):
            parent._run_benchmark_with_video(parent._last_benchmark_video_path)
        elif hasattr(parent, "benchmark_video"):
            parent.benchmark_video()

    # ---- display-only helpers -------------------------------------------
    def set_video_preview(self, path):
        """Show the first frame of the input video as a preview."""
        from PySide6.QtGui import QImage
        import cv2
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                       rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.video_view.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return True

    def show_tracked_video(self, tracked_video_path):
        """Put the finished tracked-overlay frame on screen.

        The tracked MP4 is a benchmark ARTEFACT: it is written to disk and
        listed with the other results. Live playback already displayed every
        analysed frame, so this only paints the first overlay frame — it never
        opens a second playback loop or a second decoder.
        """
        capture = cv2.VideoCapture(str(tracked_video_path))
        if not capture.isOpened():
            return False
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                       rgb.strides[0], QImage.Format.Format_RGB888).copy()
        if image.isNull():
            return False
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return False
        self.video_view.setPixmap(pixmap.scaled(
            self.video_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return True

    def set_results(self, summary):
        if summary is None:
            return
        self.result_labels["FPS"].setText(_fmt_report_value(summary.fps))
        self.result_labels["RMSE"].setText(
            f"{_fmt_report_value(summary.rmse_tracking_error_px)} px")
        self.result_labels["MAX ERROR"].setText(
            f"{_fmt_report_value(summary.maximum_tracking_error_px)} px")
        self.result_labels["TARGET LOSS"].setText(
            f"{_fmt_report_value(summary.target_loss_percentage)} %")
        self.result_labels["ACQUISITION"].setText(
            f"{_fmt_report_value(summary.acquisition_time_s)} s")
        self.result_labels["RE-ACQUISITION"].setText(
            f"{_fmt_report_value(summary.average_reacquisition_time_s)} s")

        # Load telemetry data into graph panel if available. The telemetry CSV
        # this run actually wrote is used first (no folder scan, no stale file).
        metadata = dict(getattr(summary, "metadata", None) or {})
        telemetry_path = metadata.get("telemetry_csv")
        try:
            if telemetry_path and Path(telemetry_path).exists():
                self.graph_panel.load_telemetry(telemetry_path)
            else:
                telemetry_files = sorted(
                    glob.glob("reports/*_tracked_telemetry_*.csv"), reverse=True
                )
                if telemetry_files:
                    self.graph_panel.load_telemetry(telemetry_files[0])
                else:
                    self.graph_panel.status_label.setText(
                        "Run Benchmark Video to generate performance data."
                    )
        except (OSError, ValueError) as exc:
            self.graph_panel.status_label.setText(f"Graph load error: {exc}")

    def set_progress(self, percent, label):
        self.progress_bar.setValue(int(max(0, min(100, percent))))
        self.progress_label.setText(str(label))

    # ---- video playback controls ---------------------------------------
    def _setup_video_player(self):
        """Add video playback controls to the video panel."""
        # Add playback controls below the video view
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(6)
        
        self.play_pause_button = QPushButton("▶  PLAY")
        self.play_pause_button.setObjectName("start_button")
        self.play_pause_button.setMinimumWidth(110)
        self.play_pause_button.setToolTip(
            "PLAY: the benchmark video plays while every frame is detected,\n"
            "tracked and measured. PAUSE freezes the run, RESUME continues\n"
            "from the same frame, REPLAY starts a completely fresh run."
        )
        self.play_pause_button.clicked.connect(self._toggle_video_playback)
        controls_layout.addWidget(self.play_pause_button)
        
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(100)
        self.video_slider.setValue(0)
        self.video_slider.setToolTip("Progress of the current benchmark run")
        self.video_slider.setEnabled(False)   # progress read-out only
        controls_layout.addWidget(self.video_slider, 1)
        
        self.current_frame_label = QLabel("0 / 0")
        self.current_frame_label.setObjectName("caption")
        controls_layout.addWidget(self.current_frame_label)
        
        vp = self.video_panel.layout()
        # Controls sit UNDER the test view; the view keeps the stretch.
        vp.addLayout(controls_layout)

        # Transport state. NOTE: there is deliberately no QTimer here. The
        # benchmark engine itself paces playback and emits every analysed
        # frame, so this screen can never own a second animation loop, a
        # duplicate video listener or a leaked timer.
        self._video_current_frame = 0
        self._video_playing = False
        self._live_frame = None
        self._source_path = None
        self._source_frame_count = 0
        self._playback_state = self.PLAYBACK_IDLE
        self._set_playback_state(self.PLAYBACK_IDLE)

    # ---- transport ------------------------------------------------------
    def _set_playback_state(self, state, enabled=None):
        """Single owner of the PLAY / PAUSE / RESUME / REPLAY label.

        ``enabled`` defaults to "is a video loaded?"; the caller passes False
        for the brief starting window so a double click cannot queue two runs.
        """
        self._playback_state = state
        if state == self.PLAYBACK_RUNNING:
            text, name = "⏸  PAUSE", "pause_button"
        elif state == self.PLAYBACK_PAUSED:
            text, name = "▶  RESUME", "start_button"
        elif state == self.PLAYBACK_DONE:
            text, name = "▶  REPLAY", "start_button"
        else:
            text, name = "▶  PLAY", "start_button"
        self.play_pause_button.setText(text)
        self.play_pause_button.setObjectName(name)
        # Re-polish so the stylesheet rule for the new object name applies.
        try:
            style = self.play_pause_button.style()
            style.unpolish(self.play_pause_button)
            style.polish(self.play_pause_button)
        except (AttributeError, RuntimeError):
            pass  # styling is cosmetic — never let it break the transport
        self._video_playing = state == self.PLAYBACK_RUNNING
        if enabled is None:
            enabled = self._source_path is not None
        self.play_pause_button.setEnabled(bool(enabled))

    def _parent_call(self, name, *args):
        """Call a method on the owning window if it provides one."""
        method = getattr(self.parent(), name, None)
        if method is None:
            return None
        return method(*args)

    def _toggle_video_playback(self):
        """PLAY → PAUSE → RESUME → REPLAY on ONE processing loop.

        Every branch drives the *benchmark* (the single engine loop), never a
        separate video timer, so the picture and the analysis are always the
        same frames.
        """
        state = self._playback_state
        if state == self.PLAYBACK_RUNNING:
            self._parent_call("_request_benchmark_pause")
            self._set_playback_state(self.PLAYBACK_PAUSED)
        elif state == self.PLAYBACK_PAUSED:
            self._parent_call("_request_benchmark_resume")
            self._set_playback_state(self.PLAYBACK_RUNNING)
        else:  # IDLE or DONE → start a completely fresh run
            self._request_run_benchmark()

    # ---- live frame + lifecycle ----------------------------------------
    def load_benchmark_source(self, video_path, frame_count=0):
        """Register the video the transport will drive.

        Called when the screen opens: the transport becomes usable (PLAY
        enabled) and no processing starts until the user presses PLAY.
        """
        self._source_path = str(video_path) if video_path else None
        self._source_frame_count = int(frame_count or 0)
        self._video_current_frame = 0
        self.video_slider.setValue(0)
        self.current_frame_label.setText(f"0 / {self._source_frame_count}")
        self._set_playback_state(self.PLAYBACK_IDLE)

    def _on_live_frame(self, overlay_frame, live):
        """Show the frame that was JUST analysed and plot its measurements.

        Both come from the same engine iteration, so the graph point always
        belongs to the frame on screen — no interpolation, no invented data.
        """
        try:
            if overlay_frame is None:
                return
            self._live_frame = overlay_frame
            rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
            image = QImage(
                rgb.data, rgb.shape[1], rgb.shape[0],
                rgb.strides[0], QImage.Format.Format_RGB888,
            ).copy()
            if image.isNull():
                return
            pixmap = QPixmap.fromImage(image)
            if pixmap.isNull():
                return
            self.video_view.setPixmap(pixmap.scaled(
                self.video_view.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            ))

            frame_no = int(live.get("frame") or 0)
            self._video_current_frame = frame_no
            total = self._source_frame_count or frame_no
            self.current_frame_label.setText(f"{frame_no} / {total}")
            if total > 0:
                self.video_slider.setValue(int(100.0 * frame_no / total))

            # Live measurements, straight from the engine's own iteration.
            running_fps = live.get("fps")
            if isinstance(running_fps, (int, float)) and math.isfinite(running_fps):
                self.result_labels["FPS"].setText(f"{running_fps:.1f}")
            self.graph_panel.append_live_sample(
                live.get("timestamp_s"), frame_no, live
            )
        except (cv2.error, KeyError, TypeError, ValueError, RuntimeError):
            pass  # a display hiccup must never stop the benchmark

    def mark_run_finished(self):
        """Run reached the end of the video: the transport becomes REPLAY."""
        total = self._source_frame_count or self._video_current_frame
        self.video_slider.setValue(100)
        self._video_current_frame = total
        self.current_frame_label.setText(f"{total} / {total}")
        self._set_playback_state(self.PLAYBACK_DONE)

    def reset_playback_position(self):
        """Return the position read-out to the start of the clip (REPLAY)."""
        self._video_current_frame = 0
        self.video_slider.setValue(0)
        total = self._source_frame_count
        self.current_frame_label.setText(f"0 / {total}")

    def mark_run_stopped(self):
        """Run stopped early (cancel/failure): back to a plain PLAY state."""
        self._set_playback_state(self.PLAYBACK_IDLE)

    def set_load_error(self, message):
        """Clear, non-fatal message when a benchmark video cannot be loaded.

        Never raises, never closes the screen: the RUN BENCHMARK button stays
        available so the user can retry with another file.
        """
        self.video_view.setText(f"⚠  {message}")
        self.video_title.setText("TEST VIEW — LOAD ERROR")
        self.progress_label.setText(message)
        self._source_path = None
        self._source_frame_count = 0
        self._set_playback_state(self.PLAYBACK_IDLE, enabled=False)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def cleanup_video_player(self):
        """Stop this screen's run and release every resource it owns.

        The owning window holds the worker, so it is cancelled here and joined
        (cancel() un-parks a PAUSED run, so the join returns within one frame).
        Without this, a closed benchmark screen could leave its analysis loop
        running and its decoder open for the lifetime of the application.
        """
        parent = self.parent()
        worker = getattr(parent, "_benchmark_worker", None)
        if worker is not None and getattr(parent, "_benchmark_screen", None) is self:
            try:
                worker.cancel()
                if worker.isRunning():
                    worker.wait(5000)
            except RuntimeError:
                pass  # already destroyed by Qt — nothing to join

    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        """Stop the benchmark loop and release the decoder on close.

        Without this, every closed benchmark screen kept its OpenCV capture
        and its analysis loop alive for the lifetime of the application.
        """
        try:
            self.cleanup_video_player()
        except Exception:
            pass  # closing must never raise
        super().closeEvent(event)


class HandoffWindowDialog(QDialog):
    """Live HANDOFF WINDOW readout.

    Shows the continuous span during which every coarse-alignment handoff
    condition holds, using ONLY real values passed in from the running
    mission (center error, max tracking error, target loss, re-acquisition).
    Pure telemetry view — it cannot influence tracking, the readiness gate,
    or any mission state.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ASTRALINK — HANDOFF WINDOW")
        self.setModal(False)
        self.setMinimumSize(360, 300)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        self.status_label = QLabel("HANDOFF: NOT READY")
        self.status_label.setObjectName("link_not_ready")
        layout.addWidget(self.status_label)

        self.window_label = QLabel("WINDOW: CLOSED — conditions not satisfied")
        self.window_label.setObjectName("caption")
        self.window_label.setWordWrap(True)
        layout.addWidget(self.window_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        self.value_labels = {}
        rows = [
            ("FIRST QUALIFIED", "first_qualified_s", lambda v: f"{v:.2f} s" if v is not None else "—"),
            ("STABLE DURATION", "stable_duration_s", lambda v: f"{v:.2f} s"),
            ("STABLE FRAMES", "stable_frames",
             lambda v: f"{v}/{self._required_frames}" if v is not None else "—"),
            ("CENTER ERROR", "center_error_px", lambda v: f"{v:.2f} px" if v is not None else "—"),
            ("MAX TRACKING ERROR", "max_tracking_error_px", lambda v: f"{v:.2f} px" if v is not None else "—"),
            ("TARGET LOSS", "loss_percent", lambda v: f"{v:.2f} %"),
            ("LAST RE-ACQUISITION", "reacquisition_s", lambda v: f"{v:.2f} s" if v is not None else "—"),
        ]
        for index, (name, key, fmt) in enumerate(rows):
            title = QLabel(name)
            title.setObjectName("small_label")
            value = QLabel("—")
            value.setObjectName("metric_value")
            grid.addWidget(title, index, 0)
            grid.addWidget(value, index, 1)
            self.value_labels[key] = (value, fmt)
        layout.addLayout(grid)

        note = QLabel(
            "Window opens when every handoff condition holds and closes as soon "
            "as any condition fails. Values come from the live mission."
        )
        note.setObjectName("caption")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()

        self._required_frames = 0

    def update_values(self, data):
        """Refresh the readout from a live handoff-window snapshot."""
        self._required_frames = data.get("stable_frames_required", 0)
        for key, (label, fmt) in self.value_labels.items():
            label.setText(fmt(data.get(key)))
        if data.get("ready"):
            self.status_label.setText("HANDOFF: READY")
            self.status_label.setObjectName("link_ready")
        elif data.get("open"):
            self.status_label.setText("HANDOFF: NOT READY — qualifying")
            self.status_label.setObjectName("link_recover")
        else:
            self.status_label.setText("HANDOFF: NOT READY")
            self.status_label.setObjectName("link_not_ready")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        if data.get("open"):
            first_s = data.get("first_qualified_s")
            duration = data.get("stable_duration_s", 0.0)
            self.window_label.setText(
                f"WINDOW: OPEN — qualified at {first_s:.2f} s, stable for {duration:.2f} s"
            )
        else:
            self.window_label.setText("WINDOW: CLOSED — conditions not satisfied")


class WhyLockedDialog(QDialog):
    """WHY LOCKED? — readable explanation of the current handoff verdict.

    Every line is a real gate condition with its real value. Passing rows get
    a ✓, failing rows a ✕ with the limit, not-yet-measurable rows a neutral
    “—” (never a fake check). Display-only: it cannot influence the mission.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ASTRALINK — WHY LOCKED?")
        self.setModal(False)
        self.setMinimumSize(420, 360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self.title_label = QLabel("WHY LOCKED?")
        self.title_label.setObjectName("panel_title")
        layout.addWidget(self.title_label)

        self.verdict_label = QLabel("HANDOFF → NOT READY")
        self.verdict_label.setObjectName("link_not_ready")
        layout.addWidget(self.verdict_label)

        self.rows_layout = QGridLayout()
        self.rows_layout.setHorizontalSpacing(14)
        self.rows_layout.setVerticalSpacing(5)
        layout.addLayout(self.rows_layout)
        self._row_widgets = []

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("caption")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        layout.addStretch()

    def update_evidence(self, evidence):
        """Rebuild the condition rows from a live evidence snapshot."""
        for mark, label, value in self._row_widgets:
            label.deleteLater()
            value.deleteLater()
            mark.deleteLater()
        self._row_widgets = []

        conditions = evidence.get("conditions", [])
        for index, (name, value_text, passed) in enumerate(conditions):
            if passed is True:
                mark_text, mark_color = "✓", "#43f08f"
            elif passed is False:
                mark_text, mark_color = "✕", "#ff8a92"
            else:
                mark_text, mark_color = "—", "#8294a5"
            mark = QLabel(mark_text)
            mark.setStyleSheet(f"font-size: 12px; font-weight: 800; color: {mark_color};")
            name_label = QLabel(name)
            name_label.setObjectName("small_label")
            value_label = QLabel(value_text)
            value_label.setObjectName("metric_value")
            self.rows_layout.addWidget(mark, index, 0)
            self.rows_layout.addWidget(name_label, index, 1)
            self.rows_layout.addWidget(value_label, index, 2)
            self._row_widgets.append((mark, name_label, value_label))

        if evidence.get("ready"):
            self.title_label.setText("WHY LOCKED?")
            self.verdict_label.setText("HANDOFF CONDITIONS MET  —  COARSE ALIGNMENT → READY")
            self.verdict_label.setObjectName("link_ready")
            self.summary_label.setText(
                "All coarse-alignment handoff conditions hold with the values shown. "
                "Every value is measured from the live mission."
            )
        else:
            self.title_label.setText("WHY NOT LOCKED?")
            self.verdict_label.setText("HANDOFF CONDITIONS NOT MET  —  HANDOFF → NOT READY")
            self.verdict_label.setObjectName("link_not_ready")
            failed = [name for name, _v, passed in conditions if passed is False]
            reason = f"Failing: {', '.join(failed)}" if failed else "Conditions not yet all satisfied."
            self.summary_label.setText(reason)
        self.verdict_label.style().unpolish(self.verdict_label)
        self.verdict_label.style().polish(self.verdict_label)


def _fmt_report_value(value):
    return "—" if value is None else f"{float(value):.2f}"


app = QApplication(sys.argv)
window = FSOCWindow()
window.show()
# Launch with the window filling the screen work area on real desktops: the
# Virtual Camera feed is the flexible element of its column, so maximizing
# hands it every pixel the user's display can give. Restore geometry and all
# other panels are untouched (unmaximizing returns the previous default
# size). Headless sessions (QT_QPA_PLATFORM=offscreen, used by the automated
# test suite and CI) keep the plain-geometry path so programmatic resize
# checks stay deterministic.
if QApplication.platformName() != "offscreen":
    window.showMaximized()
sys.exit(app.exec())
