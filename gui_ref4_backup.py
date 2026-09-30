from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtGui import QBrush, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import config
from benchmark_video import run_video_benchmark
from controller import CameraController
from detector import detect_beacon
from performance_report import PerformanceSummary, save_performance_report
from simulator import Simulator
from tracker import BeaconTracker


# ---------------------------------------------------------------------------
# Small dependency-free visual widgets
# ---------------------------------------------------------------------------
class RadarWidget(QWidget):
    """Top-down angular radar for the virtual FSOC camera.

    Coordinate convention is shared with the simulator:
      +pan  = right / East
      -pan  = left / West
      +tilt = down / South
      -tilt = up / North

    The blue boresight is the camera's CURRENT pointing direction. The green
    and yellow markers are the measured beacon and one-step Kalman prediction.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 470)
        self.target_angles = (0.0, 0.0)
        self.predicted_angles = None
        self.camera_angles = (0.0, 0.0)
        self.state = "SEARCHING"
        self.range_m = 1000.0
        self.history = []
        self.max_angle = 3.0
        self.all_targets = []
        self.designated_id = "TGT-01"

    def set_data(self, target_angles, predicted_angles, camera_angles, state, range_m, targets=None, designated_id="TGT-01"):
        self.target_angles = (float(target_angles[0]), float(target_angles[1]))
        self.predicted_angles = (
            (float(predicted_angles[0]), float(predicted_angles[1]))
            if predicted_angles is not None else None
        )
        self.camera_angles = (float(camera_angles[0]), float(camera_angles[1]))
        self.state = str(state)
        self.range_m = float(range_m)
        self.all_targets = list(targets or [])
        self.designated_id = str(designated_id)

        visible = [self.target_angles, self.camera_angles]
        if self.predicted_angles is not None:
            visible.append(self.predicted_angles)

        # Include the complete camera FOV footprint in the scale.
        # Iterate over a snapshot: extending the same list while iterating
        # over it causes an endless loop and eventually a MemoryError.
        half_h = max(float(config.FOV_HORIZONTAL) * 0.5, 0.1)
        half_v = max(float(config.FOV_VERTICAL) * 0.5, 0.1)
        base_points = list(visible)
        for pan, tilt in base_points:
            visible.append((pan - half_h, tilt - half_v))
            visible.append((pan + half_h, tilt + half_v))

        largest = max(max(abs(pan), abs(tilt)) for pan, tilt in visible)
        self.max_angle = max(3.0, min(15.0, largest + 0.7))

        self.history.append(self.target_angles)
        self.history = self.history[-180:]
        self.update()

    def _point(self, angles, center, radius):
        pan, tilt = angles
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
        size = 11.0
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
        radius = min(rect.width(), rect.height()) * 0.35

        blue = QColor(60, 150, 255)
        blue_fill = QColor(60, 150, 255, 30)
        green = QColor(50, 255, 90)
        yellow = QColor(255, 220, 55)
        purple = QColor(170, 105, 255)
        white = QColor(235, 240, 245)
        gray = QColor(95, 105, 115)

        # Radar rings.
        painter.setPen(QPen(gray, 1))
        for fraction in (0.25, 0.5, 0.75, 1.0):
            rr = radius * fraction
            painter.drawEllipse(center, rr, rr)

        # Crosshair and cardinal directions.
        painter.drawLine(
            QPointF(center.x() - radius, center.y()),
            QPointF(center.x() + radius, center.y()),
        )
        painter.drawLine(
            QPointF(center.x(), center.y() - radius),
            QPointF(center.x(), center.y() + radius),
        )
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(white)
        painter.drawText(int(center.x()) - 5, int(center.y() - radius) - 10, "N")
        painter.drawText(int(center.x() + radius) + 8, int(center.y()) + 4, "E")
        painter.drawText(int(center.x()) - 5, int(center.y() + radius) + 20, "S")
        painter.drawText(int(center.x() - radius) - 18, int(center.y()) + 4, "W")

        # Target history.
        if len(self.history) > 1:
            painter.setPen(QPen(purple, 2))
            previous = self._point(self.history[0], center, radius)
            for angles in self.history[1:]:
                current = self._point(angles, center, radius)
                painter.drawLine(previous, current)
                previous = current

        # ---------------- CAMERA BORESIGHT / FOV ----------------
        camera_point = self._point(self.camera_angles, center, radius)
        half_h = max(float(config.FOV_HORIZONTAL) * 0.5, 0.1)
        half_v = max(float(config.FOV_VERTICAL) * 0.5, 0.1)
        cam_pan, cam_tilt = self.camera_angles
        fov_corners = [
            self._point((cam_pan - half_h, cam_tilt - half_v), center, radius),
            self._point((cam_pan + half_h, cam_tilt - half_v), center, radius),
            self._point((cam_pan + half_h, cam_tilt + half_v), center, radius),
            self._point((cam_pan - half_h, cam_tilt + half_v), center, radius),
        ]

        painter.setBrush(QBrush(blue_fill))
        painter.setPen(QPen(blue, 1))
        painter.drawPolygon(QPolygonF(fov_corners))
        for corner in fov_corners:
            painter.drawLine(center, corner)

        # Strong boresight arrow on top of the FOV.
        self._draw_arrow(painter, center, camera_point, blue, 4)
        painter.setBrush(QBrush(blue))
        painter.setPen(QPen(blue, 2))
        painter.drawEllipse(camera_point, 7, 7)

        # ---------------- TARGET MARKERS ----------------
        # Draw all visible targets. The designated target gets the strong green
        # marker; decoys remain visible as smaller white/magenta markers.
        target_point = self._point(self.target_angles, center, radius)
        target_map = {str(t.get("id")): t for t in self.all_targets}
        for tid, target in target_map.items():
            if not target.get("visible", True):
                continue
            tp = self._point((target.get("pan", 0.0), target.get("tilt", 0.0)), center, radius)
            is_designated = tid == self.designated_id
            marker_color = green if is_designated else QColor(255, 90, 180)
            marker_size = 11 if is_designated else 7
            painter.setBrush(QBrush(marker_color))
            painter.setPen(QPen(QColor(235, 245, 250), 2 if is_designated else 1))
            painter.drawEllipse(tp, marker_size, marker_size)
            painter.setPen(QPen(marker_color, 1))
            painter.drawEllipse(tp, marker_size + 6, marker_size + 6)
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(marker_color)
            painter.drawText(tp + QPointF(10, -7), tid + ("  DESIGNATED" if is_designated else "  DECOY"))

        # Fallback marker if no target list is available.
        if not target_map:
            painter.setBrush(QBrush(green))
            painter.setPen(QPen(QColor(225, 255, 225), 2))
            painter.drawEllipse(target_point, 11, 11)
            painter.setPen(QPen(green, 2))
            painter.drawEllipse(target_point, 18, 18)

        painter.setBrush(QBrush(yellow))
        painter.setPen(QPen(QColor(255, 255, 220), 2))
        if self.predicted_angles is not None:
            predicted_point = self._point(self.predicted_angles, center, radius)
            painter.drawEllipse(predicted_point, 8, 8)
            painter.setPen(QPen(yellow, 2))
            painter.drawEllipse(predicted_point, 14, 14)
            painter.drawLine(target_point, predicted_point)
        else:
            predicted_point = None

        # Optical centre marker.
        painter.setBrush(QBrush(Qt.transparent))
        painter.setPen(QPen(white, 1))
        painter.drawEllipse(center, 8, 8)
        painter.drawLine(center + QPointF(-14, 0), center + QPointF(14, 0))
        painter.drawLine(center + QPointF(0, -14), center + QPointF(0, 14))

        # Labels / live readout.
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(green)
        painter.drawText(target_point + QPointF(18, -8), "BEACON")

        if predicted_point is not None:
            painter.setPen(yellow)
            painter.drawText(predicted_point + QPointF(18, 18), "KALMAN PREDICTION")

        painter.setPen(blue)
        label_x = camera_point.x() + 10
        label_y = camera_point.y() - 8
        if label_y < 42:
            label_y = camera_point.y() + 20
        painter.drawText(QPointF(label_x, label_y), "CAMERA POINTING")

        painter.setPen(white)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(
            12,
            18,
            f"BORESIGHT  PAN {cam_pan:+.2f}°  |  TILT {cam_tilt:+.2f}°",
        )
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(170, 180, 190))
        painter.drawText(
            12,
            34,
            f"BEACON APP  PAN {self.target_angles[0]:+.2f}°  |  TILT {self.target_angles[1]:+.2f}°",
        )
        painter.drawText(
            rect.width() - 165,
            rect.height() - 12,
            f"FOV {config.FOV_HORIZONTAL:.1f}° × {config.FOV_VERTICAL:.1f}°",
        )
        painter.drawText(12, rect.height() - 12, f"RANGE {self.range_m:.0f} m")


class TrendGraph(QWidget):
    """Dependency-free live line graph for tracking telemetry."""

    def __init__(self, title, unit="", parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.series = []
        self.setMinimumHeight(160)

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._status = "NO CAMERA FRAME"
        self._set_frame_count = 0
        self.setMinimumSize(540, 340)
        self.setObjectName("camera_view")

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
        self.setWindowTitle("FSOC Virtual Camera Tracking System")
        self.resize(1600, 980)

        self.simulator = Simulator()
        self.simulator.set_target_count(3)
        self.simulator.set_designated_target("TGT-01")
        self.controller = CameraController()
        self.tracker = BeaconTracker()

        # Runtime parameters
        self.beacon_speed = 4.0
        self.beacon_range_m = 1000.0
        self.camera_fov_deg = 4.0
        self.camera_range_deg = 5.0
        self.camera_slew_rate = 5.0
        self.prediction_gain = 0.85
        self.beacon_size_px = 10.0
        self.noise_level = 10
        self.turbulence_strength = 10.0
        self.camera_jitter_strength = 5.0
        self.platform_motion_strength = 10.0
        self.atmosphere_level = 25
        self.disturbance_preset = "Clear"

        # Refinement 1: multi-target scenario. Keep several targets visible by
        # default so the application immediately demonstrates the PS capability.
        self.target_count = 3
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
        self.last_detection = None
        self.report_last_saved = None
        self.camera_target_history = []
        self.camera_prediction_history = []

        # Link-readiness stability gate. A link handoff is considered ready
        # only after several consecutive good tracking frames.
        self.link_ready_streak = 0
        self.link_required_frames = 10

        # Refinement 2: explicit PAT search/recovery control.
        self.search_speed_deg_s = 4.0
        self.recovery_radius_px = 50
        self.search_direction = 1
        self.search_phase = 0.0
        self.search_was_active = False
        self.last_pat_stage = "SEARCH"

        # Display-only camera zoom. This NEVER changes the 640×480 tracking frame.
        self.display_zoom = 1.0

        self._build_ui()
        self._refresh_designated_target_selector()
        self._apply_runtime_settings()
        self._reset_metrics()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(33)  # 30 Hz camera update target

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header
        header = QFrame()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 10, 16, 10)
        title_box = QVBoxLayout()
        title = QLabel("FSOC VIRTUAL CAMERA TRACKING SYSTEM")
        title.setObjectName("title")
        subtitle = QLabel(
            "AI Beacon Verification  •  Kalman Prediction  •  Coarse Pointing  •  Disturbance Testbed"
        )
        subtitle.setObjectName("subtitle")
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

        # Left control panel in a scroll area
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setMinimumWidth(350)
        control_scroll.setMaximumWidth(390)
        control_scroll.setFrameShape(QFrame.NoFrame)
        controls = QFrame()
        controls.setObjectName("panel")
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(8)
        control_scroll.setWidget(controls)

        controls_title = QLabel("SIMULATION CONTROL")
        controls_title.setObjectName("panel_title")
        cl.addWidget(controls_title)

        action = QHBoxLayout()
        action.setSpacing(6)
        self.start_button = QPushButton("▶  START")
        self.pause_button = QPushButton("⏸  PAUSE")
        self.reset_button = QPushButton("↻  RESET")
        self.benchmark_button = QPushButton("🎞  BENCHMARK VIDEO")
        self.report_button = QPushButton("📊  EXPORT REPORT")
        self.loss_test_button = QPushButton("🔎  TEST LOSS / RECOVER")
        self.start_button.setObjectName("start_button")
        self.pause_button.setObjectName("pause_button")
        self.reset_button.setObjectName("reset_button")
        for btn in [self.start_button, self.pause_button, self.reset_button]:
            action.addWidget(btn)
        cl.addLayout(action)
        cl.addWidget(self.benchmark_button)
        cl.addWidget(self.report_button)

        benchmark_mode = QFrame()
        benchmark_mode.setObjectName("subpanel")
        bmg = QGridLayout(benchmark_mode)
        bmg.setContentsMargins(8, 6, 8, 6)
        bmg.setHorizontalSpacing(8)
        bmg.addWidget(self._label("MP4 Target Mode"), 0, 0)
        self.benchmark_signature_selector = QComboBox()
        self.benchmark_signature_selector.addItems([
            "AUTO (Bright Beacon)",
            "GREEN-SQUARE",
            "AMBER-DIAMOND",
            "MAGENTA-CIRCLE",
            "CYAN-TRIANGLE",
            "RED-CIRCLE",
        ])
        self.benchmark_signature_selector.setToolTip(
            "AUTO is recommended for the supplied monochrome-style benchmark video."
        )
        bmg.addWidget(self.benchmark_signature_selector, 0, 1)
        self.benchmark_help = QLabel(
            "MP4 → tracked MP4 + telemetry CSV + performance report"
        )
        self.benchmark_help.setObjectName("caption")
        bmg.addWidget(self.benchmark_help, 1, 0, 1, 2)
        cl.addWidget(benchmark_mode)
        cl.addWidget(self.loss_test_button)
        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)
        self.benchmark_button.clicked.connect(self.benchmark_video)
        self.report_button.clicked.connect(self.export_report)
        self.loss_test_button.clicked.connect(self._trigger_target_loss_test)

        scenario = QFrame()
        scenario.setObjectName("subpanel")
        sg = QGridLayout(scenario)
        sg.setContentsMargins(8, 8, 8, 8)
        sg.setHorizontalSpacing(8)
        sg.setVerticalSpacing(6)
        sg.addWidget(self._label("Beacon Motion"), 0, 0)
        self.motion_selector = QComboBox()
        self.motion_selector.addItems(["Straight Line", "Circular", "Figure 8", "Random"])
        sg.addWidget(self.motion_selector, 0, 1)
        sg.addWidget(self._label("Image Noise"), 1, 0)
        self.noise_selector = QComboBox()
        self.noise_selector.addItems(["None", "Salt & Pepper", "Gaussian", "Poisson", "All Sensor Noise"])
        sg.addWidget(self.noise_selector, 1, 1)
        sg.addWidget(self._label("Platform Pattern"), 2, 0)
        self.platform_pattern_selector = QComboBox()
        self.platform_pattern_selector.addItems(["Linear", "Circular", "Random", "Figure 8"])
        sg.addWidget(self.platform_pattern_selector, 2, 1)
        sg.addWidget(self._label("Atmosphere"), 3, 0)
        self.atmosphere_selector = QComboBox()
        self.atmosphere_selector.addItems(["Clear", "Haze", "Fog", "Rain", "Low Light"])
        sg.addWidget(self.atmosphere_selector, 3, 1)

        sg.addWidget(self._label("Disturbance Preset"), 4, 0)
        self.disturbance_preset_selector = QComboBox()
        self.disturbance_preset_selector.addItems(["Clear", "Sensor Noise", "Vibration", "Atmosphere", "Full Combined", "Extreme PAT", "Custom"])
        self.disturbance_preset_selector.setCurrentText(self.disturbance_preset)
        sg.addWidget(self.disturbance_preset_selector, 4, 1)

        sg.addWidget(self._label("Moving Targets"), 5, 0)
        self.target_count_selector = QComboBox()
        self.target_count_selector.addItems(["1", "2", "3", "4", "5"])
        self.target_count_selector.setCurrentText(str(self.target_count))
        sg.addWidget(self.target_count_selector, 5, 1)

        sg.addWidget(self._label("Designated Target"), 6, 0)
        self.designated_target_selector = QComboBox()
        self.designated_target_selector.addItems(["TGT-01", "TGT-02", "TGT-03"])
        self.designated_target_selector.setCurrentText(self.designated_target_id)
        sg.addWidget(self.designated_target_selector, 6, 1)

        self.target_help = QLabel("Green = designated communication target  •  Pink = decoys")
        self.target_help.setObjectName("caption")
        sg.addWidget(self.target_help, 7, 0, 1, 2)

        self.motion_selector.currentTextChanged.connect(self.change_motion_pattern)
        self.target_count_selector.currentTextChanged.connect(self._on_target_count_changed)
        self.designated_target_selector.currentTextChanged.connect(self._on_designated_target_changed)
        self.noise_selector.currentTextChanged.connect(self.change_noise_type)
        self.disturbance_preset_selector.currentTextChanged.connect(self._apply_disturbance_preset)
        self.platform_pattern_selector.currentTextChanged.connect(self._apply_disturbances)
        self.atmosphere_selector.currentTextChanged.connect(self._apply_disturbances)
        cl.addWidget(scenario)

        parameter_box = QFrame()
        parameter_box.setObjectName("subpanel")
        pg = QGridLayout(parameter_box)
        pg.setContentsMargins(8, 8, 8, 8)
        pg.setHorizontalSpacing(8)
        pg.setVerticalSpacing(6)

        self.speed_slider, self.speed_value = self._make_slider(1, 100, 40, 10, "px/frame")
        self.range_slider, self.range_value = self._make_slider(1, 100, 10, 1, "m×100")
        self.fov_slider, self.fov_value = self._make_slider(10, 100, 40, 10, "°")
        self.cam_range_slider, self.cam_range_value = self._make_slider(10, 100, 50, 10, "±°")
        self.slew_slider, self.slew_value = self._make_slider(10, 100, 50, 10, "°/s")
        self.pred_slider, self.pred_value = self._make_slider(0, 100, 85, 100, "gain")
        self.size_slider, self.size_value = self._make_slider(5, 20, 10, 1, "px")
        self.noise_level_slider, self.noise_level_value = self._make_slider(0, 20, 10, 1, "level")
        self.turbulence_slider, self.turbulence_value = self._make_slider(0, 20, 10, 1, "px")
        self.jitter_slider, self.jitter_value = self._make_slider(0, 20, 5, 1, "±px/frame")
        self.platform_slider, self.platform_value = self._make_slider(0, 20, 10, 1, "±px")
        self.atmosphere_slider, self.atmosphere_value = self._make_slider(0, 100, 25, 1, "%")

        specs = [
            ("Beacon Speed", self.speed_slider, self.speed_value),
            ("Beacon Range", self.range_slider, self.range_value),
            ("Camera FOV", self.fov_slider, self.fov_value),
            ("Pan/Tilt Range", self.cam_range_slider, self.cam_range_value),
            ("Slew Rate", self.slew_slider, self.slew_value),
            ("Prediction Gain", self.pred_slider, self.pred_value),
            ("Beacon Size", self.size_slider, self.size_value),
            ("Noise Level", self.noise_level_slider, self.noise_level_value),
        ]
        for index, (label_text, slider, value_label) in enumerate(specs):
            row, col = divmod(index, 2)
            card = self._slider_card(label_text, slider, value_label)
            pg.addWidget(card, row, col)
        cl.addWidget(parameter_box)

        disturbances = QFrame()
        disturbances.setObjectName("subpanel")
        dg = QGridLayout(disturbances)
        dg.setContentsMargins(8, 8, 8, 8)
        dg.setHorizontalSpacing(8)
        dg.setVerticalSpacing(5)
        dt = QLabel("DISTURBANCES")
        dt.setObjectName("section_title")
        dg.addWidget(dt, 0, 0, 1, 2)

        self.turbulence_check = QCheckBox("Atmospheric Turbulence")
        self.jitter_check = QCheckBox("Camera Jitter")
        self.platform_check = QCheckBox("Platform Motion")
        self.turbulence_check.setChecked(False)
        self.jitter_check.setChecked(False)
        self.platform_check.setChecked(False)
        dg.addWidget(self.turbulence_check, 1, 0, 1, 2)
        dg.addWidget(self.jitter_check, 2, 0, 1, 2)
        dg.addWidget(self.platform_check, 3, 0, 1, 2)

        for row, label_text, slider, value_label in [
            (4, "Turbulence", self.turbulence_slider, self.turbulence_value),
            (5, "Camera Jitter", self.jitter_slider, self.jitter_value),
            (6, "Platform Motion", self.platform_slider, self.platform_value),
            (7, "Atmosphere Intensity", self.atmosphere_slider, self.atmosphere_value),
        ]:
            dg.addWidget(self._label(label_text), row, 0)
            dg.addWidget(slider, row, 1)
            dg.addWidget(value_label, row, 2)
        cl.addWidget(disturbances)
        self.turbulence_check.stateChanged.connect(self._apply_disturbances)
        self.jitter_check.stateChanged.connect(self._apply_disturbances)
        self.platform_check.stateChanged.connect(self._apply_disturbances)

        for slider, handler in [
            (self.speed_slider, self._on_speed_changed),
            (self.range_slider, self._on_range_changed),
            (self.fov_slider, self._on_fov_changed),
            (self.cam_range_slider, self._on_camera_range_changed),
            (self.slew_slider, self._on_slew_changed),
            (self.pred_slider, self._on_prediction_changed),
            (self.size_slider, self._on_size_changed),
            (self.noise_level_slider, self._on_noise_level_changed),
            (self.turbulence_slider, self._apply_disturbances),
            (self.jitter_slider, self._apply_disturbances),
            (self.platform_slider, self._apply_disturbances),
            (self.atmosphere_slider, self._apply_disturbances),
        ]:
            slider.valueChanged.connect(handler)

        self.info = QLabel()
        self.info.setObjectName("info_box")
        self.info.setWordWrap(True)
        cl.addWidget(self.info)
        cl.addStretch()

        # Center column: radar + graphs
        center = QVBoxLayout()
        center.setSpacing(10)

        radar_panel = QFrame()
        radar_panel.setObjectName("panel")
        rl = QVBoxLayout(radar_panel)
        rl.setContentsMargins(10, 10, 10, 8)
        radar_header = QHBoxLayout()
        radar_title = QLabel("VIRTUAL FSOC RADAR")
        radar_title.setObjectName("panel_title")
        radar_header.addWidget(radar_title)
        radar_header.addStretch()
        self.radar_state = QLabel("● SEARCHING")
        self.radar_state.setObjectName("radar_state")
        radar_header.addWidget(self.radar_state)
        rl.addLayout(radar_header)
        radar_caption = QLabel("Target trajectory  •  Kalman prediction  •  Camera boresight  •  FOV")
        radar_caption.setObjectName("caption")
        rl.addWidget(radar_caption)
        self.radar = RadarWidget()
        rl.addWidget(self.radar, 1)

        legend = QFrame()
        legend.setObjectName("legend_panel")
        lg = QHBoxLayout(legend)
        lg.setContentsMargins(8, 5, 8, 5)
        lg.setSpacing(12)
        for color, text in [
            ("#39ff66", "GREEN  Beacon"),
            ("#ffd83d", "YELLOW  Kalman prediction"),
            ("#4aa3ff", "BLUE  Camera boresight / FOV"),
            ("#a66cff", "PURPLE  Target trail"),
        ]:
            label = QLabel(f'<span style="color:{color}">●</span> {text}')
            label.setObjectName("legend_item")
            lg.addWidget(label)
        lg.addStretch()
        rl.addWidget(legend)
        center.addWidget(radar_panel, 3)

        graph_panel = QFrame()
        graph_panel.setObjectName("panel")
        gg = QGridLayout(graph_panel)
        gg.setContentsMargins(8, 8, 8, 8)
        gg.setHorizontalSpacing(8)
        self.error_graph = TrendGraph("TRACKING / CENTROID ERROR", "px")
        self.motion_graph = TrendGraph("PAN / TILT CONTROL RESPONSE", "deg")
        gg.addWidget(self.error_graph, 0, 0)
        gg.addWidget(self.motion_graph, 0, 1)
        center.addWidget(graph_panel, 1)

        # Right column: camera + telemetry + Kalman
        right = QVBoxLayout()
        right.setSpacing(10)

        camera_panel = QFrame()
        camera_panel.setObjectName("panel")
        cpl = QVBoxLayout(camera_panel)
        cpl.setContentsMargins(10, 10, 10, 10)
        cam_head = QHBoxLayout()
        cam_title = QLabel("VIRTUAL CAMERA")
        cam_title.setObjectName("panel_title")
        cam_head.addWidget(cam_title)
        cam_head.addStretch()
        self.camera_lock_badge = QLabel("● SEARCHING")
        self.camera_lock_badge.setObjectName("camera_badge")
        cam_head.addWidget(self.camera_lock_badge)
        cpl.addLayout(cam_head)
        self.camera_info = QLabel()
        self.camera_info.setObjectName("caption")
        cpl.addWidget(self.camera_info)
        self.camera_label = CameraFeedWidget()
        cpl.addWidget(self.camera_label, 1)

        # Display-only zoom controls. Tracking continues to use the original
        # 640×480 camera frame, so zoom cannot change benchmark accuracy.
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)
        zoom_label = QLabel("DISPLAY ZOOM")
        zoom_label.setObjectName("small_label")
        zoom_row.addWidget(zoom_label)

        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setObjectName("zoom_button")
        self.zoom_out_button.setToolTip("Zoom out")
        self.zoom_out_button.clicked.connect(lambda: self._step_zoom(-0.5))
        zoom_row.addWidget(self.zoom_out_button)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 40)
        self.zoom_slider.setValue(10)
        self.zoom_slider.setToolTip("Display zoom: 1× to 4×")
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        zoom_row.addWidget(self.zoom_slider, 1)

        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setObjectName("zoom_button")
        self.zoom_in_button.setToolTip("Zoom in")
        self.zoom_in_button.clicked.connect(lambda: self._step_zoom(0.5))
        zoom_row.addWidget(self.zoom_in_button)

        self.zoom_reset_button = QPushButton("1×")
        self.zoom_reset_button.setObjectName("zoom_reset_button")
        self.zoom_reset_button.setToolTip("Reset display zoom to 1×")
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        zoom_row.addWidget(self.zoom_reset_button)

        self.zoom_value_label = QLabel("1.0×")
        self.zoom_value_label.setObjectName("zoom_value")
        self.zoom_value_label.setMinimumWidth(34)
        zoom_row.addWidget(self.zoom_value_label)
        cpl.addLayout(zoom_row)
        right.addWidget(camera_panel, 3)

        telemetry = QFrame()
        telemetry.setObjectName("panel")
        tl = QGridLayout(telemetry)
        tl.setContentsMargins(10, 8, 10, 8)
        tl.setHorizontalSpacing(7)
        tl.setVerticalSpacing(4)
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
        metrics = [
            ("Processing FPS", self.t_fps),
            ("Tracking Error", self.t_error),
            ("Centroid Error", self.t_centroid),
            ("Avg Error", self.t_avg),
            ("Max Error", self.t_max),
            ("RMSE", self.t_rmse),
            ("Acquisition", self.t_acq),
            ("Re-acquisition", self.t_reacq),
            ("Target Loss", self.t_loss),
            ("Lock Retention", self.t_lock),
            ("State", self.t_state),
            ("Pan", self.t_pan),
            ("Tilt", self.t_tilt),
            ("Velocity", self.t_velocity),
            ("Filter Confidence", self.t_confidence),
            ("AI Confidence", self.t_ai),
        ]
        for index, (label_text, value) in enumerate(metrics):
            row = 1 + index // 4
            col = index % 4
            card = QFrame()
            card.setObjectName("metric_card")
            ml = QVBoxLayout(card)
            ml.setContentsMargins(6, 4, 6, 4)
            ml.setSpacing(0)
            name = QLabel(label_text)
            name.setObjectName("metric_label")
            value.setObjectName("metric_value")
            ml.addWidget(name)
            ml.addWidget(value)
            tl.addWidget(card, row, col)
        right.addWidget(telemetry, 2)

        kalman_panel = QFrame()
        kalman_panel.setObjectName("kalman_panel")
        kl = QGridLayout(kalman_panel)
        kl.setContentsMargins(10, 8, 10, 8)
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
        kl.addWidget(self.kalman_state_label, 1, 0, 1, 2)
        kl.addWidget(self.kalman_pos_label, 2, 0)
        kl.addWidget(self.kalman_vel_label, 2, 1)
        kl.addWidget(self.kalman_uncertainty_label, 3, 0)
        kl.addWidget(self.kalman_next_label, 3, 1)
        right.addWidget(kalman_panel, 1)

        readiness_panel = QFrame()
        readiness_panel.setObjectName("readiness_panel")
        rgl = QGridLayout(readiness_panel)
        rgl.setContentsMargins(10, 8, 10, 8)
        rgl.setHorizontalSpacing(10)
        rgl.setVerticalSpacing(3)
        readiness_title = QLabel("LINK READINESS")
        readiness_title.setObjectName("panel_title")
        rgl.addWidget(readiness_title, 0, 0, 1, 2)
        self.link_readiness_label = QLabel("● NOT READY")
        self.link_readiness_label.setObjectName("link_not_ready")
        self.link_readiness_label.setToolTip(
            "Coarse-alignment readiness only. A physical FSOC communication link may still require fine pointing and hardware verification."
        )
        rgl.addWidget(self.link_readiness_label, 1, 0, 1, 2)
        self.link_conditions_label = QLabel("Tracking: —  |  Error: —  |  FPS: —  |  Confidence: —")
        self.link_conditions_label.setObjectName("caption")
        rgl.addWidget(self.link_conditions_label, 2, 0, 1, 2)
        right.addWidget(readiness_panel, 1)

        body.addWidget(control_scroll, 1)
        body.addLayout(center, 2)
        body.addLayout(right, 2)
        root.addLayout(body, 1)

        # Footer
        footer = QFrame()
        footer.setObjectName("footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 5, 10, 5)
        self.footer_status = QLabel("Ready • Same tracking engine used for simulation and benchmark video")
        self.footer_status.setObjectName("footer_status")
        fl.addWidget(self.footer_status)
        fl.addStretch()
        fl.addWidget(QLabel("PS targets: ≤10 px error  •  <5% loss  •  ≤2 s acquisition  •  ≤1 s re-acquisition  •  ≥20 FPS"))
        root.addWidget(footer)

        self.setStyleSheet("""
            QMainWindow { background-color: #05080d; }
            QWidget { color: #e7edf3; font-family: "Segoe UI"; }
            #header, #panel, #footer {
                background-color: #0d151f;
                border: 1px solid #233545;
                border-radius: 10px;
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
            #metric_label { font-size: 7px; color: #738596; }
            #metric_value { font-size: 12px; font-weight: 700; color: #eef3f7; }
            #slider_value { font-size: 8px; font-weight: 700; color: #f06a6a; }
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
            #start_button { background-color: #123622; border-color: #2eaa63; color: #67f0a0; }
            #pause_button { background-color: #3b2d11; border-color: #aa8030; color: #f4c967; }
            #reset_button { background-color: #3b171b; border-color: #a54850; color: #ff8a92; }
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
        """)

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setObjectName("small_label")
        return label

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
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setObjectName("small_label")
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(value_label)
        layout.addLayout(row)
        layout.addWidget(slider)
        return card

    # ------------------------------------------------------------------
    # Runtime settings
    # ------------------------------------------------------------------
    def _apply_runtime_settings(self):
        self.simulator.set_beacon_speed(self.beacon_speed)
        self.simulator.set_beacon_range(self.beacon_range_m)
        self.simulator.set_beacon_size(self.beacon_size_px)
        self.simulator.set_fov(self.camera_fov_deg)
        config.MAX_PAN_SPEED = self.camera_slew_rate
        config.MAX_TILT_SPEED = self.camera_slew_rate
        self.controller.set_angle_limits(self.camera_range_deg, self.camera_range_deg * 0.75)
        self.controller.set_gain(2.0)
        self.simulator.noise_level = self.noise_level
        self._apply_disturbances()

        self.speed_value.setText(f"{self.beacon_speed:.1f} px/frame")
        self.range_value.setText(f"{self.beacon_range_m:.0f} m")
        self.fov_value.setText(f"{self.camera_fov_deg:.1f}°")
        self.cam_range_value.setText(f"±{self.camera_range_deg:.1f}°")
        self.slew_value.setText(f"{self.camera_slew_rate:.1f}°/s")
        self.pred_value.setText(f"{self.prediction_gain:.2f}")
        self.size_value.setText(f"{self.beacon_size_px:.0f} px")
        self.noise_level_value.setText(str(self.noise_level))
        self.turbulence_value.setText(f"{self.turbulence_strength:.0f} px")
        self.jitter_value.setText(f"±{self.camera_jitter_strength:.0f} px/frame")
        self.platform_value.setText(f"±{self.platform_motion_strength:.0f} px")
        self.atmosphere_value.setText(f"{self.atmosphere_level}%")
        self.camera_info.setText(
            f"{config.FRAME_WIDTH} × {config.FRAME_HEIGHT}  |  FOV {config.FOV_HORIZONTAL:.1f}° × {config.FOV_VERTICAL:.1f}°  |  30 Hz"
        )

    def _apply_disturbances(self, _value=None):
        # Any direct edit to a disturbance control switches the profile to
        # Custom; preset application itself calls this method with no signal
        # argument after the widgets have been updated.
        if _value is not None and hasattr(self, "disturbance_preset_selector"):
            if self.disturbance_preset_selector.currentText() != "Custom":
                self.disturbance_preset_selector.blockSignals(True)
                self.disturbance_preset_selector.setCurrentText("Custom")
                self.disturbance_preset_selector.blockSignals(False)
        self.turbulence_strength = float(self.turbulence_slider.value())
        self.camera_jitter_strength = float(self.jitter_slider.value())
        self.platform_motion_strength = float(self.platform_slider.value())
        self.atmosphere_level = int(self.atmosphere_slider.value())
        self.turbulence_value.setText(f"{self.turbulence_strength:.0f} px")
        self.jitter_value.setText(f"±{self.camera_jitter_strength:.0f} px/frame")
        self.platform_value.setText(f"±{self.platform_motion_strength:.0f} px")
        self.atmosphere_value.setText(f"{self.atmosphere_level}%")
        self.disturbance_preset = self.disturbance_preset_selector.currentText() if hasattr(self, "disturbance_preset_selector") else self.disturbance_preset
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
            disturbance_preset=self.disturbance_preset,
        )

    def _apply_disturbance_preset(self, preset):
        if not hasattr(self, "disturbance_preset_selector"):
            return
        preset = str(preset)
        self.disturbance_preset = preset
        if preset == "Custom":
            self._apply_disturbances()
            self._reset_metrics()
            return

        profiles = {
            "Clear": {"noise": "None", "noise_level": 0, "turbulence": False, "turbulence_px": 0, "jitter": False, "jitter_px": 0, "platform": False, "platform_px": 0, "atmosphere": "Clear", "atmosphere_level": 0},
            "Sensor Noise": {"noise": "All Sensor Noise", "noise_level": 20, "turbulence": False, "turbulence_px": 0, "jitter": False, "jitter_px": 0, "platform": False, "platform_px": 0, "atmosphere": "Clear", "atmosphere_level": 0},
            "Vibration": {"noise": "None", "noise_level": 0, "turbulence": True, "turbulence_px": 10, "jitter": True, "jitter_px": 15, "platform": True, "platform_px": 15, "atmosphere": "Clear", "atmosphere_level": 0},
            "Atmosphere": {"noise": "None", "noise_level": 0, "turbulence": True, "turbulence_px": 8, "jitter": False, "jitter_px": 0, "platform": False, "platform_px": 0, "atmosphere": "Fog", "atmosphere_level": 70},
            "Full Combined": {"noise": "All Sensor Noise", "noise_level": 15, "turbulence": True, "turbulence_px": 12, "jitter": True, "jitter_px": 12, "platform": True, "platform_px": 12, "atmosphere": "Haze", "atmosphere_level": 55},
            "Extreme PAT": {"noise": "All Sensor Noise", "noise_level": 20, "turbulence": True, "turbulence_px": 20, "jitter": True, "jitter_px": 20, "platform": True, "platform_px": 20, "atmosphere": "Fog", "atmosphere_level": 85},
        }
        profile = profiles.get(preset, profiles["Clear"])

        widgets = [
            self.noise_selector, self.atmosphere_selector, self.platform_pattern_selector,
            self.turbulence_check, self.jitter_check, self.platform_check,
            self.noise_level_slider, self.turbulence_slider, self.jitter_slider, self.platform_slider, self.atmosphere_slider,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.noise_selector.setCurrentText(profile["noise"])
            self.noise_level_slider.setValue(profile["noise_level"])
            self.turbulence_check.setChecked(profile["turbulence"])
            self.turbulence_slider.setValue(profile["turbulence_px"])
            self.jitter_check.setChecked(profile["jitter"])
            self.jitter_slider.setValue(profile["jitter_px"])
            self.platform_check.setChecked(profile["platform"])
            self.platform_slider.setValue(profile["platform_px"])
            self.platform_pattern_selector.setCurrentText("Linear")
            self.atmosphere_selector.setCurrentText(profile["atmosphere"])
            self.atmosphere_slider.setValue(profile["atmosphere_level"])
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        self.simulator.set_noise_type(profile["noise"])
        self.simulator.noise_level = profile["noise_level"]
        self.turbulence_strength = float(profile["turbulence_px"])
        self.camera_jitter_strength = float(profile["jitter_px"])
        self.platform_motion_strength = float(profile["platform_px"])
        self.atmosphere_level = int(profile["atmosphere_level"])
        self._apply_disturbances()
        self._reset_metrics()
        self.footer_status.setText(f"Disturbance preset: {preset}")

    def _on_speed_changed(self, value):
        self.beacon_speed = value / 10.0
        self.speed_value.setText(f"{self.beacon_speed:.1f} px/frame")
        self.simulator.set_beacon_speed(self.beacon_speed)

    def _on_range_changed(self, value):
        self.beacon_range_m = float(value * 100)
        self.range_value.setText(f"{self.beacon_range_m:.0f} m")
        self.simulator.set_beacon_range(self.beacon_range_m)

    def _on_fov_changed(self, value):
        self.camera_fov_deg = value / 10.0
        self.fov_value.setText(f"{self.camera_fov_deg:.1f}°")
        self.simulator.set_fov(self.camera_fov_deg)
        self.radar.max_angle = max(3.0, self.camera_fov_deg * 0.9)
        self.camera_info.setText(
            f"{config.FRAME_WIDTH} × {config.FRAME_HEIGHT}  |  FOV {config.FOV_HORIZONTAL:.1f}° × {config.FOV_VERTICAL:.1f}°  |  30 Hz"
        )

    def _on_camera_range_changed(self, value):
        self.camera_range_deg = value / 10.0
        self.cam_range_value.setText(f"±{self.camera_range_deg:.1f}°")
        self.controller.set_angle_limits(self.camera_range_deg, self.camera_range_deg * 0.75)

    def _on_slew_changed(self, value):
        self.camera_slew_rate = value / 10.0
        self.slew_value.setText(f"{self.camera_slew_rate:.1f}°/s")
        config.MAX_PAN_SPEED = self.camera_slew_rate
        config.MAX_TILT_SPEED = self.camera_slew_rate

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
        self.last_detection = None
        self.camera_target_history.clear()
        self.camera_prediction_history.clear()
        self.link_ready_streak = 0
        self.fps = 0.0
        self.search_direction = 1
        self.search_phase = 0.0
        self.search_was_active = False
        self.last_pat_stage = "SEARCH"

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
        self.controller = CameraController()
        self._apply_runtime_settings()
        self.tracker.reset()
        self.radar.history.clear()
        self._reset_metrics()

    def change_noise_type(self, noise_type):
        if hasattr(self, "disturbance_preset_selector") and self.disturbance_preset_selector.currentText() != "Custom":
            self.disturbance_preset_selector.blockSignals(True)
            self.disturbance_preset_selector.setCurrentText("Custom")
            self.disturbance_preset_selector.blockSignals(False)
        self.disturbance_preset = "Custom"
        self.simulator.set_noise_type(noise_type)
        self._reset_metrics()

    def start_simulation(self):
        self.simulation_running = True
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
        self.disturbance_preset = self.disturbance_preset_selector.currentText()
        self.controller = CameraController()
        self.tracker.reset()
        self._refresh_designated_target_selector()
        self._apply_runtime_settings()
        self.radar.history.clear()
        self._reset_metrics()
        self.simulation_running = True
        self.timer.start(33)
        self.system_status.setText("● SYSTEM ACTIVE")
        self.footer_status.setText("Reset • ready for a new scenario")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    def update_simulation(self):
        if not self.simulation_running:
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
        if self.tracker.state_vector is not None and self.tracker.x is not None:
            expected_position = (float(self.tracker.x), float(self.tracker.y))

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
                # During recovery, keep steering toward the Kalman prediction.
                pan, tilt = self.controller.update(
                    control_x - center_x,
                    control_y - center_y,
                )
                self.simulator.update_camera(pan, tilt)
                self.search_was_active = False
            elif state == "LOST":
                # Prediction grace is exhausted: perform a bounded search sweep.
                self._perform_search_sweep()
                pan, tilt = self.controller.pan, self.controller.tilt
            else:
                pan, tilt = self.controller.pan, self.controller.tilt
                self.search_was_active = False

            # Acquisition window: do not pollute tracking accuracy with the
            # initial corner-to-center acquisition manoeuvre.
            if state == "TRACKING" and measured_error <= self.lock_error_threshold_px:
                self.lock_candidate_frames += 1
            else:
                self.lock_candidate_frames = 0

            if not self.metrics_started and self.lock_candidate_frames >= self.lock_required_frames:
                self.metrics_started = True
                self.acquisition_time_s = self.frames_processed / config.UPDATE_RATE

            if self.metrics_started and state == "TRACKING":
                self.tracking_frames += 1
                self.error_sum += measured_error
                self.error_squared_sum += measured_error * measured_error
                self.error_count += 1
                self.average_error = self.error_sum / self.error_count
                self.maximum_error = max(self.maximum_error, measured_error)
                self.error_history.append(float(measured_error))

            # Loss / re-acquisition events are driven by the detector itself.
            # This also captures deliberate target-loss tests where the target
            # is hidden and therefore has no ground-truth image position.
            if detection is None:
                if not self.in_loss and self.metrics_started:
                    self.loss_events += 1
                    self.in_loss = True
                    self.loss_start_time = self.frames_processed / config.UPDATE_RATE
            elif self.in_loss and state in {"RE-ACQUIRING", "TRACKING"}:
                if self.loss_start_time is not None:
                    reacq = self.frames_processed / config.UPDATE_RATE - self.loss_start_time
                    self.reacquisition_times.append(reacq)
                    self.successful_reacquisitions += 1
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
            if state in {"PREDICTING", "RE-ACQUIRING"}:
                cv2.circle(frame, (pred_x, pred_y), int(self.recovery_radius_px), (255, 180, 0), 1)
                cv2.putText(frame, "PREDICTED SEARCH REGION",
                            (max(8, pred_x - 74), max(18, pred_y - int(self.recovery_radius_px) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 180, 0), 1, cv2.LINE_AA)

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
            self.radar.set_data(
                self.simulator.get_target_angles(),
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
                f"{(sum(self.reacquisition_times) / len(self.reacquisition_times)):.2f} s"
                if self.reacquisition_times else "—"
            )
            self.t_loss.setText(f"{self._target_loss_rate():.2f} %")
            self.t_lock.setText(f"{self._lock_retention():.2f} %")
            self.t_state.setText(state)
            self.t_pan.setText(f"{pan:.2f}°")
            self.t_tilt.setText(f"{tilt:.2f}°")
            self.t_velocity.setText(f"{velocity:.2f} px/s")
            self.t_confidence.setText(f"{confidence:.3f}")
            self.t_ai.setText(f"{ai_conf:.3f}")
        else:
            state = self.tracker.state
            if state in {"SEARCHING", "LOST"}:
                self._perform_search_sweep()
            if self.metrics_started and not self.in_loss:
                self.loss_events += 1
                self.in_loss = True
                self.loss_start_time = self.frames_processed / config.UPDATE_RATE
            self.radar.set_data(
                self.simulator.get_target_angles(),
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
            self.t_pan.setText(f"{self.controller.pan:.2f}°")
            self.t_tilt.setText(f"{self.controller.tilt:.2f}°")
            search_overlay = frame.copy()
            cv2.line(search_overlay, (center_x - 18, center_y), (center_x + 18, center_y), (255, 0, 0), 1)
            cv2.line(search_overlay, (center_x, center_y - 18), (center_x, center_y + 18), (255, 0, 0), 1)
            stage_text = "SEARCHING FOR DESIGNATED TARGET" if state == "SEARCHING" else "TARGET LOST • CAMERA SEARCHING"
            cv2.putText(search_overlay, stage_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(search_overlay, f"PAN {self.controller.pan:+.2f}°  TILT {self.controller.tilt:+.2f}°", (12, config.FRAME_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            self._display_camera_frame(search_overlay, f"{self._pat_stage(state)}  •  SEARCH SWEEP", (center_x, center_y))
            self._update_kalman_panel(
                state,
                None,
                None,
                1.0 / config.UPDATE_RATE,
                getattr(self.tracker, "position_uncertainty", None),
            )

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
            f"Disturbance Preset {self.disturbance_preset} | Active {self.simulator.get_disturbance_status().get('count', 0)}\n"
            "AI Beacon Verification: ON  •  Kalman Prediction: ON"
        )

        self.camera_info.setText(
            f"640 × 480  |  FOV {config.FOV_HORIZONTAL:.1f}° × {config.FOV_VERTICAL:.1f}°  |  UPDATE {config.UPDATE_RATE} Hz  |  TARGETS {self.target_count}"
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
        self._display_camera_frame(frame, camera_status, display_focus)

        # Link-readiness indicator: this is a coarse-alignment gate, not a
        # physical communication-link guarantee. The gate deliberately uses
        # the LIVE detector/tracker state plus the same PS-style coarse
        # alignment limits, and requires a short stable run so the badge does
        # not flicker READY on a single lucky frame.
        current_state = self.tracker.state
        current_error = measured_error if tracked is not None else None
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
            and ai_conf >= 0.05
        )

        if ready_frame:
            self.link_ready_streak += 1
        else:
            self.link_ready_streak = 0

        ready = self.link_ready_streak >= self.link_required_frames

        if ready:
            self.link_readiness_label.setText("● HANDOFF READY  •  LINK READY")
            self.link_readiness_label.setObjectName("link_ready")
            self.last_pat_stage = "HANDOFF READY"
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

        error_text = f"{current_error:.2f} px" if current_error is not None else "—"
        self.link_conditions_label.setText(
            f"Tracking: {current_state}  |  Error: {error_text}  |  "
            f"FPS: {self.fps:.1f}  |  Detector: {detector_conf:.2f}  |  "
            f"AI: {ai_conf:.2f}  |  Signature: {float(detection_info.get('signature_score', 0.0)):.2f}  |  Stable: {self.link_ready_streak}/{self.link_required_frames}"
        )

    def _on_zoom_changed(self, value):
        self.display_zoom = max(1.0, min(4.0, float(value) / 10.0))
        self.zoom_value_label.setText(f"{self.display_zoom:.1f}×")

    def _step_zoom(self, delta):
        new_value = int(round((self.display_zoom + delta) * 10.0))
        new_value = max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), new_value))
        self.zoom_slider.setValue(new_value)

    def _pat_stage(self, state, link_ready=False):
        if link_ready:
            return "HANDOFF READY"
        if state == "SEARCHING":
            return "SEARCH"
        if state == "ACQUIRING":
            return "ACQUIRE"
        if state == "TRACKING":
            return "TRACK"
        if state in {"PREDICTING", "RE-ACQUIRING"}:
            return "RECOVER"
        if state == "LOST":
            return "SEARCH"
        return "SEARCH"

    def _perform_search_sweep(self):
        """Bounded camera scan used before acquisition and after target loss."""
        max_pan = max(0.5, float(self.camera_range_deg))
        max_tilt = max(0.4, float(self.camera_range_deg) * 0.75)
        step = max(0.05, float(self.search_speed_deg_s) / max(float(config.UPDATE_RATE), 1.0))

        if not self.search_was_active:
            self.search_direction = 1
            self.search_phase = 0.0
            self.search_was_active = True

        next_pan = float(self.controller.pan) + self.search_direction * step
        if next_pan >= max_pan:
            next_pan = max_pan
            self.search_direction = -1
        elif next_pan <= -max_pan:
            next_pan = -max_pan
            self.search_direction = 1

        self.search_phase += 0.18
        target_tilt = max_tilt * 0.28 * math.sin(self.search_phase)
        tilt_delta = max(-step, min(step, target_tilt - float(self.controller.tilt)))
        next_tilt = float(self.controller.tilt) + tilt_delta
        next_tilt = max(-max_tilt, min(max_tilt, next_tilt))

        self.controller.pan = next_pan
        self.controller.tilt = next_tilt
        self.simulator.update_camera(next_pan, next_tilt)

    def _trigger_target_loss_test(self):
        """Schedule a deterministic short loss/recovery event."""
        start = int(self.simulator.frame_count + 30)
        duration = 18
        self.simulator.set_beacon_loss(start, duration)
        self.footer_status.setText(
            f"Recovery test armed • loss in ~1.0 s for {duration / config.UPDATE_RATE:.2f} s"
        )

    def _reset_zoom(self):
        self.zoom_slider.setValue(10)

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
        if self.visible_frames <= 0:
            return 0.0
        return 100.0 * self.detection_misses / self.visible_frames

    def _lock_retention(self):
        if self.visible_frames <= 0:
            return 0.0
        return 100.0 * self.tracking_frames / self.visible_frames

    # ------------------------------------------------------------------
    # Reports / benchmark video
    # ------------------------------------------------------------------
    def _make_summary(self, source="live simulation"):
        duration = max(time.perf_counter() - self.run_start_time, 1e-6)
        fps = self.frames_processed / duration if self.frames_processed else 0.0
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

    def _save_current_report(self, prefix="performance"):
        summary = self._make_summary()
        paths = save_performance_report(summary, output_dir="reports", prefix=prefix)
        self.report_last_saved = paths
        return paths

    def export_report(self):
        try:
            paths = self._save_current_report(prefix="simulation_performance")
        except OSError as exc:
            QMessageBox.critical(self, "Report Error", str(exc))
            return
        QMessageBox.information(
            self,
            "Performance Report Saved",
            f"JSON: {paths[0]}\nCSV: {paths[1]}",
        )
        self.footer_status.setText(f"Report saved • {paths[0].name}")

    def benchmark_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Benchmark MP4",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
        )
        if not video_path:
            return

        self.pause_simulation()
        self.footer_status.setText("Running MP4 benchmark • generating tracked video + telemetry…")
        QApplication.processEvents()

        mode = self.benchmark_signature_selector.currentText()
        signature = None if mode.startswith("AUTO") else mode

        try:
            summary, paths = run_video_benchmark(
                video_path,
                output_dir="reports",
                preferred_signature=signature,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "Benchmark Error", str(exc))
            self.footer_status.setText("Benchmark failed")
            return

        gt_note = "Ground truth used." if summary.ground_truth_available else (
            "No ground-truth CSV found • centroid accuracy is not claimed."
        )
        acquisition = f"{summary.acquisition_time_s:.3f} s" if summary.acquisition_time_s is not None else "—"
        self.footer_status.setText(f"Benchmark complete • {paths[1].name}")
        QMessageBox.information(
            self,
            "MP4 Benchmark Complete",
            f"Source: {Path(video_path).name}\n"
            f"Processed FPS: {summary.fps:.2f}\n"
            f"Avg centre error: {summary.average_tracking_error_px:.2f} px\n"
            f"Max centre error: {summary.maximum_tracking_error_px:.2f} px\n"
            f"Detection loss: {summary.target_loss_percentage:.2f}%\n"
            f"Acquisition: {acquisition}\n"
            f"Re-acquisitions: {summary.successful_reacquisitions}\n\n"
            f"{gt_note}\n\n"
            f"Summary JSON: {paths[0]}\n"
            f"Summary CSV: {paths[1]}\n"
            f"Telemetry CSV: {paths[2]}\n"
            f"Tracked MP4: {paths[3]}",
        )

app = QApplication(sys.argv)
window = FSOCWindow()
window.show()
sys.exit(app.exec())
