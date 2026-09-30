"""Video-first post-run analysis workspace for uploaded-video benchmarks.

Shows the tracked output video as the dominant element (large, horizontal,
aspect-true, fit/zoom), followed by PS compliance, tracking performance,
video/test details, and a tracking-state timeline reconstructed from the
benchmark telemetry CSV.

Honesty rule: metrics that require ground truth are displayed as an em dash
with a clear explanation when ground truth was not provided or was invalid.
They are never fabricated. Only genuinely measurable quantities (detection
availability, lock retention, loss/re-acquisition events, processing speed,
metadata) are shown without ground truth.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF, QTimer, QUrl
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from performance_report import evaluate_ps_requirements


def validate_ground_truth_file(path) -> str:
    """Classify a ground-truth CSV as AVAILABLE / NOT_PROVIDED / INVALID.

    AVAILABLE requires at least one parsable ``frame,x,y`` row. An unreadable
    file, wrong columns, or zero valid rows is INVALID — it is treated exactly
    like no ground truth so the run never reports zero-error "perfection".
    """
    try:
        source = Path(path)
    except (TypeError, ValueError):
        return "NOT_PROVIDED"
    if path is None or not source.exists():
        return "NOT_PROVIDED"
    try:
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"frame", "x", "y"}
            headers = {(h or "").strip().lower() for h in (reader.fieldnames or [])}
            if not required.issubset(headers):
                return "INVALID"
            for row in reader:
                try:
                    int(float(row.get("frame", "")))
                    float(row.get("x", ""))
                    float(row.get("y", ""))
                except (TypeError, ValueError):
                    continue
                return "AVAILABLE"
    except OSError:
        return "INVALID"
    return "INVALID"


STATE_COLORS = {
    "SEARCHING": QColor(96, 112, 128),
    "ACQUIRING": QColor(255, 211, 76),
    "TRACKING": QColor(57, 226, 138),
    "PREDICTING": QColor(255, 179, 71),
    "RE-ACQUIRING": QColor(255, 159, 67),
    "LOST": QColor(255, 107, 129),
}
STATE_LEGEND = ("SEARCHING", "ACQUIRING", "TRACKING", "PREDICTING", "RE-ACQUIRING", "LOST")
STATUS_COLORS = {"PASS": "#43f08f", "FAIL": "#ff8a92", "PENDING": "#f4c967"}


def _na(value, digits=2, suffix=""):
    """Format a metric; missing evidence becomes an em dash, never a number."""
    if value is None:
        return "\u2014"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "\u2014"
    return f"{f:.{digits}f}{suffix}"


def _finite(value):
    """Return float(value) for a real finite measurement, else None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_state_timeline(telemetry_csv) -> tuple:
    """Compress per-frame tracker states into [(start, end, state)] segments."""
    path = Path(telemetry_csv)
    if not path.exists():
        return [], 0
    states = []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    frame = int(float(row.get("frame", "")))
                except (TypeError, ValueError):
                    continue
                state = str(row.get("state", "")).strip().upper()
                states.append((frame, state))
    except OSError:
        return [], 0
    if not states:
        return [], 0

    segments = []
    start, current = states[0][0], states[0][1]
    for frame, state in states[1:]:
        if state != current:
            segments.append((start, frame - 1, current))
            start, current = frame, state
    segments.append((start, states[-1][0], current))
    return segments, states[-1][0]


class TrackedVideoPlayer(QWidget):
    """Responsive aspect-true player for the tracked output video.

    The tracked MP4 already carries the benchmark overlay (state, detection
    box, boresight cross, Kalman prediction ring, concise confidences), so the
    player adds no additional clutter over the video itself.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._capture = None
        self._pixmap = QPixmap()
        self._zoom = 1.0
        self._frame_index = 0
        self._total_frames = 0
        self._fps = 30.0
        self._playing = False
        self._callback = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

    # ----------------------------- public API -----------------------------
    def load(self, video_path) -> bool:
        self.close_video()
        path = Path(video_path)
        if not path.exists():
            return False
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return False
        self._capture = capture
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if (fps > 1.0 and math.isfinite(fps)) else 30.0
        self._total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._frame_index = 0
        self._grab_frame()
        return not self._pixmap.isNull()

    def close_video(self):
        self._timer.stop()
        self._playing = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._pixmap = QPixmap()
        self.update()

    def play_pause(self):
        if self._capture is None:
            return
        if self._playing:
            self._timer.stop()
            self._playing = False
        else:
            if self._frame_index >= max(self._total_frames - 1, 0):
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_index = 0
            self._timer.start(max(1, int(round(1000.0 / self._fps))))
            self._playing = True
        self._sync()

    def seek(self, frame_index):
        if self._capture is None:
            return
        frame_index = max(0, min(int(frame_index), max(self._total_frames - 1, 0)))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._frame_index = frame_index
        self._grab_frame()

    def set_zoom(self, zoom):
        self._zoom = max(1.0, min(3.0, float(zoom)))
        self._sync()

    def zoom(self):
        return self._zoom

    def is_loaded(self):
        return self._capture is not None and not self._pixmap.isNull()

    def is_playing(self):
        return self._playing

    def total_frames(self):
        return self._total_frames

    def current_frame(self):
        return self._frame_index

    def fps(self):
        return self._fps

    # ----------------------------- internals ------------------------------
    def _grab_frame(self):
        if self._capture is None:
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._timer.stop()
            self._playing = False
            self._sync()
            return
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.update()
        self._sync()

    def _next_frame(self):
        if self._capture is None:
            self._timer.stop()
            return
        self._frame_index += 1
        self._grab_frame()
        if self._frame_index >= max(self._total_frames - 1, 0):
            self._timer.stop()
            self._playing = False

    def set_callback(self, callback):
        self._callback = callback

    def _sync(self):
        if self._callback is not None:
            try:
                self._callback()
            except RuntimeError:
                pass

    # ------------------------------ painting ------------------------------
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(1, 3, 5))

        if self._pixmap.isNull():
            painter.setPen(QColor(128, 146, 160))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "TRACKED VIDEO UNAVAILABLE")
            painter.end()
            return

        if self._zoom <= 1.001:
            target = self._pixmap.scaled(
                self.rect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            # Zoom crops around the frame centre; the source is never stretched.
            crop_w = max(8, int(self._pixmap.width() / self._zoom))
            crop_h = max(8, int(self._pixmap.height() / self._zoom))
            x0 = (self._pixmap.width() - crop_w) // 2
            y0 = (self._pixmap.height() - crop_h) // 2
            crop = self._pixmap.copy(x0, y0, crop_w, crop_h)
            target = crop.scaled(self.rect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        x = (self.width() - target.width()) // 2
        y = (self.height() - target.height()) // 2
        painter.drawPixmap(x, y, target)
        painter.end()

    def closeEvent(self, event):
        self.close_video()
        super().closeEvent(event)


def read_error_series(telemetry_csv) -> tuple:
    """Per-frame error series for the graph, read from the benchmark's own
    telemetry CSV. Prefers the ground-truth centroid error; falls back to the
    always-measured center (boresight) error when truth was unavailable.
    Only genuinely measured values are returned — blank cells are skipped,
    never invented. Returns (kind, [(frame, value), ...])."""
    path = Path(telemetry_csv)
    if not path.exists():
        return "", []
    centroid, center = [], []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    frame = int(float(row.get("frame", "")))
                except (TypeError, ValueError):
                    continue
                raw_c = (row.get("centroid_error_px") or "").strip()
                if raw_c:
                    try:
                        centroid.append((frame, float(raw_c)))
                    except ValueError:
                        pass
                raw_b = (row.get("center_error_px") or "").strip()
                if raw_b:
                    try:
                        center.append((frame, float(raw_b)))
                    except ValueError:
                        pass
    except OSError:
        return "", []
    if centroid:
        return "CENTROID (GT) ERROR", centroid
    if center:
        return "CENTER (BORESIGHT) ERROR", center
    return "", []


class ErrorSeriesWidget(QWidget):
    """Large readable CENTER-ERROR-vs-FRAME plot of the measured series."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series = []
        self.setMinimumHeight(150)

    def set_series(self, series, kind=""):
        self._series = list(series)
        self._kind = str(kind)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(9, 17, 25))
        painter.setPen(QPen(QColor(27, 43, 59), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(158, 176, 191))
        title = f"{self._kind or 'ERROR'} vs FRAME  (measured)" if self._series \
            else "ERROR vs FRAME — no measured series for this run"
        painter.drawText(10, 16, title)

        r = self.rect().adjusted(38, 26, -10, -22)
        if not self._series:
            painter.setPen(QColor(115, 133, 150))
            painter.drawText(r, Qt.AlignCenter,
                             "No ground-truth error series for this run")
            painter.end()
            return
        frames = [f for f, _ in self._series]
        values = [v for _, v in self._series]
        f0, f1 = min(frames), max(frames)
        vmin, vmax = min(values), max(values)
        if vmax - vmin < 1e-6:
            pad = max(1.0, abs(vmax) * 0.15)
        else:
            pad = (vmax - vmin) * 0.12
        vmin -= pad
        vmax += pad
        span_f = max(f1 - f0, 1)

        painter.setPen(QPen(QColor(39, 58, 74), 1))
        for i in range(1, 4):
            y = r.top() + r.height() * i / 4.0
            painter.drawLine(r.left(), int(y), r.right(), int(y))

        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QColor(115, 133, 150))
        painter.drawText(4, r.top() + 5, f"{vmax:.1f}")
        painter.drawText(4, r.bottom(), f"{vmin:.1f}")
        painter.drawText(r.left(), self.height() - 6, f"frame {f0}")
        painter.drawText(r.right() - 46, self.height() - 6, f"frame {f1}")

        painter.setPen(QPen(QColor(255, 216, 61), 2))
        previous = None
        for frame, value in self._series:
            x = r.left() + r.width() * (frame - f0) / span_f
            y = r.bottom() - ((value - vmin) / max(vmax - vmin, 1e-9)) * r.height()
            point = QPointF(int(x), int(y))
            if previous is not None:
                painter.drawLine(previous, point)
            previous = point
        painter.end()


class StateTimelineWidget(QWidget):
    """Horizontal tracking-state timeline built from telemetry states."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._segments = []
        self._total = 0
        self.setMinimumHeight(58)

    def set_timeline(self, segments, total_frames):
        self._segments = segments
        self._total = int(total_frames)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(9, 17, 25))
        painter.setPen(QPen(QColor(27, 43, 59), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(158, 176, 191))
        painter.drawText(10, 16, "TRACKING STATE TIMELINE")

        bar = self.rect().adjusted(10, 26, -10, -30)
        if not self._segments or self._total <= 0:
            painter.setPen(QColor(115, 133, 150))
            painter.drawText(bar, Qt.AlignCenter, "Timeline unavailable for this run")
            painter.end()
            return

        for start, end, state in self._segments:
            x0 = bar.x() + bar.width() * (start - 1) / max(self._total, 1)
            x1 = bar.x() + bar.width() * max(end, start) / max(self._total, 1)
            color = STATE_COLORS.get(state, QColor(120, 130, 140))
            painter.fillRect(int(x0), bar.y(), max(1, int(x1 - x0)), bar.height(), color)

        x = 12
        y = self.height() - 10
        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
        for state in STATE_LEGEND:
            painter.setPen(QPen(STATE_COLORS[state], 6))
            painter.drawLine(x, y - 3, x + 10, y - 3)
            painter.setPen(QColor(150, 172, 185))
            painter.drawText(x + 14, y, state)
            x += 26 + len(state) * 5
        painter.end()


class _Card(QFrame):
    def __init__(self, label, value="\u2014", sub="", parent=None):
        super().__init__(parent)
        self.setObjectName("metric_card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        name = QLabel(label)
        name.setObjectName("metric_label")
        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("metric_value")
        layout.addWidget(name)
        layout.addWidget(self.value_label)
        if sub:
            sub_label = QLabel(sub)
            sub_label.setObjectName("metric_label")
            layout.addWidget(sub_label)


class VideoAnalysisWindow(QMainWindow):
    """Post-run benchmark workstation for one uploaded-video analysis."""

    def __init__(self, summary, paths, video_path, gt_status="NOT PROVIDED", parent=None):
        super().__init__(parent)
        self._summary = summary
        self._paths = [Path(p) for p in paths]
        self._video_path = Path(video_path)
        self._gt_status = str(gt_status).upper()
        self._main = parent

        self.setWindowTitle("ASTRALINK — Analyze External Video")
        self.resize(1480, 940)
        self.setMinimumSize(1040, 700)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # ---------------- Header + actions ----------------
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("ASTRALINK — ANALYZE EXTERNAL VIDEO (SAME DETECTION / KALMAN PIPELINE)")
        title.setObjectName("analysis_title")
        subtitle = QLabel(f"{self._video_path.name}  •  {self._gt_banner_text()}")
        subtitle.setObjectName("analysis_subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.start_button = QPushButton("▶  START / PLAY")
        self.start_button.setObjectName("primary_button")
        self.start_button.setMinimumHeight(34)
        self.start_button.setMinimumWidth(130)
        self.open_tracked_button = self._button("⧉  OPEN TRACKED VIDEO")
        self.open_pdf_button = self._button("OPEN REPORT PDF")
        self.open_folder_button = self._button("OPEN REPORT FOLDER")
        self.close_button = self._button("CLOSE / BACK")
        for button in (
            self.start_button, self.open_tracked_button, self.open_pdf_button,
            self.open_folder_button, self.close_button,
        ):
            header.addWidget(button)
        root.addLayout(header)

        self.start_button.clicked.connect(self._toggle_play)
        self.open_tracked_button.clicked.connect(self._open_tracked_video)
        self.open_pdf_button.clicked.connect(self._open_pdf)
        self.open_folder_button.clicked.connect(self._open_folder)
        self.close_button.clicked.connect(self.close)

        # ---------------- Video-first viewport (~70% height) ----------------
        video_panel = QFrame()
        video_panel.setObjectName("panel")
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(10, 8, 10, 8)
        video_layout.setSpacing(6)

        player_head = QHBoxLayout()
        player_title = QLabel("TRACKED OUTPUT VIDEO")
        player_title.setObjectName("panel_title")
        player_head.addWidget(player_title)
        player_head.addStretch()
        self.position_label = QLabel("FRAME 0")
        self.position_label.setObjectName("caption")
        player_head.addWidget(self.position_label)
        video_layout.addLayout(player_head)

        self.player = TrackedVideoPlayer()
        self.player.set_callback(self._sync_player_controls)
        video_layout.addWidget(self.player, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("▶  PLAY")
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 0)
        self.scrub.sliderMoved.connect(self.player.seek)
        controls.addWidget(self.scrub, 1)
        zoom_out = QPushButton("−")
        zoom_out.setObjectName("zoom_button")
        zoom_out.clicked.connect(lambda: self.player.set_zoom(self.player.zoom() - 0.5))
        controls.addWidget(zoom_out)
        zoom_fit = QPushButton("FIT 1×")
        zoom_fit.setObjectName("zoom_button")
        zoom_fit.clicked.connect(lambda: self.player.set_zoom(1.0))
        controls.addWidget(zoom_fit)
        zoom_in = QPushButton("+")
        zoom_in.setObjectName("zoom_button")
        zoom_in.clicked.connect(lambda: self.player.set_zoom(self.player.zoom() + 0.5))
        controls.addWidget(zoom_in)
        self.zoom_label = QLabel("1.0×")
        self.zoom_label.setObjectName("caption")
        controls.addWidget(self.zoom_label)
        root.addWidget(video_panel, 7)

        # ---------------- Compact secondary information ----------------
        # VIDEO → TRACKING → ERROR → HANDOFF priority: everything below the
        # video lives in one height-clamped scrollable zone so it can never
        # squeeze the video viewport out of its dominant ~70% share.
        self._secondary = QWidget()
        secondary_layout = QVBoxLayout(self._secondary)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(6)

        self._secondary_scroll = QScrollArea()
        self._secondary_scroll.setWidgetResizable(True)
        self._secondary_scroll.setFrameShape(QFrame.NoFrame)
        self._secondary_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._secondary_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._secondary_scroll.setMinimumHeight(132)
        self._secondary_scroll.setMaximumHeight(196)
        self._secondary_scroll.setWidget(self._secondary)
        root.addWidget(self._secondary_scroll, 0)

        # ---------------- Handoff strip (bottom band, compact) ----------------
        handoff_strip = QFrame()
        handoff_strip.setObjectName("handoff_strip")
        handoff_layout = QVBoxLayout(handoff_strip)
        handoff_layout.setContentsMargins(10, 5, 10, 5)
        handoff_layout.setSpacing(1)
        self._handoff_status = QLabel()
        self._handoff_evidence = QLabel()
        self._handoff_evidence.setObjectName("handoff_evidence")
        handoff_layout.addWidget(self._handoff_status)
        handoff_layout.addWidget(self._handoff_evidence)
        root.addWidget(handoff_strip, 0)

        # ---------------- PS compliance strip ----------------
        ps_panel = QFrame()
        ps_panel.setObjectName("panel")
        ps_layout = QVBoxLayout(ps_panel)
        ps_layout.setContentsMargins(10, 8, 10, 8)
        ps_layout.setSpacing(4)
        ps_title = QLabel("PS COMPLIANCE — MEASURED | REQUIRED | STATUS")
        ps_title.setObjectName("panel_title")
        ps_layout.addWidget(ps_title)
        self.ps_grid = QGridLayout()
        self.ps_grid.setHorizontalSpacing(8)
        self.ps_grid.setVerticalSpacing(0)
        ps_layout.addLayout(self.ps_grid)
        secondary_layout.addWidget(ps_panel)

        # ---------------- Performance + details ----------------
        info_row = QHBoxLayout()
        perf_panel = QFrame()
        perf_panel.setObjectName("panel")
        perf_layout = QVBoxLayout(perf_panel)
        perf_layout.setContentsMargins(10, 8, 10, 8)
        perf_layout.setSpacing(4)
        perf_title = QLabel("TRACKING PERFORMANCE")
        perf_title.setObjectName("panel_title")
        perf_layout.addWidget(perf_title)
        perf_grid = QGridLayout()
        perf_grid.setHorizontalSpacing(8)
        perf_grid.setVerticalSpacing(6)
        perf_layout.addLayout(perf_grid)
        self.gt_note = QLabel()
        self.gt_note.setObjectName("gt_note")
        self.gt_note.setWordWrap(True)
        perf_layout.addWidget(self.gt_note)
        info_row.addWidget(perf_panel, 3)

        details_panel = QFrame()
        details_panel.setObjectName("panel")
        details_layout = QVBoxLayout(details_panel)
        details_layout.setContentsMargins(10, 8, 10, 8)
        details_layout.setSpacing(4)
        details_title = QLabel("VIDEO / TEST DETAILS")
        details_title.setObjectName("panel_title")
        details_layout.addWidget(details_title)
        self.details_label = QLabel()
        self.details_label.setObjectName("analysis_details")
        self.details_label.setWordWrap(True)
        details_layout.addWidget(self.details_label)
        info_row.addWidget(details_panel, 2)
        secondary_layout.addLayout(info_row)

        # ---------------- Timeline ----------------
        timeline_panel = QFrame()
        timeline_panel.setObjectName("panel")
        timeline_layout = QVBoxLayout(timeline_panel)
        timeline_layout.setContentsMargins(10, 8, 10, 8)
        self.timeline = StateTimelineWidget()
        timeline_layout.addWidget(self.timeline)
        secondary_layout.addWidget(timeline_panel)

        # ---------------- Error vs frame (display-only) ----------------
        # Plots the per-frame error series recorded by the SAME benchmark run
        # (its telemetry CSV). No values are recomputed, invented or altered.
        error_panel = QFrame()
        error_panel.setObjectName("panel")
        error_layout = QVBoxLayout(error_panel)
        error_layout.setContentsMargins(10, 8, 10, 8)
        self.error_graph = ErrorSeriesWidget()
        error_layout.addWidget(self.error_graph)
        secondary_layout.addWidget(error_panel)

        self._apply_style()

        loaded = False
        tracked = self._tracked_path()
        if tracked is not None:
            loaded = self.player.load(tracked)
        if not loaded:
            self.position_label.setText("TRACKED VIDEO UNAVAILABLE")
        self.start_button.setEnabled(loaded)
        self.play_button.setEnabled(loaded)
        self.scrub.setEnabled(loaded)
        if loaded:
            self.position_label.setText("READY — PRESS START / PLAY")
        self._sync_player_controls()

        self._populate_performance(perf_grid)
        self._populate_ps_compliance()
        self._populate_details()
        self._populate_timeline()
        self._populate_handoff()

    # ------------------------------------------------------------------
    def _button(self, text):
        button = QPushButton(text)
        button.setMinimumHeight(30)
        return button

    def _tracked_path(self):
        if len(self._paths) > 3:
            tracked = Path(self._paths[3])
            if tracked.exists():
                return tracked
        return None

    def _gt_banner_text(self):
        if self._gt_status == "AVAILABLE":
            return "GROUND TRUTH: AVAILABLE — accuracy metrics measured"
        if self._gt_status == "INVALID":
            return "GROUND TRUTH: INVALID (no usable frame,x,y rows) — accuracy metrics not calculated"
        return "GROUND TRUTH: NOT PROVIDED — centroid/tracking accuracy was not calculated"

    def _populate_performance(self, grid):
        s = self._summary
        gt = bool(getattr(s, "ground_truth_available", False)) and self._gt_status == "AVAILABLE"

        loss_value = _finite(s.target_loss_percentage)
        detected_pct = (
            100.0 * max(
                int(s.frames_processed) - int(round(loss_value * int(s.frames_processed) / 100.0)), 0
            ) / max(int(s.frames_processed), 1)
            if loss_value is not None else None
        )
        reacq = s.maximum_reacquisition_time_s or s.average_reacquisition_time_s
        meta = getattr(s, "metadata", {}) or {}
        forced_reacq = _finite(meta.get("forced_reacquisition_time_s"))
        if reacq is None:
            reacq = forced_reacq  # measured forced-loss rehearsal time
        reacq_sub = (
            f"forced-loss measured" if forced_reacq is not None
            else f"{int(s.successful_reacquisitions)} successful"
        )
        cards = [
            ("Center Error (Final)", _na(s.final_center_error_px, 2, " px"),
             "pointing converged" if s.center_error_converged else "boresight offset"),
            ("Detection Rate", _na(detected_pct, 2, " %"), "detection availability"),
            ("Tracking Error (GT)", _na(s.average_tracking_error_px, 2, " px") if gt else "\u2014",
             "mean detection vs truth"),
            ("Average Error (GT)", _na(s.average_tracking_error_px, 2, " px") if gt else "\u2014",
             "mean detection vs truth"),
            ("Max Deviation (GT)", _na(s.maximum_tracking_error_px, 2, " px") if gt else "\u2014",
             "max detection vs truth"),
            ("RMSE (GT)", _na(s.rmse_tracking_error_px, 2, " px") if gt else "\u2014",
             "rms detection vs truth"),
            ("Centroid Max (GT)", _na(s.maximum_centroid_error_px, 2, " px") if gt else "\u2014",
             "max centroid vs truth"),
            ("Target Loss", _na(s.target_loss_percentage, 2, " %"), "detection misses/evaluated"),
            ("Lock Retention", _na(s.lock_retention_percentage, 2, " %"), None),
            ("Re-acquisition", _na(reacq, 2, " s"), reacq_sub),
            ("Processing FPS", _na(s.fps, 1), None),
            ("Processing Time", _na(s.duration_s, 2, " s"), f"avg {_na(s.average_processing_time_ms)} ms/frame"),
            ("Frames Processed", str(int(s.frames_processed)), None),
        ]

        for index, (label, value, sub) in enumerate(cards):
            row, col = divmod(index, 6)
            grid.addWidget(_Card(label, value, sub or ""), row, col)

        if gt:
            self.gt_note.setText(
                "Ground truth was supplied: tracking/centroid accuracy is measured from the provided frame,x,y data."
            )
            self.gt_note.setStyleSheet("color:#67e9a0; font-size:9px; font-weight:700;")
        else:
            self.gt_note.setText(
                "Ground truth unavailable — centroid/tracking accuracy was not calculated. "
                "Detection-based metrics above are genuine; accuracy fields show \u2014 on purpose."
            )

    def _populate_ps_compliance(self):
        s = self._summary
        gt = self._gt_status == "AVAILABLE"
        checks = evaluate_ps_requirements(s)["checks"]

        acq = checks.get("acquisition_time", {})
        track = checks.get("tracking_error", {})
        loss = checks.get("target_loss", {})
        reacq_check = checks.get("reacquisition_time", {})
        speed = checks.get("processing_speed", {})

        # Ground-truth-dependent accuracy is never a PASS without valid truth,
        # and the displayed value is the real measured maximum deviation.
        measured_max_dev = _finite(s.maximum_tracking_error_px)
        track_status = str(track.get("status", "PENDING")) if (gt and measured_max_dev is not None) else "PENDING"
        rows = [
            ("Acquisition Time", _na(acq.get("value"), 2, " s"),
             str(acq.get("requirement", "—")), str(acq.get("status", "PENDING"))),
            ("Max Deviation", _na(measured_max_dev, 2, " px") if (gt and measured_max_dev is not None) else "\u2014",
             str(track.get("requirement", "—")), track_status),
            ("Target Loss", _na(loss.get("value"), 2, " %"),
             str(loss.get("requirement", "—")), str(loss.get("status", "PENDING"))),
            ("Re-acquisition", _na(reacq_check.get("value"), 2, " s"),
             str(reacq_check.get("requirement", "—")), str(reacq_check.get("status", "PENDING"))),
            ("Processing Speed", _na(speed.get("value"), 1, " FPS"),
             str(speed.get("requirement", "—")), str(speed.get("status", "PENDING"))),
        ]

        for index, (param, measured, requirement, status) in enumerate(rows):
            cell = QFrame()
            cell.setObjectName("metric_card")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(8, 4, 8, 4)
            cell_layout.setSpacing(0)
            name = QLabel(param)
            name.setObjectName("metric_label")
            value = QLabel(f"{measured}   |   {requirement}")
            value.setObjectName("metric_value")
            badge = QLabel(status)
            badge.setFont(QFont("Segoe UI", 8, QFont.Bold))
            badge.setStyleSheet(f"color:{STATUS_COLORS.get(status, '#f4c967')};")
            cell_layout.addWidget(name)
            cell_layout.addWidget(value)
            cell_layout.addWidget(badge)
            self.ps_grid.addWidget(cell, 0, index)

    def _populate_handoff(self):
        """COARSE→FINE handoff readiness, derived only from real measurements.

        Ready requires: target acquired, stable tracking evidence, tracking
        error within the PS limit (valid ground truth), the measured final
        CENTER ERROR within the PS limit (the controller must actually have
        converged the pointing), and target loss within the PS limit. Missing
        evidence degrades the status honestly instead of faking readiness.
        """
        s = self._summary
        meta = getattr(s, "metadata", {}) or {}
        reasons = []
        ready = True

        acquired = s.acquisition_time_s is not None
        ready &= acquired
        reasons.append(
            f"TARGET ACQUIRED ({_na(s.acquisition_time_s, 2, ' s')})" if acquired
            else "TARGET NOT ACQUIRED"
        )

        evaluated = int(meta.get("evaluated_frames", 0) or 0)
        stable = evaluated >= 30
        ready &= stable
        reasons.append(
            f"STABLE TRACKING ({evaluated} EVALUATED FRAMES)" if stable
            else f"INSUFFICIENT TRACKING EVIDENCE ({evaluated} EVALUATED FRAMES)"
        )

        max_dev = _finite(s.maximum_tracking_error_px)
        gt_ok = self._gt_status == "AVAILABLE" and max_dev is not None
        error_ok = gt_ok and max_dev <= 10.0
        ready &= error_ok
        reasons.append(
            f"MAX DEVIATION {_na(max_dev, 2, ' px')} (LIMIT 10 px)" if gt_ok
            else "ACCURACY UNVERIFIED — NO VALID GROUND TRUTH"
        )

        # Center error: the real measured final pointing offset from the
        # closed-loop coarse-pointing run. Legacy summaries without pointing
        # evidence cannot prove convergence and honestly block handoff.
        center = _finite(s.final_center_error_px)
        center_ok = center is not None and center <= 10.0
        ready &= center_ok
        if center is None:
            reasons.append("CENTER ERROR UNVERIFIED — NO POINTING CONVERGENCE EVIDENCE")
        elif center_ok:
            reasons.append(f"CENTER ERROR {center:.2f} px (LIMIT 10 px)")
        else:
            reasons.append(f"CENTER ERROR {center:.2f} px EXCEEDS 10 px LIMIT")

        loss_pct = _finite(s.target_loss_percentage)
        loss_ok = loss_pct is not None and loss_pct < 5.0
        ready &= loss_ok
        if loss_pct is None:
            reasons.append("TARGET LOSS UNAVAILABLE — NO POST-LOCK EVALUATED FRAMES")
        elif loss_ok:
            reasons.append(
                f"TARGET LOSS {loss_pct:.2f}% (LIMIT <5%, {int(s.target_loss_events)} EVENTS)"
            )
        else:
            reasons.append(
                f"TARGET LOSS {loss_pct:.2f}% MEETS/EXCEEDS 5% LIMIT ({int(s.target_loss_events)} EVENTS)"
            )

        # Real measured re-acquisition time (forced-loss rehearsal in the
        # benchmark engine, or observed loss/re-acquisition events).
        reacq_s = _finite(meta.get("forced_reacquisition_time_s"))
        if reacq_s is None:
            reacq_s = _finite(s.maximum_reacquisition_time_s) or _finite(s.average_reacquisition_time_s)
        reacq_note = (
            f"RE-ACQUISITION {reacq_s:.2f} s ({'WITHIN' if reacq_s <= 1.0 else 'EXCEEDS'} 1 s LIMIT)"
            if reacq_s is not None
            else "RE-ACQUISITION NOT EXERCISED THIS RUN"
        )

        centered = center_ok and error_ok
        if acquired and stable and centered:
            coarse = "COARSE ALIGNMENT: LOCKED"
        elif acquired and stable:
            coarse = "COARSE ALIGNMENT: TRACKING"
        else:
            coarse = "COARSE ALIGNMENT: NOT LOCKED"

        if ready:
            status, color = "HANDOFF TO FINE ALIGNMENT: READY", "#43f08f"
        elif not acquired:
            status, color = "HANDOFF: NOT READY — TARGET NOT ACQUIRED", "#ff8a92"
        elif not stable:
            status, color = "HANDOFF: NOT READY — INSUFFICIENT TRACKING EVIDENCE", "#ff8a92"
        elif not gt_ok:
            status, color = "HANDOFF: NOT READY — ACCURACY UNVERIFIED (NO VALID GROUND TRUTH)", "#f4c967"
        elif not center_ok:
            status, color = (
                f"HANDOFF: NOT READY — CENTER ERROR {center:.2f} px > 10 px"
                if center is not None
                else "HANDOFF: NOT READY — CENTER ERROR UNVERIFIED"
            ), "#ff8a92" if center is not None else "#f4c967"
        elif loss_pct is None:
            status, color = "HANDOFF: NOT READY — TARGET LOSS UNVERIFIED", "#f4c967"
        else:
            status, color = "HANDOFF: NOT READY — ACCURACY OUTSIDE PS LIMIT", "#ff8a92"

        self._handoff_status.setText(f"{coarse}   •   {status}")
        self._handoff_status.setStyleSheet(
            f"color:{color}; font-size:10px; font-weight:800; letter-spacing:1px;"
        )
        self._handoff_evidence.setText("   |   ".join(reasons + [reacq_note]))

    def _populate_details(self):
        s = self._summary
        meta = getattr(s, "metadata", {}) or {}
        source_fps = meta.get("source_fps")
        fps_value = float(source_fps) if isinstance(source_fps, (int, float)) and source_fps else None
        duration = s.frames_processed / fps_value if fps_value else None
        truth_path = meta.get("ground_truth_path")
        line_one = (
            f"VIDEO: {self._video_path.name}   •   {meta.get('source_width', '?')}×{meta.get('source_height', '?')}"
            f"   •   SOURCE {source_fps if source_fps is not None else '—'} FPS"
            f"   •   DURATION {_na(duration, 2, ' s')}"
            f"   •   FRAMES {meta.get('frame_count_hint', '—')} HINT / {int(s.frames_processed)} PROCESSED"
        )
        line_two = (
            f"TEST: GROUND TRUTH {self._gt_status}"
            f"{(' (' + Path(truth_path).name + ')') if truth_path else ''}"
            f"   •   SIGNATURE GATE {meta.get('preferred_signature', 'AUTO')}"
            f"   •   LOSS EVENTS {int(s.target_loss_events)}"
            f"   •   RE-ACQUISITIONS {int(s.successful_reacquisitions)}"
            f"   •   REPORT JSON {self._paths[0].name if self._paths else '—'}"
        )
        self.details_label.setText(line_one + "\n" + line_two)

    def _populate_timeline(self):
        if len(self._paths) > 2:
            segments, total = read_state_timeline(self._paths[2])
            self.timeline.set_timeline(segments, total)
        else:
            self.timeline.set_timeline([], 0)
        if len(self._paths) > 2:
            kind, series = read_error_series(self._paths[2])
            self.error_graph.set_series(series, kind)
        else:
            self.error_graph.set_series([])

    # ------------------------------------------------------------------
    def _toggle_play(self):
        self.player.play_pause()
        self._sync_player_controls()

    def _sync_player_controls(self):
        if self.player.is_loaded():
            self.scrub.setMaximum(max(0, self.player.total_frames() - 1))
            self.scrub.blockSignals(True)
            self.scrub.setValue(self.player.current_frame())
            self.scrub.blockSignals(False)
            if self.player.is_playing():
                self.position_label.setText(
                    f"FRAME {self.player.current_frame()} / {max(self.player.total_frames() - 1, 0)}"
                    f"  •  {self.player.fps():.1f} FPS"
                )
            elif not self.player.is_playing() and self.player.current_frame() <= 0:
                self.position_label.setText("READY — PRESS START / PLAY")
            else:
                self.position_label.setText(
                    f"PAUSED @ FRAME {self.player.current_frame()} / {max(self.player.total_frames() - 1, 0)}"
                )
            playing = self.player.is_playing()
            self.start_button.setText("⏸  PAUSE" if playing else "▶  START / PLAY")
            self.play_button.setText("⏸  PAUSE" if playing else "▶  PLAY")
        self.zoom_label.setText(f"{self.player.zoom():.2f}×")

    def _open_tracked_video(self):
        tracked = self._tracked_path()
        if tracked is not None:
            self._open(tracked)
        else:
            QMessageBox.warning(self, "Not Available", "Tracked video file was not produced for this run.")

    def _open_pdf(self):
        candidates = [p for p in self._paths if p is not None and p.exists()]
        if candidates:
            self._open(candidates[0])
        else:
            QMessageBox.warning(self, "Not Available", "Report PDF not found for this run.")

    def _open_folder(self):
        existing = [p.parent for p in self._paths if p is not None and p.parent.exists()]
        target = existing[0] if existing else Path("reports").resolve()
        self._open(target)

    def _open(self, target):
        handler = getattr(self._main, "_open_path", None)
        if handler is not None:
            handler(target)
            return
        resolved = Path(target).resolve()
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved))):
            import os
            if hasattr(os, "startfile"):
                try:
                    os.startfile(str(resolved))
                except OSError as exc:
                    QMessageBox.warning(self, "Cannot Open", f"Could not open:\n{resolved}\n\n{exc}")
            else:
                QMessageBox.warning(self, "Cannot Open", f"Could not open:\n{resolved}")

    def closeEvent(self, event):
        self.player.close_video()
        super().closeEvent(event)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #05080d; color: #e7edf3; font-family: "Segoe UI"; }
            #panel { background-color: #0d151f; border: 1px solid #233545; border-radius: 10px; }
            #analysis_title { font-size: 15px; font-weight: 800; color: #f5f7fa; }
            #analysis_subtitle { font-size: 9px; color: #8293a5; }
            #panel_title { font-size: 11px; font-weight: 700; color: #f0f4f8; }
            #handoff_strip { background-color: #0b141d; border: 1px solid #1a2a39; border-radius: 6px; }
            #handoff_evidence { font-size: 8px; color: #8294a5; }
            #caption { font-size: 8px; color: #8294a5; }
            #metric_card { background-color: #0b141d; border: 1px solid #1a2a39; border-radius: 6px; }
            #metric_label { font-size: 7px; color: #738596; }
            #metric_value { font-size: 13px; font-weight: 700; color: #eef3f7; }
            #analysis_details { font-size: 8px; color: #b6c2cc; }
            #primary_button {
                min-height: 34px; background-color: #0f4d2e; border: 2px solid #43f08f;
                border-radius: 8px; padding: 6px 14px; font-size: 11px; font-weight: 800;
                color: #eafff2;
            }
            #primary_button:hover { background-color: #14663c; }
            #primary_button:disabled { border-color: #2a3a48; color: #5a6b78; background-color: #0d151f; }
            #gt_note { font-size: 9px; font-weight: 700; color: #f4c967; }
            QPushButton {
                min-height: 28px; background-color: #16232e; border: 1px solid #3a4e5e;
                border-radius: 6px; padding: 4px 8px; font-size: 9px; font-weight: 700; color: #e7edf3;
            }
            QPushButton:hover { background-color: #22323e; }
            #zoom_button { min-width: 34px; }
            QSlider::groove:horizontal { height: 4px; background: #304252; border-radius: 2px; }
            QSlider::handle:horizontal { width: 12px; margin: -4px 0; border-radius: 6px; background: #e86767; }
        """)
