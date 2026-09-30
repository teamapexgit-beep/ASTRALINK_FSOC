"""
Benchmark Graph Panel Widget

Displays time-series graphs from benchmark telemetry CSV data.
Shows 5 graphs: tracking error, position vs prediction, pan/tilt, FPS, confidence.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPen,
)


# ---------------------------------------------------------------------------
# Shared geometry for the graph cards.  Every title is MEASURED at paint time
# and wrapped/elided inside this padding, so a title can never overflow its
# card, overlap the plot area, or be clipped — at any widget width.
# ---------------------------------------------------------------------------
_GRAPH_PAD_X = 8
_GRAPH_TITLE_TOP = 4
_GRAPH_TITLE_GAP = 3
_GRAPH_TITLE_MAX_LINES = 3

# A long live run keeps only the most recent window of REAL samples, so the
# canvas stays cheap to repaint (a rolling window — older real samples leave,
# nothing is ever fabricated). The video benchmark (a few hundred frames) is
# far below this and therefore keeps its complete series.
_MAX_LIVE_SAMPLES = 900
_MAX_LIVE_SLACK = 128


def _finite(value):
    """Return ``value`` when it is a real finite number, else ``None``.

    Guards every value on its way into a graph: NaN, +Inf, -Inf, strings,
    None and arbitrary objects all become ``None`` (a gap), so a malformed
    row can never reach QPainter and blow up or draw garbage.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _finite_series(values):
    """Map ``_finite`` over a sequence, preserving its length."""
    return [_finite(v) for v in values]
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class BenchmarkGraphPanel(QWidget):
    """
    Comprehensive graph panel for benchmark video analysis results.
    
    Displays 5 time-series graphs from telemetry CSV data:
    1. Tracking / Centroid Error vs Time
    2. Target Position vs Kalman Predicted Position
    3. Pan / Tilt Control Response vs Time
    4. FPS / Processing Performance vs Time
    5. Tracking Confidence vs Time
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Small minimum: the panel re-flows its own graph grid (3 / 2 / 1
        # columns) as it narrows, so it is usable inside the control column
        # and on a tablet without ever forcing horizontal scrolling.
        self.setMinimumSize(240, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        self.telemetry_data = None
        self.timestamps = []
        self.frames = []
        
        # Data arrays for graphing
        self.center_errors = []
        self.centroid_errors = []
        self.target_x = []
        self.target_y = []
        self.predicted_x = []
        self.predicted_y = []
        self.pan_values = []
        self.tilt_values = []
        self.fps_values = []
        self.filter_confidence = []
        self.detector_confidence = []
        self.ai_confidence = []
        self.states = []
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(8)
        
        self.title = QLabel("BENCHMARK PERFORMANCE GRAPHS")
        self.title.setObjectName("panel_title")
        self.layout.addWidget(self.title)
        
        self.status_label = QLabel(
            "Run Benchmark Video to generate performance data."
        )
        self.status_label.setObjectName("caption")
        self.status_label.setWordWrap(True)
        self.layout.addWidget(self.status_label)
        
        # Create 5 graph widgets in a scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(160)
        
        self.graph_container = QWidget()
        self.graph_layout = QGridLayout(self.graph_container)
        self.graph_layout.setContentsMargins(8, 8, 8, 8)
        self.graph_layout.setSpacing(10)
        self.graph_layout.setHorizontalSpacing(12)
        
        scroll.setWidget(self.graph_container)
        self._scroll = scroll
        self.layout.addWidget(scroll, 1)
        
        # Create 5 graph widgets (2x3 grid, last cell empty)
        self.graph_widgets = []
        graph_configs = [
            ("TRACKING / CENTROID ERROR vs TIME", "px", _BenchmarkSingleGraph._draw_error_graph),
            ("TARGET POSITION vs KALMAN PREDICTION", "px", _BenchmarkSingleGraph._draw_position_graph),
            ("PAN / TILT CONTROL RESPONSE vs TIME", "deg", _BenchmarkSingleGraph._draw_pan_tilt_graph),
            ("FPS / PROCESSING PERFORMANCE vs TIME", "FPS", _BenchmarkSingleGraph._draw_fps_graph),
            ("TRACKING CONFIDENCE vs TIME", "", _BenchmarkSingleGraph._draw_confidence_graph),
        ]
        
        for idx, (title, unit, draw_fn) in enumerate(graph_configs):
            graph_widget = _BenchmarkSingleGraph(title, unit, draw_fn)
            self.graph_widgets.append(graph_widget)
        self._graph_columns = None
        self._relayout_graphs(3)
        
        # Live refresh timer — keeps graphs live, updating every second,
        # without freezing. Uses QTimer with a standing empty callback to
        # schedule a repaint cycle; no logic layer, no blocking.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(1000)
        self._refresh_timer.timeout.connect(self._on_refresh_tick)
        self._refresh_timer.start()
    
    def refresh(self):
        """Explicit public refresh call (e.g. when new benchmark data arrives).

        Repaints all graph widgets immediately without waiting for the
        next timer tick. Safe to call from any thread context the GUI owns.
        """
        self.update()
    
    def resizeEvent(self, event):
        """Handle resize without glitches: refresh layout and redraw."""
        super().resizeEvent(event)
        # Responsive grid: keep the graphs wide enough to stay readable.
        # Use the panel's own width (the scroll viewport is still stale while
        # a resize is being processed, which used to leave the grid one
        # density step behind).
        width = max(1, self.width())
        try:
            if width >= 900:
                self._relayout_graphs(3)
            elif width >= 600:
                self._relayout_graphs(2)
            else:
                self._relayout_graphs(1)
        except Exception:
            pass
        # Force redraw on resize so axes/lines scale correctly
        self.update()

    def _relayout_graphs(self, columns):
        """Re-flow the five graphs into ``columns`` columns (3 / 2 / 1)."""
        columns = max(1, min(5, int(columns)))
        if self._graph_columns == columns:
            return
        self._graph_columns = columns
        for graph in self.graph_widgets:
            self.graph_layout.removeWidget(graph)
        for index, graph in enumerate(self.graph_widgets):
            self.graph_layout.addWidget(graph, index // columns, index % columns)
        for col in range(5):
            self.graph_layout.setColumnStretch(col, 1 if col < columns else 0)
    
    
    def _on_refresh_tick(self):
        """Sync point: receives every 1000 ms; schedules a repaint.

        Deliberately does nothing else: no re-reading of files, no state
        mutation, no signal emission. The paintEvent() redraws the
        current in-memory telemetry arrays on the widget's event loop.
        This keeps the graphs live and error-free.
        """
        self.update()
    
    def load_telemetry(self, csv_path: str) -> bool:
        """
        Load telemetry CSV data and prepare for graphing.
        
        Args:
            csv_path: Path to the telemetry CSV file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(csv_path)
        if not path.exists():
            self.status_label.setText(f"ERROR: Telemetry file not found: {csv_path}")
            return False
        
        try:
            self.timestamps = []
            self.frames = []
            self.center_errors = []
            self.centroid_errors = []
            self.target_x = []
            self.target_y = []
            self.predicted_x = []
            self.predicted_y = []
            self.pan_values = []
            self.tilt_values = []
            self.fps_values = []
            self.filter_confidence = []
            self.detector_confidence = []
            self.ai_confidence = []
            self.states = []
            
            with path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # _finite() turns NaN / +-Inf / junk into None (a gap),
                        # so a malformed row can never reach QPainter as a NaN.
                        self.timestamps.append(_finite(row['timestamp_s']) or 0.0)
                        self.frames.append(int(float(row['frame'])))
                        
                        # Tracking error
                        self.center_errors.append(_finite(row.get('center_error_px', '')))
                        
                        # Centroid error
                        self.centroid_errors.append(_finite(row.get('centroid_error_px', '')))
                        
                        # Target position
                        self.target_x.append(_finite(row.get('target_x', '')))
                        self.target_y.append(_finite(row.get('target_y', '')))
                        
                        # Predicted position
                        self.predicted_x.append(_finite(row.get('predicted_x', '')))
                        self.predicted_y.append(_finite(row.get('predicted_y', '')))
                        
                        # Pan/Tilt
                        self.pan_values.append(_finite(row.get('pan_deg', '')))
                        self.tilt_values.append(_finite(row.get('tilt_deg', '')))
                        
                        # FPS
                        self.fps_values.append(_finite(row.get('fps_running', '')))
                        
                        # Confidence values
                        self.filter_confidence.append(_finite(row.get('filter_confidence', '')))
                        self.detector_confidence.append(_finite(row.get('detector_confidence', '')))
                        self.ai_confidence.append(_finite(row.get('ai_confidence', '')))
                        
                        # State
                        state = row.get('state', '')
                        self.states.append(str(state) if state else 'UNKNOWN')
                        
                    except (TypeError, ValueError, KeyError):
                        # Skip malformed rows
                        continue
            
            if not self.timestamps:
                self.status_label.setText("ERROR: No valid data in telemetry CSV")
                return False
            
            self.telemetry_data = path
            self.status_label.setText(f"Loaded {len(self.timestamps)} frames from telemetry CSV")
            
            # Update all graph widgets
            for graph_widget in self.graph_widgets:
                graph_widget.set_data(
                    self.timestamps, self.frames,
                    self.center_errors, self.centroid_errors,
                    self.target_x, self.target_y,
                    self.predicted_x, self.predicted_y,
                    self.pan_values, self.tilt_values,
                    self.fps_values,
                    self.filter_confidence, self.detector_confidence, self.ai_confidence,
                    self.states
                )
            
            return True
            
        except Exception as e:
            self.status_label.setText(f"ERROR: Failed to load telemetry: {e}")
            return False
    
    def get_graph_data(self):
        """Return the current graph data arrays for external use.

        Used by the main simulation loop to inject live telemetry
        values into the graph panel every frame.
        """
        return {
            'timestamps': list(self.timestamps),
            'center_errors': list(self.center_errors),
            'centroid_errors': list(self.centroid_errors),
            'target_x': list(self.target_x),
            'target_y': list(self.target_y),
            'predicted_x': list(self.predicted_x),
            'predicted_y': list(self.predicted_y),
            'pan_values': list(self.pan_values),
            'tilt_values': list(self.tilt_values),
            'fps_values': list(self.fps_values),
            'filter_confidence': list(self.filter_confidence),
            'detector_confidence': list(self.detector_confidence),
            'ai_confidence': list(self.ai_confidence),
        }
    
    def set_live_data(self, center_error=None, centroid_error=None,
                      target_x=None, target_y=None,
                      predicted_x=None, predicted_y=None,
                      pan_deg=None, tilt_deg=None,
                      fps=None,
                      filter_confidence=None, detector_confidence=None, ai_confidence=None):
        """Push live telemetry values to every graph widget.

        Called every frame from the main simulation loop so the benchmark
        performance graphs reflect the live atmospheric and disturbance
        state in real time.
        """
        for gw in self.graph_widgets:
            gw.set_live_data(
                center_error, centroid_error,
                target_x, target_y,
                predicted_x, predicted_y,
                pan_deg, tilt_deg,
                fps,
                filter_confidence, detector_confidence, ai_confidence,
            )

    def reset(self, status_text=None):
        """Clear every series and graph widget (a fresh benchmark run).

        REPLAY and every new run go through here, so no curve from a previous
        run can survive into the next one.
        """
        self.timestamps = []
        self.frames = []
        self.center_errors = []
        self.centroid_errors = []
        self.target_x = []
        self.target_y = []
        self.predicted_x = []
        self.predicted_y = []
        self.pan_values = []
        self.tilt_values = []
        self.fps_values = []
        self.filter_confidence = []
        self.detector_confidence = []
        self.ai_confidence = []
        self.states = []
        self.telemetry_data = None
        for graph in self.graph_widgets:
            graph.reset_data()
        if status_text is not None:
            self.status_label.setText(status_text)
        self.update()

    def append_live_sample(self, timestamp_s, frame, live):
        """Append ONE real processed frame and redraw.

        ``live`` is the per-frame dictionary produced by the benchmark engine
        for the frame that was just analysed, so the plotted point always
        corresponds to a frame that was actually processed — no invented or
        interpolated data.  Non-finite values degrade to a gap, never to a
        NaN that would poison the canvas.
        """
        timestamp = _finite(timestamp_s)
        if timestamp is None or live is None:
            return
        self.timestamps.append(timestamp)
        self.frames.append(int(frame) if isinstance(frame, (int, float)) else len(self.timestamps))
        self.center_errors.append(_finite(live.get("center_error")))
        self.centroid_errors.append(_finite(live.get("centroid_error")))
        self.target_x.append(_finite(live.get("target_x")))
        self.target_y.append(_finite(live.get("target_y")))
        self.predicted_x.append(_finite(live.get("predicted_x")))
        self.predicted_y.append(_finite(live.get("predicted_y")))
        self.pan_values.append(_finite(live.get("pan")))
        self.tilt_values.append(_finite(live.get("tilt")))
        self.fps_values.append(_finite(live.get("fps")))
        self.filter_confidence.append(_finite(live.get("filter_confidence")))
        self.detector_confidence.append(_finite(live.get("detector_confidence")))
        self.ai_confidence.append(_finite(live.get("ai_confidence")))
        self.states.append(str(live.get("state") or "UNKNOWN"))

        # Trim the rolling window in one slice once it grows past the cap
        # (amortised O(1)) so a long simulation cannot make repainting heavy.
        if len(self.timestamps) > _MAX_LIVE_SAMPLES + _MAX_LIVE_SLACK:
            drop = len(self.timestamps) - _MAX_LIVE_SAMPLES
            for series in (
                self.timestamps, self.frames, self.center_errors,
                self.centroid_errors, self.target_x, self.target_y,
                self.predicted_x, self.predicted_y, self.pan_values,
                self.tilt_values, self.fps_values, self.filter_confidence,
                self.detector_confidence, self.ai_confidence, self.states,
            ):
                del series[:drop]

        # Plain append per widget: the whole series is NOT rebuilt and NOT
        # re-copied on every frame, so live playback stays cheap.
        for graph_widget in self.graph_widgets:
            graph_widget.append_live_sample(timestamp, frame, live)
        self.status_label.setText(
            f"Live — {len(self.timestamps)} frames processed"
        )
        self.update()



class _BenchmarkSingleGraph(QWidget):
    """
    Single graph widget for benchmark results.
    
    Draws a time-series line graph with proper axes, labels, and legend.
    """
    
    def __init__(self, title: str, unit: str, draw_function):
        super().__init__()
        self.title = title
        self.unit = unit
        self.draw_function = draw_function
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        self.data = {}
        # Populated ONLY when a draw function raises. Tests assert it stays
        # None; it also keeps the raw exception out of the rendered canvas.
        self.draw_error = None
        # Live marker values must EXIST before the first repaint, otherwise
        # paintEvent() would read an attribute that was never created.
        self._apply_live()

    def _apply_live(self, center_error=None, centroid_error=None,
                    target_x=None, target_y=None,
                    predicted_x=None, predicted_y=None,
                    pan_deg=None, tilt_deg=None, fps=None,
                    filter_confidence=None, detector_confidence=None,
                    ai_confidence=None):
        """Store the live marker values, dropping anything non-finite."""
        self.live_center_error = _finite(center_error)
        self.live_centroid_error = _finite(centroid_error)
        self.live_target_x = _finite(target_x)
        self.live_target_y = _finite(target_y)
        self.live_predicted_x = _finite(predicted_x)
        self.live_predicted_y = _finite(predicted_y)
        self.live_pan_deg = _finite(pan_deg)
        self.live_tilt_deg = _finite(tilt_deg)
        self.live_fps = _finite(fps)
        self.live_filter_confidence = _finite(filter_confidence)
        self.live_detector_confidence = _finite(detector_confidence)
        self.live_ai_confidence = _finite(ai_confidence)

    def append_live_sample(self, timestamp_s, frame, live):
        """Append one already-sanitized processed frame.

        The panel is the single producer and sanitizes every value on the way
        in, so this is an O(1) append plus a repaint — never a rebuild of the
        series (which would re-copy the whole run on every frame).
        """
        if not self.data.get('timestamps'):
            self.data = {
                'timestamps': [], 'frames': [], 'center_errors': [],
                'centroid_errors': [], 'target_x': [], 'target_y': [],
                'predicted_x': [], 'predicted_y': [], 'pan_values': [],
                'tilt_values': [], 'fps_values': [], 'filter_confidence': [],
                'detector_confidence': [], 'ai_confidence': [], 'states': [],
            }
        self.data['timestamps'].append(_finite(timestamp_s) or 0.0)
        self.data['frames'].append(int(frame) if isinstance(frame, int) else len(self.data['frames']) + 1)
        # EVERY value is validated here: NaN, +-Inf, strings, None and any
        # other object become a gap, so a malformed frame can neither crash
        # the chart nor draw nonsense.
        self.data['center_errors'].append(_finite(live.get('center_error')))
        self.data['centroid_errors'].append(_finite(live.get('centroid_error')))
        self.data['target_x'].append(_finite(live.get('target_x')))
        self.data['target_y'].append(_finite(live.get('target_y')))
        self.data['predicted_x'].append(_finite(live.get('predicted_x')))
        self.data['predicted_y'].append(_finite(live.get('predicted_y')))
        self.data['pan_values'].append(_finite(live.get('pan')))
        self.data['tilt_values'].append(_finite(live.get('tilt')))
        self.data['fps_values'].append(_finite(live.get('fps')))
        self.data['filter_confidence'].append(_finite(live.get('filter_confidence')))
        self.data['detector_confidence'].append(_finite(live.get('detector_confidence')))
        self.data['ai_confidence'].append(_finite(live.get('ai_confidence')))
        _state = live.get('state')
        self.data['states'].append(str(_state) if _state else 'UNKNOWN')
        # Same rolling window as the panel, sliced in one go: a long live run
        # stays cheap to repaint without ever inventing or interpolating data.
        if len(self.data['timestamps']) > _MAX_LIVE_SAMPLES + _MAX_LIVE_SLACK:
            drop = len(self.data['timestamps']) - _MAX_LIVE_SAMPLES
            for series in self.data.values():
                del series[:drop]
        # Keep the live marker on the frame that is actually on screen.
        self._apply_live(
            live.get('center_error'), live.get('centroid_error'),
            live.get('target_x'), live.get('target_y'),
            live.get('predicted_x'), live.get('predicted_y'),
            live.get('pan'), live.get('tilt'), live.get('fps'),
            live.get('filter_confidence'), live.get('detector_confidence'),
            live.get('ai_confidence'),
        )
        self.update()

    def reset_data(self):
        """Drop every stored series (used by REPLAY / a fresh benchmark run)."""
        self.data = {}
        for name in (
            "live_center_error", "live_centroid_error", "live_target_x",
            "live_target_y", "live_predicted_x", "live_predicted_y",
            "live_pan_deg", "live_tilt_deg", "live_fps",
            "live_filter_confidence", "live_detector_confidence",
            "live_ai_confidence",
        ):
            setattr(self, name, None)
        self.draw_error = None
        self.update()
    
    def set_data(self, timestamps, frames,
                 center_errors, centroid_errors,
                 target_x, target_y,
                 predicted_x, predicted_y,
                 pan_values, tilt_values,
                 fps_values,
                 filter_confidence, detector_confidence, ai_confidence,
                 states):
        """Store data for graphing."""
        self.data = {
            'timestamps': timestamps,
            'frames': frames,
            'center_errors': _finite_series(center_errors),
            'centroid_errors': _finite_series(centroid_errors),
            'target_x': _finite_series(target_x),
            'target_y': _finite_series(target_y),
            'predicted_x': _finite_series(predicted_x),
            'predicted_y': _finite_series(predicted_y),
            'pan_values': _finite_series(pan_values),
            'tilt_values': _finite_series(tilt_values),
            'fps_values': _finite_series(fps_values),
            'filter_confidence': _finite_series(filter_confidence),
            'detector_confidence': _finite_series(detector_confidence),
            'ai_confidence': _finite_series(ai_confidence),
            'states': list(states),
        }
        self.live_center_error = None
        self.live_centroid_error = None
        self.live_target_x = None
        self.live_target_y = None
        self.live_predicted_x = None
        self.live_predicted_y = None
        self.live_pan_deg = None
        self.live_tilt_deg = None
        self.live_fps = None
        self.live_filter_confidence = None
        self.live_detector_confidence = None
        self.live_ai_confidence = None
        self.update()
    
    def set_live_data(self, center_error=None, centroid_error=None,
                      target_x=None, target_y=None,
                      predicted_x=None, predicted_y=None,
                      pan_deg=None, tilt_deg=None,
                      fps=None,
                      filter_confidence=None, detector_confidence=None, ai_confidence=None):
        """Update the live values shown in the graphs.

        Called every frame from the main simulation loop so the graphs
        reflect the current atmospheric and disturbance settings in real
        time without reloading the telemetry CSV.
        """
        self._apply_live(
            center_error, centroid_error, target_x, target_y,
            predicted_x, predicted_y, pan_deg, tilt_deg, fps,
            filter_confidence, detector_confidence, ai_confidence,
        )
        self.update()
    
    # ---- responsive, measured title -------------------------------------
    def _title_font(self):
        """Pick the title point size that suits the CURRENT widget width.

        Titles shrink one step on narrow cards so they stay readable and
        still fit; they never wrap into the plot area or overflow the card.
        """
        width = max(1, self.width())
        if width >= 420:
            size = 9
        elif width >= 300:
            size = 8
        else:
            size = 7
        font = QFont("Segoe UI", size)
        font.setBold(True)
        return font

    @staticmethod
    def _wrap_title(text, metrics, first_width, other_width, max_lines):
        """Wrap ``text`` to at most ``max_lines`` lines that each fit.

        Words are grouped greedily; if even the allowed line count is not
        enough, the tail is merged into the last line and elided so the text
        ends cleanly instead of being cut mid-word or overflowing.
        """
        words = str(text).split()
        if not words:
            return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            limit = first_width if not lines else other_width
            candidate = f"{current} {word}"
            if metrics.horizontalAdvance(candidate) <= limit:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)

        if len(lines) > max_lines:
            tail = " ".join(lines[max_lines - 1:])
            lines = lines[:max_lines - 1] + [tail]

        # Guarantee fit even for a single unbreakable word.
        wrapped = []
        for index, line in enumerate(lines):
            limit = first_width if index == 0 else other_width
            wrapped.append(metrics.elidedText(line, Qt.ElideRight, max(20, int(limit))))
        return wrapped

    def _title_layout(self):
        """Return ``(font, lines, block_height)`` for the current width."""
        font = self._title_font()
        metrics = QFontMetrics(font)
        unit_width = 0
        if self.unit:
            unit_width = metrics.horizontalAdvance(str(self.unit)) + 10

        available = max(40, self.width() - 2 * _GRAPH_PAD_X - unit_width)
        max_lines = _GRAPH_TITLE_MAX_LINES if self.width() < 340 else 2
        lines = self._wrap_title(self.title, metrics, available, available, max_lines)
        line_height = metrics.height()
        block = len(lines) * line_height + max(0, len(lines) - 1) * _GRAPH_TITLE_GAP
        return font, lines, block

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(13, 21, 31))
        
        # Title block is measured first: the plot area is then laid out BELOW
        # it, so a wrapped title can never overlap the graph or leave the card.
        title_font, title_lines, title_block_h = self._title_layout()
        metrics = QFontMetrics(title_font)
        line_h = metrics.height()
        
        margin_left = 50
        margin_right = 15
        margin_top = int(_GRAPH_TITLE_TOP + title_block_h + _GRAPH_TITLE_GAP + 4)
        margin_bottom = 35
        
        # Calculate plot area
        plot_x = margin_left
        plot_y = margin_top
        plot_w = self.width() - margin_left - margin_right
        plot_h = self.height() - margin_top - margin_bottom
        
        if plot_w < 30 or plot_h < 30:
            painter.end()
            return
        
        # Draw background for plot area
        painter.fillRect(plot_x, plot_y, plot_w, plot_h, QColor(15, 23, 35))
        
        # Draw every title line inside a bounded rect (clipped, never spilled).
        painter.setFont(title_font)
        painter.setPen(QColor(235, 240, 245))
        first_line_w = self.width() - 2 * _GRAPH_PAD_X
        if self.unit:
            first_line_w -= metrics.horizontalAdvance(str(self.unit)) + 10
        y = _GRAPH_TITLE_TOP
        for index, line in enumerate(title_lines):
            width = first_line_w if index == 0 else self.width() - 2 * _GRAPH_PAD_X
            painter.drawText(
                int(_GRAPH_PAD_X), int(y), max(20, int(width)), int(line_h),
                int(Qt.AlignLeft | Qt.AlignVCenter), line,
            )
            y += line_h + _GRAPH_TITLE_GAP
        
        # Draw unit, right-aligned on the title's first line.
        if self.unit:
            unit_metrics = QFontMetrics(QFont("Segoe UI", 8))
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor(130, 146, 160))
            unit_w = unit_metrics.horizontalAdvance(str(self.unit)) + 4
            painter.drawText(
                int(self.width() - _GRAPH_PAD_X - unit_w), int(_GRAPH_TITLE_TOP),
                int(unit_w), int(line_h),
                int(Qt.AlignRight | Qt.AlignVCenter), str(self.unit),
            )
        
        # Draw grid lines
        painter.setPen(QPen(QColor(40, 55, 70), 1))
        for i in range(1, 4):
            y = plot_y + plot_h * i / 4.0
            painter.drawLine(int(plot_x), int(y), int(plot_x + plot_w), int(y))
        for i in range(1, 6):
            x = plot_x + plot_w * i / 6.0
            painter.drawLine(int(x), int(plot_y), int(x), int(plot_y + plot_h))
        
        # Check if we have data
        if not self.data or not self.data.get('timestamps'):
            painter.setFont(QFont("Segoe UI", 10))
            painter.setPen(QColor(128, 146, 160))
            painter.drawText(
                int(plot_x), int(plot_y), int(plot_w), int(plot_h),
                int(Qt.AlignCenter), "No data available",
            )
            painter.end()
            return
        
        # Merge live values into the data dict before drawing so the
        # _draw_* methods can read data.get('live_*') reliably.
        draw_data = dict(self.data)
        draw_data['live_center_error'] = self.live_center_error
        draw_data['live_centroid_error'] = self.live_centroid_error
        draw_data['live_target_x'] = self.live_target_x
        draw_data['live_target_y'] = self.live_target_y
        draw_data['live_predicted_x'] = self.live_predicted_x
        draw_data['live_predicted_y'] = self.live_predicted_y
        draw_data['live_pan_deg'] = self.live_pan_deg
        draw_data['live_tilt_deg'] = self.live_tilt_deg
        draw_data['live_fps'] = self.live_fps
        draw_data['live_filter_confidence'] = self.live_filter_confidence
        draw_data['live_detector_confidence'] = self.live_detector_confidence
        draw_data['live_ai_confidence'] = self.live_ai_confidence
        
        self.draw_error = None
        try:
            self.draw_function(painter, draw_data, plot_x, plot_y, plot_w, plot_h)
        except Exception as exc:  # never let a graph crash the application
            # The reason is recorded for tests/diagnostics and NOT painted:
            # raw exception text (the old "Error drawing graph: name 'QBrush'…")
            # must never appear on the canvas.
            self.draw_error = f"{type(exc).__name__}: {exc}"
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(255, 107, 107))
            painter.drawText(
                int(plot_x), int(plot_y), int(plot_w), int(plot_h),
                int(Qt.AlignCenter), "Graph unavailable",
            )
        
        painter.end()
    
    @staticmethod
    def _draw_error_graph(painter, data, plot_x, plot_y, plot_w, plot_h):
        """Draw tracking/centroid error graph."""
        # Filter out None values for valid data points
        timestamps = data['timestamps']
        center_errors = data['center_errors']
        centroid_errors = data['centroid_errors']
        
        # Get valid data points
        valid_center = [(t, e) for t, e in zip(timestamps, center_errors) if e is not None]
        valid_centroid = [(t, e) for t, e in zip(timestamps, centroid_errors) if e is not None]
        
        if not valid_center and not valid_centroid:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(128, 146, 160))
            painter.drawText(int(plot_x + 10), int(plot_y + plot_h - 15), "No error data available")
            return
        
        # Calculate Y range
        all_values = [e for _, e in valid_center] + [e for _, e in valid_centroid]
        if all_values:
            vmin = min(all_values)
            vmax = max(all_values)
            if abs(vmax - vmin) < 1:
                vmin = max(0, vmin - 1)
                vmax = vmax + 1
            else:
                pad = (vmax - vmin) * 0.1
                vmin = max(0, vmin - pad)
                vmax = vmax + pad
        else:
            vmin, vmax = 0, 10
        
        # Draw axes labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(2, plot_y + 10, f"{vmax:.1f}")
        painter.drawText(2, plot_y + plot_h - 2, f"{vmin:.1f}")
        
        # Draw threshold line at 10px (PS requirement)
        if vmax > 10:
            threshold_y = plot_y + plot_h - ((10 - vmin) / (vmax - vmin)) * plot_h
            painter.setPen(QPen(QColor(255, 138, 146), 1, Qt.DashLine))
            painter.drawLine(int(plot_x), int(threshold_y), 
                           int(plot_x + plot_w), int(threshold_y))
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QColor(255, 138, 146))
            painter.drawText(int(plot_x + plot_w - 35), int(threshold_y - 3), "10 px limit")
        
        # Draw centroid error (if available)
        if valid_centroid:
            painter.setPen(QPen(QColor(255, 216, 61), 2))  # Yellow
            for i in range(len(valid_centroid) - 1):
                x1 = plot_x + (valid_centroid[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y1 = plot_y + plot_h - ((valid_centroid[i][1] - vmin) / (vmax - vmin)) * plot_h
                x2 = plot_x + (valid_centroid[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y2 = plot_y + plot_h - ((valid_centroid[i+1][1] - vmin) / (vmax - vmin)) * plot_h
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw center error (tracking error)
        if valid_center:
            painter.setPen(QPen(QColor(80, 255, 120), 2))  # Green
            for i in range(len(valid_center) - 1):
                x1 = plot_x + (valid_center[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y1 = plot_y + plot_h - ((valid_center[i][1] - vmin) / (vmax - vmin)) * plot_h
                x2 = plot_x + (valid_center[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y2 = plot_y + plot_h - ((valid_center[i+1][1] - vmin) / (vmax - vmin)) * plot_h
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Live data point: if we have a current live error, show it as a
        # bright highlighted dot so the graph always reflects the latest
        # atmospheric/disturbance state.
        lw = max(1, min(plot_w, plot_w))
        if data.get('live_center_error') is not None:
            lc = data['live_center_error']
            if vmin != vmax:
                ly_pos = plot_y + plot_h - ((lc - vmin) / (vmax - vmin)) * plot_h
            else:
                ly_pos = plot_y + plot_h / 2
            lx_pos = plot_x + plot_w
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(int(lx_pos - 3), int(ly_pos - 3), 6, 6)
        
        # Draw legend
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        legend_y = int(plot_y + plot_h + 5)
        painter.setPen(QColor(80, 255, 120))
        painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
        painter.setPen(QColor(235, 240, 245))
        painter.drawText(int(plot_x + 16), legend_y + 3, "Tracking Error")
        
        if valid_centroid:
            legend_y += 12
            painter.setPen(QColor(255, 216, 61))
            painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
            painter.setPen(QColor(235, 240, 245))
            painter.drawText(int(plot_x + 16), legend_y + 3, "Centroid Error")
    
    @staticmethod
    def _draw_position_graph(painter, data, plot_x, plot_y, plot_w, plot_h):
        """Draw target position vs Kalman prediction graph."""
        timestamps = data['timestamps']
        target_x = data['target_x']
        target_y = data['target_y']
        predicted_x = data['predicted_x']
        predicted_y = data['predicted_y']
        
        # Get valid data
        valid_target = [(t, x, y) for t, x, y in zip(timestamps, target_x, target_y) 
                       if x is not None and y is not None]
        valid_pred = [(t, x, y) for t, x, y in zip(timestamps, predicted_x, predicted_y) 
                     if x is not None and y is not None]
        
        if not valid_target and not valid_pred:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(128, 146, 160))
            painter.drawText(int(plot_x + 10), int(plot_y + plot_h - 15), "No position data")
            return
        
        # Calculate ranges for X and Y
        all_x = [x for _, x, _ in valid_target] + [x for _, x, _ in valid_pred]
        all_y = [y for _, _, y in valid_target] + [y for _, _, y in valid_pred]
        
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            x_pad = (x_max - x_min) * 0.1 if x_max > x_min else 10
            y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 10
            
            x_min = max(0, x_min - x_pad)
            x_max = x_max + x_pad
            y_min = max(0, y_min - y_pad)
            y_max = y_max + y_pad
        else:
            x_min, x_max = 0, 640
            y_min, y_max = 0, 480
        
        # Draw axes labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(2, plot_y + 10, f"X: {x_max:.0f}")
        painter.drawText(2, plot_y + plot_h - 2, f"X: {x_min:.0f}")
        painter.drawText(plot_x + plot_w - 30, plot_y + plot_h - 2, f"Y: {y_min:.0f}")
        painter.drawText(plot_x + plot_w - 30, plot_y + 10, f"Y: {y_max:.0f}")
        
        # Draw target position (X coordinate over time)
        if valid_target:
            painter.setPen(QPen(QColor(80, 255, 120), 2))  # Green for target
            for i in range(len(valid_target) - 1):
                x1 = plot_x + (valid_target[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y1 = plot_y + plot_h - ((valid_target[i][1] - x_min) / (x_max - x_min)) * plot_h
                x2 = plot_x + (valid_target[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y2 = plot_y + plot_h - ((valid_target[i+1][1] - x_min) / (x_max - x_min)) * plot_h
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw prediction (X coordinate over time)
        if valid_pred:
            painter.setPen(QPen(QColor(0, 220, 255), 2))  # Cyan for prediction
            for i in range(len(valid_pred) - 1):
                x1 = plot_x + (valid_pred[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y1 = plot_y + plot_h - ((valid_pred[i][1] - x_min) / (x_max - x_min)) * plot_h
                x2 = plot_x + (valid_pred[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y2 = plot_y + plot_h - ((valid_pred[i+1][1] - x_min) / (x_max - x_min)) * plot_h
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Live data: show the current live target position as a bright
        # highlighted marker so the graph always reflects the latest
        # atmospheric/disturbance state.
        if data.get('live_target_x') is not None and data.get('live_target_y') is not None:
            lx = data['live_target_x']
            ly = data['live_target_y']
            if x_max > x_min and y_max > y_min:
                lx_pos = plot_x + ((lx - x_min) / (x_max - x_min)) * plot_w
                ly_pos = plot_y + plot_h - ((ly - y_min) / (y_max - y_min)) * plot_h
            else:
                lx_pos = plot_x + plot_w / 2
                ly_pos = plot_y + plot_h / 2
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(int(lx_pos - 5), int(ly_pos - 5), 10, 10)
            painter.drawEllipse(int(lx_pos - 9), int(ly_pos - 9), 18, 18)
        
        # Draw legend
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        legend_y = int(plot_y + plot_h + 5)
        painter.setPen(QColor(80, 255, 120))
        painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
        painter.setPen(QColor(235, 240, 245))
        painter.drawText(int(plot_x + 16), legend_y + 3, "Target X")
        
        legend_y += 12
        painter.setPen(QColor(0, 220, 255))
        painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
        painter.setPen(QColor(235, 240, 245))
        painter.drawText(int(plot_x + 16), legend_y + 3, "Predicted X")
    
    @staticmethod
    def _draw_pan_tilt_graph(painter, data, plot_x, plot_y, plot_w, plot_h):
        """Draw pan/tilt control response graph."""
        timestamps = data['timestamps']
        pan_values = data['pan_values']
        tilt_values = data['tilt_values']
        
        valid_pan = [(t, v) for t, v in zip(timestamps, pan_values) if v is not None]
        valid_tilt = [(t, v) for t, v in zip(timestamps, tilt_values) if v is not None]
        
        if not valid_pan and not valid_tilt:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(128, 146, 160))
            painter.drawText(int(plot_x + 10), int(plot_y + plot_h - 15), "No pan/tilt data")
            return
        
        # Calculate range
        all_values = [v for _, v in valid_pan] + [v for _, v in valid_tilt]
        if all_values:
            vmin = min(all_values)
            vmax = max(all_values)
            if abs(vmax - vmin) < 0.1:
                vmin = vmin - 0.5
                vmax = vmax + 0.5
            else:
                pad = (vmax - vmin) * 0.15
                vmin = vmin - pad
                vmax = vmax + pad
        else:
            vmin, vmax = -5, 5
        
        # Draw axes labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(2, plot_y + 10, f"{vmax:.2f}°")
        painter.drawText(2, plot_y + plot_h - 2, f"{vmin:.2f}°")
        
        # Draw pan
        if valid_pan:
            painter.setPen(QPen(QColor(65, 155, 255), 2))  # Blue for pan
            for i in range(len(valid_pan) - 1):
                x1 = plot_x + (valid_pan[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y1 = plot_y + plot_h - ((valid_pan[i][1] - vmin) / (vmax - vmin)) * plot_h
                x2 = plot_x + (valid_pan[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y2 = plot_y + plot_h - ((valid_pan[i+1][1] - vmin) / (vmax - vmin)) * plot_h
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw tilt
        if valid_tilt:
            painter.setPen(QPen(QColor(255, 159, 67), 2))  # Orange for tilt
            for i in range(len(valid_tilt) - 1):
                x1 = plot_x + (valid_tilt[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y1 = plot_y + plot_h - ((valid_tilt[i][1] - vmin) / (vmax - vmin)) * plot_h
                x2 = plot_x + (valid_tilt[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y2 = plot_y + plot_h - ((valid_tilt[i+1][1] - vmin) / (vmax - vmin)) * plot_h
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Live data: show the current live pan/tilt values as bright
        # highlighted markers so the graph always reflects the latest
        # atmospheric and disturbance state.
        lw = max(1, min(plot_w, plot_w))
        if data.get('live_pan_deg') is not None:
            lp = data['live_pan_deg']
            if vmin != vmax:
                ly_pos = plot_y + plot_h - ((lp - vmin) / (vmax - vmin)) * plot_h
            else:
                ly_pos = plot_y + plot_h / 2
            lx_pos = plot_x + plot_w
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(int(lx_pos - 3), int(ly_pos - 3), 6, 6)
        
        if data.get('live_tilt_deg') is not None:
            lt = data['live_tilt_deg']
            if vmin != vmax:
                ly_pos = plot_y + plot_h - ((lt - vmin) / (vmax - vmin)) * plot_h
            else:
                ly_pos = plot_y + plot_h / 2
            lh_x = plot_x + plot_w
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(int(lh_x - 3), int(ly_pos - 3), 6, 6)
        
        # Draw legend
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        legend_y = int(plot_y + plot_h + 5)
        painter.setPen(QColor(65, 155, 255))
        painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
        painter.setPen(QColor(235, 240, 245))
        painter.drawText(int(plot_x + 16), legend_y + 3, "Pan")
        
        legend_y += 12
        painter.setPen(QColor(255, 159, 67))
        painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
        painter.setPen(QColor(235, 240, 245))
        painter.drawText(int(plot_x + 16), legend_y + 3, "Tilt")
    
    @staticmethod
    def _draw_fps_graph(painter, data, plot_x, plot_y, plot_w, plot_h):
        """Draw FPS/processing performance graph."""
        timestamps = data['timestamps']
        fps_values = data['fps_values']
        
        valid_fps = [(t, f) for t, f in zip(timestamps, fps_values) if f is not None]
        
        if not valid_fps:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(128, 146, 160))
            painter.drawText(int(plot_x + 10), int(plot_y + plot_h - 15), "No FPS data")
            return
        
        # Calculate range
        all_fps = [f for _, f in valid_fps]
        vmin = min(all_fps)
        vmax = max(all_fps)
        if abs(vmax - vmin) < 1:
            vmin = max(0, vmin - 5)
            vmax = vmax + 5
        else:
            pad = (vmax - vmin) * 0.1
            vmin = max(0, vmin - pad)
            vmax = vmax + pad
        
        # Draw threshold at 20 FPS (PS requirement)
        if vmax > 20:
            threshold_y = plot_y + plot_h - ((20 - vmin) / (vmax - vmin)) * plot_h
            painter.setPen(QPen(QColor(80, 255, 120), 1, Qt.DashLine))
            painter.drawLine(int(plot_x), int(threshold_y), 
                           int(plot_x + plot_w), int(threshold_y))
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QColor(80, 255, 120))
            painter.drawText(int(plot_x + plot_w - 35), int(threshold_y - 3), "20 FPS min")
        
        # Draw FPS line
        painter.setPen(QPen(QColor(255, 216, 61), 2))  # Yellow
        for i in range(len(valid_fps) - 1):
            x1 = plot_x + (valid_fps[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
            y1 = plot_y + plot_h - ((valid_fps[i][1] - vmin) / (vmax - vmin)) * plot_h
            x2 = plot_x + (valid_fps[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
            y2 = plot_y + plot_h - ((valid_fps[i+1][1] - vmin) / (vmax - vmin)) * plot_h
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw axes labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(2, plot_y + 10, f"{vmax:.0f}")
        painter.drawText(2, plot_y + plot_h - 2, f"{vmin:.0f}")
        
        # Live data: show the current live FPS value as a bright highlighted
        # marker so the graph always reflects the latest atmospheric and
        # disturbance state.
        lw = max(1, min(plot_w, plot_w))
        if data.get('live_fps') is not None:
            lf = data['live_fps']
            if vmin != vmax:
                ly_pos = plot_y + plot_h - ((lf - vmin) / (vmax - vmin)) * plot_h
            else:
                ly_pos = plot_y + plot_h / 2
            lx_pos = plot_x + plot_w
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(int(lx_pos - 3), int(ly_pos - 3), 6, 6)
        
        # Draw legend
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        legend_y = int(plot_y + plot_h + 5)
        painter.setPen(QColor(255, 216, 61))
        painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
        painter.setPen(QColor(235, 240, 245))
        painter.drawText(int(plot_x + 16), legend_y + 3, "Processing FPS")
    
    @staticmethod
    def _draw_confidence_graph(painter, data, plot_x, plot_y, plot_w, plot_h):
        """Draw tracking confidence graph."""
        timestamps = data['timestamps']
        filter_conf = data['filter_confidence']
        detector_conf = data['detector_confidence']
        ai_conf = data['ai_confidence']
        
        valid_filter = [(t, c) for t, c in zip(timestamps, filter_conf) if c is not None]
        valid_detector = [(t, c) for t, c in zip(timestamps, detector_conf) if c is not None]
        valid_ai = [(t, c) for t, c in zip(timestamps, ai_conf) if c is not None]
        
        if not valid_filter and not valid_detector and not valid_ai:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(128, 146, 160))
            painter.drawText(int(plot_x + 10), int(plot_y + plot_h - 15), "No confidence data")
            return
        
        # Fixed range 0-1 for confidence
        vmin, vmax = 0, 1
        
        # Draw axes labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(2, plot_y + 10, "1.00")
        painter.drawText(2, plot_y + plot_h - 2, "0.00")
        
        # Draw filter confidence
        if valid_filter:
            painter.setPen(QPen(QColor(175, 105, 255), 2))  # Purple
            for i in range(len(valid_filter) - 1):
                x1 = plot_x + (valid_filter[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y1 = plot_y + plot_h - (valid_filter[i][1] / (vmax - vmin)) * plot_h
                x2 = plot_x + (valid_filter[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y2 = plot_y + plot_h - (valid_filter[i+1][1] / (vmax - vmin)) * plot_h
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw detector confidence
        if valid_detector:
            painter.setPen(QPen(QColor(80, 255, 120), 2))  # Green
            for i in range(len(valid_detector) - 1):
                x1 = plot_x + (valid_detector[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y1 = plot_y + plot_h - (valid_detector[i][1] / (vmax - vmin)) * plot_h
                x2 = plot_x + (valid_detector[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y2 = plot_y + plot_h - (valid_detector[i+1][1] / (vmax - vmin)) * plot_h
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw AI confidence
        if valid_ai:
            painter.setPen(QPen(QColor(255, 220, 55), 2))  # Yellow
            for i in range(len(valid_ai) - 1):
                x1 = plot_x + (valid_ai[i][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y1 = plot_y + plot_h - (valid_ai[i][1] / (vmax - vmin)) * plot_h
                x2 = plot_x + (valid_ai[i+1][0] - timestamps[0]) / (timestamps[-1] - timestamps[0]) * plot_w
                y2 = plot_y + plot_h - (valid_ai[i+1][1] / (vmax - vmin)) * plot_h
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw a horizontal line at 0.2 (link ready threshold)
        threshold_y = plot_y + plot_h - (0.2 / (vmax - vmin)) * plot_h
        painter.setPen(QPen(QColor(255, 220, 55), 1, Qt.DashLine))
        painter.drawLine(int(plot_x), int(threshold_y), int(plot_x + plot_w), int(threshold_y))
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QColor(255, 220, 55))
        painter.drawText(int(plot_x + plot_w - 35), int(threshold_y - 3), "0.2 AI threshold")
        
        # Live data: show the current live confidence values as bright
        # highlighted markers. The graph always reflects the latest
        # atmospheric and disturbance state.
        lw = max(1, min(plot_w, plot_w))
        if data.get('live_filter_confidence') is not None:
            lf = data['live_filter_confidence']
            ly_pos = plot_y + plot_h - (lf / 1.0) * plot_h
            lx_pos = plot_x + plot_w
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(int(lx_pos - 3), int(ly_pos - 3), 6, 6)
        
        if data.get('live_detector_confidence') is not None:
            ld = data['live_detector_confidence']
            ly_pos = plot_y + plot_h - (ld / 1.0) * plot_h
            lx_pos = plot_x + plot_w
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(int(lx_pos - 3), int(ly_pos - 3), 6, 6)
        
        if data.get('live_ai_confidence') is not None:
            la = data['live_ai_confidence']
            ly_pos = plot_y + plot_h - (la / 1.0) * plot_h
            lx_pos = plot_x + plot_w
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(int(lx_pos - 3), int(ly_pos - 3), 6, 6)
        
        # Draw legend
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        legend_y = int(plot_y + plot_h + 5)
        
        painter.setPen(QColor(175, 105, 255))
        painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
        painter.setPen(QColor(235, 240, 245))
        painter.drawText(int(plot_x + 16), legend_y + 3, "Filter")
        
        legend_y += 12
        painter.setPen(QColor(80, 255, 120))
        painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
        painter.setPen(QColor(235, 240, 245))
        painter.drawText(int(plot_x + 16), legend_y + 3, "Detector")
        
        legend_y += 12
        painter.setPen(QColor(255, 220, 55))
        painter.drawLine(int(plot_x), legend_y, int(plot_x + 12), legend_y)
        painter.setPen(QColor(235, 240, 245))
        painter.drawText(int(plot_x + 16), legend_y + 3, "AI")
