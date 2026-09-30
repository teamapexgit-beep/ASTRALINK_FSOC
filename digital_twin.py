from __future__ import annotations

import math
import random
import zlib
from collections import deque
from typing import Iterable

from PySide6.QtCore import QPoint, QPointF, Qt, QRectF, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPolygonF, QRadialGradient
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import config
from flow_layout import FlowLayout


class DigitalTwinWidget(QWidget):
    """Independent-view 3D-style space visualization of the live FSOC state.

    The viewer is NOT aligned to the optical camera POV.  User yaw/pitch/zoom
    controls are independent of PAT pan/tilt.  Satellite positions come from
    the same simulator frame used by the radar and virtual-camera feed.
    """

    focus_requested = Signal(str)
    # Direct 3D manipulation: emits (target_id, dx_px, dy_px) image-coordinate
    # deltas while the user drags a satellite. The host forwards these into
    # the REAL simulator, so radar/camera/twin stay one synchronized mission.
    dragDelta = Signal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Modest minimum: the twin is a hero visual that must shrink on a
        # tablet / small laptop instead of forcing the window wider.
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)

        self.camera_angles = (0.0, 0.0)
        self.target_angles = (0.0, 0.0)
        self.predicted_angles = None
        self.range_m = 1000.0
        self.targets: list[dict] = []
        self.world_targets: list[dict] = []
        self.camera_position = (0.0, 0.0, 0.0)
        self.designated_id = "TGT-01"
        self.state = "SEARCHING"
        self.frame_count = 0
        self.sim_time = 0.0
        self.fps = 0.0
        self.motion_pattern = "Straight Line"
        self.turbulence_enabled = False
        self.turbulence_strength = 0.0
        self.jitter_enabled = False
        self.jitter_strength = 0.0
        self.platform_enabled = False
        self.platform_strength = 0.0
        self.noise_name = "None"
        self.atmosphere_name = "Clear"
        self.atmosphere_level = 0
        self.platform_offset_px = (0.0, 0.0)
        self.jitter_px = (0.0, 0.0)
        self.beam_wander_px = (0.0, 0.0)
        # Stage 7 label management: boxes claimed by labels drawn this frame.
        self._label_boxes = []
        self.link_ready = False

        # Independent user view: these values are never overwritten by PAT pan/tilt.
        self.view_yaw = -38.0
        self.view_pitch = 16.0
        self.zoom = 1.05
        self._dragging = False
        self._last_mouse = QPoint()
        # Direct-manipulation grab (id + last screen point) or None when the
        # user is orbiting the viewpoint instead of dragging an object.
        self._object_drag = None

        # Procedural environment (VISUAL ONLY). Star field, celestial bodies,
        # nebulae, clusters and debris are generated per environment preset with
        # a seed that never touches the simulator, tracker, controller or any
        # benchmark measurement — tracking stays fully deterministic.
        self.environment_name = "EARTH ORBIT"
        self._environment_seed = 26169
        self._build_environment(self.environment_name)

        self._target_trails: dict[str, deque[tuple[float, float, float]]] = {}
        self._camera_trail: deque[tuple[float, float, float]] = deque(maxlen=100)

        # Visualization-only scene edits. These NEVER feed back into the
        # simulator, controller, tracker, or any benchmark calculation — they
        # only shift where objects are drawn inside this independent view.
        self.scene_camera_offset = [0.0, 0.0, 0.0]
        self.scene_target_offset = [0.0, 0.0, 0.0]
        self.scene_separation = 1.0
        self._last_beam_origin_screen = None
        self.terminal_anchor_screen = None

        # Focus-mode state. "FULL SCENE" keeps the user's independent orbit;
        # terminal focus modes smoothly drive the SAME viewpoint variables to
        # a computed close-up, then hand control back to the user.
        self._view_mode = "FULL SCENE"
        self._focus_anim = None
        self._focus_from = None
        self._focus_to = None
        self._focus_t = 0.0

        # BEAM IMPACT view: a visualization mode that flies to the receiving
        # aperture and renders where the live beam lands relative to it. The
        # impact position comes ONLY from the real pointing state pushed by
        # the mission (never invented here).
        self.beam_impact_enabled = False
        self.beam_impact_center_error_px = None   # live center error (px)
        self.beam_impact_offset_px = (0.0, 0.0)   # live tracker-vs-center offset (px)
        self.beam_impact_link_ready = False
        self.beam_impact_state = "BEAM MISALIGNED"

    @property
    def view_mode(self):
        """Current cinematic mode: FULL SCENE, CAMERA TERMINAL or BEACON TERMINAL."""
        return self._view_mode

    # ---------------------- procedural environment (visual only) -----------------
    ENVIRONMENT_PRESETS = ("DEEP SPACE", "EARTH ORBIT", "LUNAR VICINITY", "SUNLIT ORBIT")

    def _build_environment(self, name):
        """Generate the full visual environment for one preset.

        Everything produced here is DRAW-ONLY: star field, celestial bodies,
        nebulae, clusters, debris and lighting direction. None of it feeds the
        simulator, controller, tracker or any benchmark measurement.
        """
        name = str(name).upper()
        if name not in self.ENVIRONMENT_PRESETS:
            name = "EARTH ORBIT"
        self.environment_name = name
        # Stable per-preset mixing (zlib.crc32, NOT hash(): Python string hash
        # is process-randomized and would break visual reproducibility).
        rng = random.Random(self._environment_seed * 7 + zlib.crc32(name.encode()))

        # ---- star field --------------------------------------------------
        density = {"DEEP SPACE": 1750, "EARTH ORBIT": 1150, "LUNAR VICINITY": 1000, "SUNLIT ORBIT": 850}[name]
        stars = []
        for _ in range(density):
            z = rng.uniform(20.0, 105.0)
            x = rng.uniform(-55.0, 55.0)
            y = rng.uniform(-36.0, 36.0)
            b = rng.uniform(0.22, 1.0)
            tint = rng.choice(("white", "white", "white", "cool", "warm"))
            tw_speed = rng.uniform(0.045, 0.16)
            tw_phase = rng.uniform(0.0, math.tau)
            stars.append((x, y, z, b, tint, tw_speed, tw_phase))
        self._stars = stars

        # ---- deep-space structures (nebulae + clusters) -------------------
        nebula_count = 4 if name != "SUNLIT ORBIT" else 2
        self._nebulae = [
            (
                rng.uniform(-46.0, 46.0),
                rng.uniform(-30.0, 30.0),
                rng.uniform(78.0, 104.0),
                rng.uniform(9.0, 20.0),
                rng.choice(((48, 82, 140), (94, 54, 128), (36, 108, 104), (128, 74, 60), (60, 70, 150))),
                rng.uniform(0.05, 0.11),
            )
            for _ in range(nebula_count)
        ]
        self._clusters = [
            (
                rng.uniform(-44.0, 44.0),
                rng.uniform(-28.0, 28.0),
                rng.uniform(80.0, 102.0),
                rng.uniform(3.4, 6.4),
            )
            for _ in range(3)
        ]

        # ---- celestial bodies per preset -----------------------------------
        bodies = []
        if name == "DEEP SPACE":
            bodies = [
                {"name": "DWARF PLANET", "pos": (34.0, 21.0, 121.0), "r": 1.2, "kind": "rock"},
                {"name": "SUN", "pos": (88.0, -42.0, 182.0), "r": 2.4, "kind": "sun"},
            ]
        elif name == "EARTH ORBIT":
            bodies = [
                {"name": "EARTH", "pos": (-41.0, -20.0, 112.0), "r": 7.0, "kind": "earth"},
                {"name": "MOON", "pos": (27.0, 18.0, 118.0), "r": 1.55, "kind": "moon"},
                {"name": "SUN", "pos": (82.0, -38.0, 176.0), "r": 3.6, "kind": "sun"},
            ]
        elif name == "LUNAR VICINITY":
            bodies = [
                {"name": "MOON", "pos": (16.0, 12.0, 96.0), "r": 4.2, "kind": "moon"},
                {"name": "EARTH", "pos": (-52.0, -28.0, 132.0), "r": 3.4, "kind": "earth"},
                {"name": "SUN", "pos": (80.0, -40.0, 178.0), "r": 3.2, "kind": "sun"},
            ]
        else:  # SUNLIT ORBIT
            bodies = [
                {"name": "SUN", "pos": (56.0, -30.0, 150.0), "r": 5.2, "kind": "sun"},
                {"name": "MOON", "pos": (-30.0, 20.0, 118.0), "r": 1.5, "kind": "moon"},
            ]
        # Small per-run placement jitter + drift phase, kept inside a sane
        # on-screen band. Drift is a slow draw-only creep (minutes scale).
        for body in bodies:
            body["pos"] = (
                body["pos"][0] + rng.uniform(-4.0, 4.0),
                body["pos"][1] + rng.uniform(-3.0, 3.0),
                body["pos"][2],
            )
            body["drift_phase"] = rng.uniform(0.0, math.tau)
        self._celestial = bodies

        # Deterministic procedural Earth surface detail. Visual context model,
        # not a geographic map; no weather is rendered.
        earth_rng = random.Random(1969)
        self._earth_land = []
        for _ in range(9):
            cx = earth_rng.uniform(-0.48, 0.46)
            cy = earth_rng.uniform(-0.42, 0.42)
            rx = earth_rng.uniform(0.08, 0.22)
            ry = earth_rng.uniform(0.045, 0.13)
            self._earth_land.append((cx, cy, rx, ry, earth_rng.uniform(-0.25, 0.25)))

        # ---- lighting / background per preset -------------------------------
        sun_dir = (1.0, 0.0, 0.0)
        for body in self._celestial:
            if body["kind"] == "sun":
                norm = math.hypot(*body["pos"]) or 1.0
                sun_dir = (body["pos"][0] / norm, body["pos"][1] / norm, body["pos"][2] / norm)
                break
        self._sun_dir = sun_dir
        self._lighting = {
            "DEEP SPACE": {"bg": (0, 1, 4), "grad0": (6, 12, 24), "grad1": (2, 7, 15), "halo": 0.72},
            "EARTH ORBIT": {"bg": (1, 2, 5), "grad0": (8, 17, 31), "grad1": (2, 8, 17), "halo": 1.0},
            "LUNAR VICINITY": {"bg": (0, 1, 3), "grad0": (7, 14, 22), "grad1": (2, 7, 12), "halo": 0.9},
            "SUNLIT ORBIT": {"bg": (2, 2, 4), "grad0": (16, 24, 36), "grad1": (4, 10, 18), "halo": 1.35},
        }[name]

        # ---- small debris for scale (few, faint, near-field) ----------------
        debris_count = {"DEEP SPACE": 2, "EARTH ORBIT": 6, "LUNAR VICINITY": 4, "SUNLIT ORBIT": 5}[name]
        self._debris = [
            (
                rng.uniform(-40.0, 40.0),
                rng.uniform(-26.0, 26.0),
                rng.uniform(24.0, 60.0),
                rng.uniform(0.4, 1.3),
                rng.uniform(0.0, math.tau),
            )
            for _ in range(debris_count)
        ]

    def set_environment(self, name):
        """Switch the visual environment preset. Visualization-only."""
        if str(name).upper() not in self.ENVIRONMENT_PRESETS:
            return
        self._build_environment(str(name).upper())
        self.update()

    def new_sky(self):
        """Regenerate the SAME environment with a fresh per-run visual seed.

        Only the decorative environment (stars, nebulae, clusters, debris,
        celestial placement) is regenerated. Simulation/benchmark data is not
        touched, so tracking results remain fully deterministic.
        """
        self._environment_seed = random.randrange(1, 10_000_000)
        self._build_environment(self.environment_name)
        self.update()

    # --------------------------- state/synchronization ---------------------------
    def set_state(
        self,
        *,
        camera_angles,
        target_angles,
        predicted_angles=None,
        range_m=1000.0,
        targets: Iterable[dict] | None = None,
        world_targets: Iterable[dict] | None = None,
        camera_position=(0.0, 0.0, 0.0),
        designated_id="TGT-01",
        state="SEARCHING",
        frame_count=0,
        sim_time=0.0,
        fps=0.0,
        motion_pattern="Straight Line",
        turbulence_enabled=False,
        turbulence_strength=0.0,
        jitter_enabled=False,
        jitter_strength=0.0,
        platform_enabled=False,
        platform_strength=0.0,
        noise_name="None",
        atmosphere_name="Clear",
        atmosphere_level=0,
        platform_offset_px=(0.0, 0.0),
        jitter_px=(0.0, 0.0),
        beam_wander_px=(0.0, 0.0),
        link_ready=False,
    ):
        self.camera_angles = (float(camera_angles[0]), float(camera_angles[1]))
        self.target_angles = (float(target_angles[0]), float(target_angles[1]))
        self.predicted_angles = None if predicted_angles is None else (float(predicted_angles[0]), float(predicted_angles[1]))
        self.range_m = float(range_m)
        self.world_targets = [dict(item) for item in (world_targets or targets or [])]
        self.targets = [dict(item) for item in (targets or self.world_targets)]
        self.camera_position = tuple(float(v) for v in camera_position)
        self.designated_id = str(designated_id)
        self.state = str(state)
        self.frame_count = int(frame_count)
        self.sim_time = float(sim_time)
        self.fps = float(fps)
        self.motion_pattern = str(motion_pattern)
        self.turbulence_enabled = bool(turbulence_enabled)
        self.turbulence_strength = float(turbulence_strength)
        self.jitter_enabled = bool(jitter_enabled)
        self.jitter_strength = float(jitter_strength)
        self.platform_enabled = bool(platform_enabled)
        self.platform_strength = float(platform_strength)
        self.noise_name = str(noise_name)
        self.atmosphere_name = str(atmosphere_name)
        self.atmosphere_level = int(atmosphere_level)
        self.platform_offset_px = tuple(float(v) for v in platform_offset_px)
        self.jitter_px = tuple(float(v) for v in jitter_px)
        self.beam_wander_px = tuple(float(v) for v in beam_wander_px)
        self.link_ready = bool(link_ready)

        for t in self.world_targets:
            tid = str(t.get("id", ""))
            if not tid:
                continue
            pos = self._world_position_from_raw(t)
            self._target_trails.setdefault(tid, deque(maxlen=70)).append(pos)
        self._camera_trail.append(self.camera_position)
        self.update()

    # ------------------- visualization-only scene/view controls ------------------
    SCENE_OFFSET_LIMIT = 24.0
    SCENE_SEPARATION_MIN = 0.3
    SCENE_SEPARATION_MAX = 3.0

    def set_scene_camera_offset(self, dx, dy, dz):
        """Shift the tracking satellite visually. Pure visualization."""
        limit = self.SCENE_OFFSET_LIMIT
        self.scene_camera_offset = [
            max(-limit, min(limit, float(dx))),
            max(-limit, min(limit, float(dy))),
            max(-limit, min(limit, float(dz))),
        ]
        self.update()

    def set_scene_target_offset(self, dx, dy, dz):
        """Shift the beacon/target satellites visually. Pure visualization."""
        limit = self.SCENE_OFFSET_LIMIT
        self.scene_target_offset = [
            max(-limit, min(limit, float(dx))),
            max(-limit, min(limit, float(dy))),
            max(-limit, min(limit, float(dz))),
        ]
        self.update()

    def set_scene_separation(self, scale):
        """Scale inter-satellite separation visually (1.0 = live geometry)."""
        self.scene_separation = max(
            self.SCENE_SEPARATION_MIN, min(self.SCENE_SEPARATION_MAX, float(scale))
        )
        self.update()

    def scene_offsets_active(self) -> bool:
        return (
            self.scene_separation != 1.0
            or any(abs(v) > 1e-9 for v in self.scene_camera_offset)
            or any(abs(v) > 1e-9 for v in self.scene_target_offset)
        )

    def reset_scene(self):
        """Restore the visualization scene to the live synchronized geometry."""
        self.scene_camera_offset = [0.0, 0.0, 0.0]
        self.scene_target_offset = [0.0, 0.0, 0.0]
        self.scene_separation = 1.0
        self.update()

    def reset_view(self):
        """Restore the default user viewpoint (independent of PAT)."""
        self._stop_focus_animation()
        self._view_mode = "FULL SCENE"
        self.beam_impact_enabled = False
        self.view_yaw = -38.0
        self.view_pitch = 16.0
        self.zoom = 1.05
        self.update()
        self.focus_requested.emit("FULL SCENE")

    # ---------------------------- beam impact view ---------------------------
    BEAM_IMPACT_ZOOM = 2.5
    BEAM_ALIGNED_ERROR_PX = 10.0   # PS coarse-alignment limit
    BEAM_IMPACT_CLAMP_FACTOR = 2.6  # draw impacts beyond this ON the clamp ring

    def set_beam_impact_data(self, center_error_px, offset_px=(0.0, 0.0), link_ready=False):
        """Receive the LIVE pointing result from the mission (read-only).

        ``center_error_px`` is the real center error the mission measures;
        ``offset_px`` is the real tracker-vs-frame-center offset in 640x480
        pixel units. This drives only the impact rendering — it can never
        influence tracking, the controller, or any measurement.
        """
        if center_error_px is None:
            self.beam_impact_center_error_px = None
            self.beam_impact_offset_px = (0.0, 0.0)
            self.beam_impact_state = "BEAM MISALIGNED"
        else:
            self.beam_impact_center_error_px = float(center_error_px)
            self.beam_impact_offset_px = (float(offset_px[0]), float(offset_px[1]))
            self.beam_impact_state = (
                "BEAM ALIGNED"
                if self.beam_impact_center_error_px <= self.BEAM_ALIGNED_ERROR_PX
                else "BEAM MISALIGNED"
            )
        self.beam_impact_link_ready = bool(link_ready)
        self.update()

    def beam_impact_status(self):
        """(status_text, center_error_px_or_None) derived from live pointing."""
        return self.beam_impact_state, self.beam_impact_center_error_px

    def toggle_beam_impact(self):
        """Enter the BEAM IMPACT view: smoothly fly to a viewpoint behind the
        receiving terminal so the incoming beam, its impact footprint and the
        aperture are all visible. Uses the SAME eased viewpoint animation as
        the terminal focus modes; user orbit stays available at all times."""
        if self.beam_impact_enabled:
            self.focus_full_scene()
            return
        designated = next(
            (t for t in self.world_targets if str(t.get("id")) == self.designated_id),
            None,
        )
        raw = self._world_position_from_raw(designated) if designated else (0.0, 0.0, 9.0)
        target = self._vis_target_pos(raw)
        # Orbit to the far side of the beacon relative to the camera so the
        # beam arrives toward the viewer and the aperture face is visible.
        cx, cy, cz = self.camera_position
        dx, dy, dz = target[0] - cx, target[1] - cy, target[2] - cz
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        yaw = math.degrees(math.atan2(-dx / norm, -dz / norm))
        _, pitch, zoom = self._world_to_view_angles(target, zoom=self.BEAM_IMPACT_ZOOM)
        pitch = max(-70.0, min(70.0, pitch - 18.0))
        self._animate_view_to(yaw, pitch, zoom, "BEAM IMPACT")
        self.beam_impact_enabled = True
        self.focus_requested.emit("BEAM IMPACT")

    # ------------------------------ focus modes ------------------------------
    def _stop_focus_animation(self):
        if self._focus_anim is not None:
            self._focus_anim.stop()
            self._focus_anim.deleteLater()
            self._focus_anim = None
        self._focus_from = None
        self._focus_to = None

    def _animate_view_to(self, yaw, pitch, zoom, mode):
        """Smoothly drive the SAME independent user viewpoint to a target
        pose. Never teleports: an eased interpolation of the real orbit
        variables, abortable by the user grabbing the scene (mouse orbit).
        """
        self._stop_focus_animation()
        self._view_mode = mode
        self._focus_from = (self.view_yaw, self.view_pitch, self.zoom)
        self._focus_to = (float(yaw), float(pitch), float(zoom))
        self._focus_t = 0.0
        self._focus_anim = QTimer(self)
        self._focus_anim.setInterval(16)
        timeout = QTimer(self)
        timeout.setSingleShot(True)
        timeout.timeout.connect(self._stop_focus_animation)
        timeout.start(2000)
        self._focus_timeout = timeout

        def _step():
            if self._focus_to is None:
                return
            self._focus_t = min(1.0, self._focus_t + 0.05)
            # Ease-out cubic for a natural cinematic settle.
            k = 1.0 - (1.0 - self._focus_t) ** 3
            fy, fp, fz = self._focus_from
            ty, tp, tz = self._focus_to
            self.view_yaw = fy + (ty - fy) * k
            self.view_pitch = max(-70.0, min(70.0, fp + (tp - fp) * k))
            self.zoom = max(0.55, min(3.8, fz + (tz - fz) * k))
            if self._focus_t >= 1.0:
                self._stop_focus_animation()
            self.update()

        self._focus_anim.timeout.connect(_step)
        self._focus_anim.start()

    def _world_to_view_angles(self, world_pos, distance=6.0, zoom=1.9):
        """View yaw/pitch that place the viewer close to ``world_pos`` looking
        at the scene origin (the satellites), for terminal close-ups."""
        x, y, z = world_pos
        horizontal = math.hypot(x, z)
        yaw = math.degrees(math.atan2(x, z)) if horizontal > 1e-6 else 0.0
        pitch = math.degrees(math.atan2(y, max(horizontal, 1e-6)))
        pitch = max(-70.0, min(70.0, pitch))
        return yaw, pitch, zoom

    def focus_camera_terminal(self):
        """FOCUS CAMERA TERMINAL: close-up of the transmitting optical
        terminal with the outgoing beam direction unchanged (the beam is
        always drawn along the live PAT/aim state in paintEvent)."""
        target = self._vis_camera_pos()
        yaw, pitch, zoom = self._world_to_view_angles(target, zoom=2.05)
        self._animate_view_to(yaw, pitch, zoom, "CAMERA TERMINAL")
        self.focus_requested.emit("CAMERA TERMINAL")

    def focus_beacon_terminal(self):
        """FOCUS BEACON TERMINAL: close-up of the receiving side with the
        arriving beam direction unchanged."""
        designated = next(
            (t for t in self.world_targets if str(t.get("id")) == self.designated_id),
            None,
        )
        raw = self._world_position_from_raw(designated) if designated else (0.0, 0.0, 9.0)
        target = self._vis_target_pos(raw)
        yaw, pitch, zoom = self._world_to_view_angles(target, zoom=2.05)
        self._animate_view_to(yaw, pitch, zoom, "BEACON TERMINAL")
        self.focus_requested.emit("BEACON TERMINAL")

    def focus_full_scene(self):
        """RETURN TO FULL SCENE: restore the independent free-orbit view."""
        self.reset_view()

    def _vis_camera_pos(self):
        """Camera satellite position as drawn (raw + visual offset)."""
        ox, oy, oz = self.scene_camera_offset
        x, y, z = self.camera_position
        return (x + ox, y + oy, z + oz)

    def _vis_target_pos(self, pos):
        """Target position as drawn: separation scaled about the RAW camera
        position, then shifted by the visual target offset."""
        cx, cy, cz = self.camera_position
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        sx = cx + (px - cx) * self.scene_separation
        sy = cy + (py - cy) * self.scene_separation
        sz = cz + (pz - cz) * self.scene_separation
        ox, oy, oz = self.scene_target_offset
        return (sx + ox, sy + oy, sz + oz)

    # ------------------------------ 3D projection --------------------------------
    def _rotate_view(self, p):
        x, y, z = p
        yaw = math.radians(self.view_yaw)
        pitch = math.radians(self.view_pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy
        cp, sp = math.cos(pitch), math.sin(pitch)
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp
        return x1, y2, z2

    def _project(self, p):
        x, y, z = self._rotate_view(p)
        depth = max(0.65, z + 64.0)
        scale = min(self.width(), self.height()) * 1.18 * self.zoom
        sx = self.width() * 0.50 + x * scale / depth
        sy = self.height() * 0.52 - y * scale / depth
        return QPointF(sx, sy), depth

    def _world_position_from_raw(self, target):
        # Local inertial visualization coordinates derived directly from the
        # simulator's raw target trajectory. This preserves selected movement
        # shape without making the 3D viewport follow the optical camera.
        x = float(target.get("x", config.FRAME_WIDTH / 2.0))
        y = float(target.get("y", config.FRAME_HEIGHT / 2.0))
        rng_m = float(target.get("range_m", self.range_m))
        wx = (x - config.FRAME_WIDTH / 2.0) / 14.5
        wy = (config.FRAME_HEIGHT / 2.0 - y) / 13.0
        wz = 8.5 + max(-2.5, min(3.5, (rng_m - 1000.0) / 260.0))
        return (wx, wy, wz)

    def _projected_prediction(self, world_targets):
        designated = next((t for t in world_targets if str(t.get("id")) == self.designated_id), None)
        if designated is None or self.predicted_angles is None:
            return None
        # Small forward prediction in local world space, using Kalman angular
        # lead only as an offset from the actual target. It does not change the
        # underlying target motion.
        pp, pt = self.predicted_angles
        tx, ty = self.camera_angles
        dx = (pp - tx) / max(config.FOV_HORIZONTAL, 0.1) * 4.8
        dy = -(pt - ty) / max(config.FOV_VERTICAL, 0.1) * 4.2
        p = self._world_position_from_raw(designated)
        return (p[0] + dx * 0.16, p[1] + dy * 0.16, p[2])

    # ------------------------- direct manipulation -------------------------
    def _hit_test_targets(self, pos, radius=22.0):
        """Nearest projected satellite within grab radius: (target, point)."""
        best = None
        for target in self.world_targets:
            if not target.get("visible", True):
                continue
            sp, _depth = self._project(self._world_position_from_raw(target))
            if 0 <= sp.x() <= self.width() and 0 <= sp.y() <= self.height():
                d = math.hypot(sp.x() - pos.x(), sp.y() - pos.y())
                if d <= radius and (best is None or d < best[2]):
                    best = (target, sp, d)
        return (best[0], best[1]) if best else None

    def _emit_drag_delta(self, pos):
        target = self._grabbed_target()
        if target is None:
            return
        # Screen px -> image px, the exact inverse of the twin's world
        # mapping factors in _world_position_from_raw (14.5 x, 13.0 y).
        # Image y grows DOWNWARD, so a downward drag increases y — no flip.
        dx = (pos.x() - self._object_drag["screen"].x()) / 14.5
        dy = (pos.y() - self._object_drag["screen"].y()) / 13.0
        self._object_drag["screen"] = pos
        self.dragDelta.emit(str(target.get("id")), float(dx), float(dy))

    def _grabbed_target(self):
        return next(
            (t for t in self.world_targets
             if str(t.get("id")) == self._object_drag["id"]),
            None,
        )

    # --------------------- optical-terminal aim / boresight ----------------------
    def _boresight_direction(self, camera_world):
        """PAT camera boresight as a world-space unit vector, READ-ONLY from the
        live pan/tilt. Scene convention: +X right, +Y up, +Z toward the viewer;
        pan (deg) rotates about +Y, tilt (deg) elevates toward +Y."""
        pan, tilt = self.camera_angles
        pr, tr = math.radians(pan), math.radians(tilt)
        dx = math.cos(tr) * math.sin(pr)
        dy = math.sin(tr)
        dz = math.cos(tr) * math.cos(pr)
        norm = math.hypot(dx, dy, dz) or 1.0
        return (dx / norm, dy / norm, dz / norm)

    def _aim_unit(self, camera_world, aim_point):
        """Unit vector from the true camera position toward the aim point."""
        cx, cy, cz = self.camera_position
        dx = aim_point[0] - cx
        dy = aim_point[1] - cy
        dz = aim_point[2] - cz
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm < 1e-9:
            return self._boresight_direction(camera_world)
        return (dx / norm, dy / norm, dz / norm)

    def _terminal_world_pos(self, camera_world, aim_unit):
        """World position of the optical terminal lens, displaced from the
        spacecraft body centre along the aim direction (visibly attached)."""
        arm = 1.05
        return (
            camera_world[0] + aim_unit[0] * arm,
            camera_world[1] + aim_unit[1] * arm,
            camera_world[2] + aim_unit[2] * arm,
        )

    def _draw_scene_axes(self, painter):
        """Compact axes gizmo (bottom-left): scene convention clarity."""
        origin = (0.0, 0.0, 0.0)
        ox, _ = self._project(origin)
        axes = (
            ((7.0, 0.0, 0.0), QColor(96, 220, 130), "+X BEACONS"),
            ((0.0, 7.0, 0.0), QColor(96, 190, 245), "+Y UP"),
            ((0.0, 0.0, 7.0), QColor(235, 205, 105), "+Z VIEWER"),
        )
        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
        for direction, color, label in axes:
            endpoint, _ = self._project(direction)
            painter.setPen(QPen(color, 2.0))
            painter.drawLine(ox, endpoint)
            painter.setPen(color)
            painter.drawText(int(endpoint.x()) + 4, int(endpoint.y()), label)

    # ----------------------------- environment rendering ------------------------
    def _draw_background(self, painter):
        rect = self.rect()
        light = self._lighting
        painter.fillRect(rect, QColor(*light["bg"]))

        grad = QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.92)
        grad.setColorAt(0.0, QColor(*light["grad0"]))
        grad.setColorAt(0.46, QColor(*light["grad1"]))
        grad.setColorAt(1.0, QColor(*light["bg"]))
        painter.fillRect(rect, QBrush(grad))

        # Very faint galactic dust band for depth. It is intentionally subtle
        # so stars remain discrete rather than becoming a space "effect".
        painter.save()
        painter.setOpacity(0.035)
        painter.setPen(QPen(QColor(180, 205, 235), 56))
        painter.drawArc(
            QRectF(-280, self.height() * 0.18, self.width() + 560, self.height() * 0.90),
            8 * 16,
            168 * 16,
        )
        painter.restore()

    def _nebula_screen_points(self):
        """Screen-space centers for nebulae/clusters: slow parallax against the
        user orbit (they are far background, so they move at a fraction of the
        projected foreground motion)."""
        yaw = math.radians(self.view_yaw)
        pitch = math.radians(self.view_pitch)
        cx, cy = self.width() * 0.5, self.height() * 0.52
        parallax = min(self.width(), self.height()) * 0.16 * self.zoom
        return (
            cx - yaw * parallax * 0.55,
            cy - pitch * parallax * 0.75,
        )

    def _draw_environment_layers(self, painter):
        """Deep-space structures: nebulae, star clusters, drifting debris.
        Subtle by design so the beacon/optical link stays the visual focus."""
        frame = self.frame_count
        bx, by = self._nebula_screen_points()
        w, h = self.width(), self.height()

        # Nebulae: two overlapping soft radial blobs each (cheap, no noise).
        for nx, ny, nz, nr, (cr, cg, cb), base_alpha in self._nebulae:
            drift = math.sin(frame * 0.0021 + nz) * 6.0
            sx = bx + nx * 7.0 + drift
            sy = by - ny * 7.0
            radius = max(30.0, nr * 7.5)
            if sx < -radius or sx > w + radius or sy < -radius or sy > h + radius:
                continue
            for frac, alpha_scale in ((0.0, 1.0), (0.55, 0.55)):
                grad = QRadialGradient(QPointF(sx + frac * radius * 0.4, sy - frac * radius * 0.3),
                                       radius * (1.0 - frac * 0.35))
                grad.setColorAt(0.0, QColor(cr, cg, cb, int(base_alpha * 255 * alpha_scale)))
                grad.setColorAt(1.0, QColor(cr, cg, cb, 0))
                painter.setBrush(QBrush(grad))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(sx, sy), radius * (1.0 - frac * 0.35), radius * 0.8)

        # Star clusters: dense kernels of points around each cluster center.
        painter.setPen(Qt.NoPen)
        for gx, gy, gz, gr in self._clusters:
            sx = bx + gx * 7.0
            sy = by - gy * 7.0
            spread = max(14.0, gr * 8.0)
            for i in range(14):
                ang = (i * 2.39996 + gz) % math.tau
                rr = spread * math.sqrt((i + 0.5) / 14.0)
                px, py = sx + math.cos(ang) * rr, sy + math.sin(ang) * rr * 0.7
                tw = 0.75 + 0.25 * math.sin(frame * 0.06 + i * 1.7 + gx)
                painter.setBrush(QColor(226, 238, 250, int(150 * tw)))
                painter.drawEllipse(QPointF(px, py), 1.0, 1.0)

        # Debris specks: near-field, slow tumble + orbit drift for scale.
        for dx, dy, dz, size, phase in self._debris:
            ang = phase + frame * 0.0035
            px3 = dx * math.cos(ang) - dz * 0.12 * math.sin(ang)
            pz3 = dz * math.cos(ang) + dx * math.sin(ang) * 0.12
            p, depth = self._project((px3, dy + math.sin(frame * 0.008 + phase) * 1.5, pz3))
            if not (-8 <= p.x() <= w + 8 and -8 <= p.y() <= h + 8):
                continue
            scale = max(0.8, size * min(1.6, 40.0 / max(depth, 1.0)))
            painter.setBrush(QColor(148, 160, 172, 130))
            painter.drawEllipse(p, scale, scale * 0.62)

    def _draw_starfield(self, painter):
        frame = self.frame_count
        bx, by = self._nebula_screen_points()
        w, h = self.width(), self.height()
        for x, y, z, brightness, tint, tw_speed, tw_phase in self._stars:
            # Layered parallax: distant stars barely move with the orbit,
            # near-field stars move slightly more (depth cue).
            px = bx + x * 7.0
            py = by - y * 7.0
            if not (-6 <= px <= w + 6 and -6 <= py <= h + 6):
                continue
            radius = 0.55 + 1.45 * brightness * max(0.38, min(1.25, 46.0 / max(z + 64.0, 1.0)))
            twinkle = 0.72 + 0.28 * math.sin(frame * tw_speed + tw_phase)
            alpha = int((70 + 170 * brightness) * twinkle)
            if tint == "cool":
                c = QColor(182, 218, 255, alpha)
            elif tint == "warm":
                c = QColor(255, 225, 195, alpha)
            else:
                c = QColor(238, 244, 250, alpha)
            painter.setPen(QPen(c, radius))
            painter.drawPoint(QPointF(px, py))

    def _draw_celestial_body(self, painter, body):
        p, depth = self._project(body["pos"])
        base_scale = min(self.width(), self.height()) * self.zoom / max(depth, 1.0)
        radius = max(2.0, body["r"] * base_scale)
        if p.x() < -radius * 1.4 or p.x() > self.width() + radius * 1.4 or p.y() < -radius * 1.4 or p.y() > self.height() + radius * 1.4:
            return

        kind = body["kind"]
        if kind == "sun":
            # Distant sun disc + halo scaled by the environment preset.
            halo_r = radius * 3.8 * self._lighting.get("halo", 1.0)
            glow = QRadialGradient(p, halo_r)
            glow.setColorAt(0.0, QColor(255, 235, 174, 155))
            glow.setColorAt(0.28, QColor(255, 216, 120, 72))
            glow.setColorAt(1.0, QColor(255, 190, 80, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(p, halo_r, halo_r)
            disk = QRadialGradient(p - QPointF(radius * 0.28, radius * 0.28), radius * 1.15)
            disk.setColorAt(0.0, QColor(255, 250, 221, 250))
            disk.setColorAt(0.65, QColor(255, 228, 150, 245))
            disk.setColorAt(1.0, QColor(222, 169, 75, 230))
            painter.setBrush(QBrush(disk))
            painter.setPen(QPen(QColor(255, 241, 188, 180), 0.7))
            painter.drawEllipse(p, radius, radius)
            painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
            painter.setPen(QColor(244, 214, 150, 180))
            painter.drawText(int(p.x() + radius + 5), int(p.y() + 2), "SUN")
            return

        if kind == "earth":
            # Procedural shaded Earth with an atmospheric rim and a few land
            # masses. No clouds/fog/rain are rendered in this space view.
            light = QRadialGradient(p - QPointF(radius * 0.28, radius * 0.24), radius * 1.15)
            light.setColorAt(0.0, QColor(83, 164, 213, 245))
            light.setColorAt(0.58, QColor(33, 93, 149, 245))
            light.setColorAt(1.0, QColor(5, 21, 45, 255))
            painter.setBrush(QBrush(light))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(p, radius, radius)

            # Night-side city glow, only as sparse points on the dark limb.
            painter.save()
            painter.setClipRegion(painter.clipRegion())
            for fracx, fracy in [(-0.48, 0.18), (-0.30, 0.34), (0.08, 0.40), (0.26, 0.18), (0.42, -0.04), (-0.02, -0.24)]:
                cx = p.x() + fracx * radius
                cy = p.y() + fracy * radius
                if (cx - p.x()) ** 2 + (cy - p.y()) ** 2 < (0.86 * radius) ** 2:
                    painter.setPen(QPen(QColor(255, 201, 112, 110), max(1.0, radius * 0.013)))
                    painter.drawPoint(QPointF(cx, cy))
            painter.restore()

            painter.save()
            painter.setOpacity(0.72)
            painter.setPen(QPen(QColor(81, 148, 115, 150), max(0.7, radius * 0.014)))
            for cx, cy, rx, ry, rot in self._earth_land:
                center = QPointF(p.x() + cx * radius, p.y() + cy * radius)
                painter.save()
                painter.translate(center)
                painter.rotate(math.degrees(rot))
                painter.drawEllipse(QRectF(-rx * radius, -ry * radius, 2 * rx * radius, 2 * ry * radius))
                painter.restore()
            painter.restore()

            rim = QRadialGradient(p, radius * 1.08)
            rim.setColorAt(0.82, QColor(120, 215, 245, 0))
            rim.setColorAt(0.96, QColor(128, 218, 255, 100))
            rim.setColorAt(1.0, QColor(128, 218, 255, 20))
            painter.setBrush(QBrush(rim))
            painter.setPen(QPen(QColor(140, 226, 255, 150), max(0.7, radius * 0.012)))
            painter.drawEllipse(p, radius * 1.02, radius * 1.02)
        else:
            light = QRadialGradient(p - QPointF(radius * 0.25, radius * 0.25), radius * 1.12)
            light.setColorAt(0.0, QColor(208, 213, 220, 230))
            light.setColorAt(0.64, QColor(130, 137, 148, 220))
            light.setColorAt(1.0, QColor(50, 55, 64, 240))
            painter.setBrush(QBrush(light))
            painter.setPen(QPen(QColor(210, 218, 225, 150), max(0.6, radius * 0.02)))
            painter.drawEllipse(p, radius, radius)
            for crater_x, crater_y, crater_r in [(-0.22, -0.12, 0.09), (0.18, 0.21, 0.065), (0.35, -0.18, 0.05)]:
                painter.setBrush(QBrush(QColor(72, 76, 84, 85)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(p.x() + crater_x * radius, p.y() + crater_y * radius), radius * crater_r, radius * crater_r)

        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
        painter.setPen(QColor(184, 204, 216, 160))
        painter.drawText(int(p.x() + radius + 5), int(p.y() + 2), body["name"])

    # ----------------------------- spacecraft drawing ---------------------------
    def _draw_satellite(self, painter, pos, name, designated=False, camera_unit=False, aim_unit=None):
        p, depth = self._project(pos)
        scale = min(self.width(), self.height()) * self.zoom / max(depth, 1.0)
        body_w = max(18.0, 22.0 * scale / 34.0)
        body_h = max(10.0, 13.0 * scale / 34.0)
        if camera_unit and aim_unit is not None:
            # Terminal drawn at its true 3D attachment point along the aim
            # direction; the gimbal/lens visually points where the system aims.
            terminal_pos = self._terminal_world_pos(pos, aim_unit)
            tp_screen, _ = self._project(terminal_pos)
            painter.setPen(QPen(QColor(150, 168, 180, 200), 2.0))
            painter.drawLine(p, tp_screen)  # gimbal arm from bus to terminal

        # Main spacecraft bus. Faceted silhouette gives a better 3D read than a box.
        body = QPolygonF([
            QPointF(p.x() - body_w, p.y() - body_h * 0.55),
            QPointF(p.x() - body_w * 0.32, p.y() - body_h),
            QPointF(p.x() + body_w * 0.96, p.y() - body_h * 0.64),
            QPointF(p.x() + body_w, p.y() + body_h * 0.62),
            QPointF(p.x() - body_w * 0.52, p.y() + body_h),
        ])
        painter.setBrush(QBrush(QColor(55, 66, 75, 248)))
        painter.setPen(QPen(QColor(181, 195, 207, 222), 1.0))
        painter.drawPolygon(body)

        # Thermal/radiator face.
        painter.setBrush(QBrush(QColor(28, 35, 43, 245)))
        painter.setPen(QPen(QColor(104, 123, 138, 170), 0.8))
        painter.drawPolygon(QPolygonF([
            QPointF(p.x() - body_w * 0.98, p.y() - body_h * 0.47),
            QPointF(p.x() - body_w * 0.54, p.y() - body_h * 0.78),
            QPointF(p.x() - body_w * 0.54, p.y() + body_h * 0.78),
            QPointF(p.x() - body_w * 0.98, p.y() + body_h * 0.47),
        ]))

        # Solar arrays with cell strips.
        for side in (-1, 1):
            x0 = p.x() + side * body_w * 1.02
            panel = QPolygonF([
                QPointF(x0, p.y() - body_h * 0.74),
                QPointF(x0 + side * body_w * 2.05, p.y() - body_h * 0.54),
                QPointF(x0 + side * body_w * 2.05, p.y() + body_h * 0.54),
                QPointF(x0, p.y() + body_h * 0.74),
            ])
            painter.setBrush(QBrush(QColor(13, 29, 50, 245)))
            painter.setPen(QPen(QColor(75, 128, 169, 225), 0.9))
            painter.drawPolygon(panel)
            for i in range(1, 5):
                frac = i / 5.0
                xx = x0 + side * body_w * 2.05 * frac
                painter.setPen(QPen(QColor(74, 119, 151, 150), 0.5))
                painter.drawLine(QPointF(xx, p.y() - body_h * 0.56), QPointF(xx, p.y() + body_h * 0.56))
            for frac in (-0.33, 0.0, 0.33):
                painter.setPen(QPen(QColor(89, 147, 180, 115), 0.45))
                painter.drawLine(
                    QPointF(x0 + side * body_w * 0.18, p.y() + frac * body_h),
                    QPointF(x0 + side * body_w * 1.92, p.y() + frac * body_h),
                )

        # Small attitude-control / star-tracker details.
        painter.setPen(QPen(QColor(208, 216, 222, 150), 0.7))
        painter.drawLine(QPointF(p.x(), p.y() - body_h), QPointF(p.x(), p.y() - body_h * 1.45))
        painter.setBrush(QBrush(QColor(175, 186, 192, 210)))
        painter.drawEllipse(QPointF(p.x(), p.y() - body_h * 1.55), 1.5, 1.5)

        if camera_unit:
            # Optical terminal: gimbal housing, barrel along the aim direction,
            # dark lens aperture and bright optical centre at the muzzle end.
            gimbal_c = QPointF(p.x() + body_w * 1.02, p.y())
            painter.setPen(QPen(QColor(104, 143, 158, 190), 1.0))
            painter.setBrush(QBrush(QColor(19, 30, 38, 250)))
            painter.drawEllipse(gimbal_c, max(7.0, body_w * 0.58), max(7.0, body_w * 0.58))
            lens = QPointF(gimbal_c.x() + body_w * 0.18, gimbal_c.y())
            painter.setBrush(QBrush(QColor(5, 11, 18, 255)))
            painter.setPen(QPen(QColor(108, 217, 243, 220), 1.1))
            painter.drawEllipse(lens, max(4.0, body_w * 0.34), max(4.0, body_w * 0.34))
            # Muzzle glow so the transmit aperture reads as an active optical port.
            glow_r = max(12.0, body_w * 1.1)
            lens_glow = QRadialGradient(lens, glow_r)
            lens_glow.setColorAt(0.0, QColor(120, 240, 255, 60))
            lens_glow.setColorAt(1.0, QColor(120, 240, 255, 0))
            painter.setBrush(QBrush(lens_glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(lens, glow_r, glow_r)
            painter.setBrush(QBrush(QColor(175, 245, 255, 235)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(lens, max(1.4, body_w * 0.10), max(1.4, body_w * 0.10))

            if aim_unit is not None:
                terminal_pos = self._terminal_world_pos(pos, aim_unit)
                t_screen, t_depth = self._project(terminal_pos)
                t_scale = min(self.width(), self.height()) * self.zoom / max(t_depth, 1.0)
                tw = max(6.5, 8.0 * t_scale / 34.0)
                th = max(4.0, 5.0 * t_scale / 34.0)
                # Barrel: short thick stub pointing along the aim direction.
                back = QPointF(
                    t_screen.x() - aim_unit[0] * tw * 1.35,
                    t_screen.y() + aim_unit[1] * tw * 1.35,
                )
                painter.setPen(QPen(QColor(206, 220, 230, 235), 1.0))
                painter.setBrush(QBrush(QColor(31, 44, 54, 250)))
                painter.drawEllipse(back, tw * 0.62, th * 0.62)
                painter.setPen(QPen(QColor(96, 220, 245, 210), 1.1))
                painter.drawLine(back, t_screen)
                # Lens aperture ring at the muzzle.
                painter.setBrush(QBrush(QColor(4, 12, 20, 255)))
                painter.setPen(QPen(QColor(110, 232, 255, 235), 1.2))
                painter.drawEllipse(t_screen, tw * 0.55, th * 0.55)
                painter.setBrush(QBrush(QColor(190, 250, 255, 240)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(t_screen, max(1.2, tw * 0.16), max(1.2, th * 0.16))
                self.terminal_anchor_screen = QPointF(t_screen)  # diagnostic record of lens position

            # A small high-gain antenna on the rear.
            dish = QPointF(p.x() - body_w * 1.15, p.y() - body_h * 1.0)
            painter.setPen(QPen(QColor(158, 170, 181, 190), 0.8))
            painter.setBrush(QBrush(QColor(37, 48, 57, 235)))
            painter.drawEllipse(dish, max(3.5, body_w * 0.34), max(2.5, body_w * 0.24))

        if designated:
            beacon = QPointF(p.x(), p.y() - body_h * 1.78)
            pulse = 1.0 + 0.15 * math.sin(self.frame_count * 0.24)
            r = max(6.5, body_w * 0.45) * pulse
            glow = QRadialGradient(beacon, r * 4.5)
            glow.setColorAt(0.0, QColor(48, 255, 112, 220))
            glow.setColorAt(0.28, QColor(44, 255, 100, 105))
            glow.setColorAt(1.0, QColor(45, 255, 100, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(beacon, r * 4.5, r * 4.5)
            painter.setBrush(QBrush(QColor(68, 255, 125, 255)))
            painter.setPen(QPen(QColor(219, 255, 228, 245), 1.0))
            painter.drawEllipse(beacon, r, r)
            painter.setPen(QPen(QColor(126, 255, 169, 120), 0.9))
            painter.drawEllipse(beacon, r * 1.7, r * 1.7)

        # ---- name plate with collision-aware placement -------------------
        # Labels claim screen space here; other annotations consult these
        # claimed boxes and shift/drop themselves instead of stacking on top
        # of the name plates (Stage 7 label management).
        label_w = max(170, len(name) * 6.2 + 22)
        label_h = 24.0
        anchor_y = p.y() + body_h * 1.72
        label_rect = QRectF(p.x() - label_w * 0.5, anchor_y, label_w, label_h)
        self._resolve_label_overlap(label_rect, 8.0)
        painter.setBrush(QBrush(QColor(2, 8, 14, 238)))
        painter.setPen(QPen(QColor(70, 96, 111, 205), 0.8))
        painter.drawRoundedRect(label_rect, 6, 6)
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(240, 247, 251))
        painter.drawText(label_rect, Qt.AlignCenter, name)
        self._claim_label_box(label_rect)
        return p

    # --------------------- Stage 7: label management ----------------------
    def _resolve_label_overlap(self, rect, margin=6.0):
        """Shift ``rect`` (in place) until it stops colliding with labels that
        claimed space earlier this frame. Priority = draw order: name plates
        claim first, so secondary annotations move — never the plates."""
        claimed = getattr(self, "_label_boxes", None)
        if not claimed:
            return rect
        for _ in range(24):
            hit = next((other for other in claimed
                        if rect.intersects(other.adjusted(-margin, -margin, margin, margin))), None)
            if hit is None:
                return rect
            # Prefer dropping straight below; if that would leave the widget,
            # flip above the occupied box instead.
            dy = hit.bottom() + margin - rect.top()
            if rect.bottom() + dy <= self.height() - 2:
                rect.translate(0.0, dy)
                continue
            dy_up = rect.bottom() - (hit.top() - margin)
            if rect.top() - dy_up >= 2:
                rect.translate(0.0, -dy_up)
                continue
            # Sideways nudge as a last resort (alternating direction).
            direction = 1.0 if (len(claimed) % 2 == 0) else -1.0
            rect.translate(direction * (rect.width() * 0.45 + margin), 0.0)
        return rect

    def _claim_label_box(self, rect):
        self._label_boxes.append(QRectF(rect))

    def _draw_secondary_label(self, painter, anchor, text, color):
        """Priority-placed secondary annotation with a leader line.

        Secondary labels draw AFTER the name plates and yield to them: if no
        free space exists near the anchor they are simply not drawn, so they
        never stack over important information.
        """
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 10
        th = fm.height() + 6
        rect = QRectF(anchor.x() + 10.0, anchor.y() - th - 6.0, tw, th)
        self._resolve_label_overlap(rect, 6.0)
        if rect.right() > self.width() - 2 or rect.bottom() > self.height() - 2 or rect.top() < 2:
            return  # no readable space left this frame — drop, don't overlap
        # Leader line from the anchor to the moved label keeps ownership clear.
        painter.setPen(QPen(color, 0.9))
        painter.drawLine(anchor, QPointF(rect.left(), rect.center().y()))
        painter.setBrush(QBrush(QColor(2, 8, 14, 205)))
        painter.drawRoundedRect(rect, 4, 4)
        painter.drawText(rect.adjusted(5, 0, -5, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        self._claim_label_box(rect)

    def _draw_trail(self, painter, trail, color):
        pts = list(trail)
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            p1, _ = self._project(pts[i - 1])
            p2, _ = self._project(pts[i])
            alpha = int(25 + 145 * i / len(pts))
            painter.setPen(QPen(QColor(*color, alpha), 1.3))
            painter.drawLine(p1, p2)

    def _draw_beam(self, painter, camera_world, target_world, aim_unit=None):
        # The beam originates at the optical terminal lens, NOT the spacecraft
        # body centre — terminal world position along the aim direction.
        origin_world = (
            self._terminal_world_pos(camera_world, aim_unit) if aim_unit is not None else camera_world
        )
        cp, _ = self._project(origin_world)
        tp, _ = self._project(target_world)
        self._last_beam_origin_screen = QPointF(cp)
        dx = tp.x() - cp.x()
        dy = tp.y() - cp.y()
        length = math.hypot(dx, dy)
        if length < 2.0:
            return
        # BEAM IMPACT overlay: when the beam-impact view is active, render
        # where the live beam lands relative to the receiving aperture.
        if self.beam_impact_enabled:
            self._draw_beam_impact(painter, tp, length)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux

        # Soft divergence cone behind the core — an illustrative visualization
        # of the widening optical path, not a calculated physics model.
        hw = min(9.0, 2.2 + length * 0.030)
        cone = QPolygonF([
            cp,
            QPointF(tp.x() + px * hw, tp.y() + py * hw),
            QPointF(tp.x() - px * hw, tp.y() - py * hw),
        ])
        cone_grad = QRadialGradient(cp, length)
        cone_grad.setColorAt(0.0, QColor(55, 255, 120, 40))
        cone_grad.setColorAt(1.0, QColor(55, 255, 120, 6))
        painter.setBrush(QBrush(cone_grad))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(cone)

        # Bright green visualization of the optical path.
        for width, alpha in ((15.0, 18), (9.0, 35), (4.0, 75)):
            painter.setPen(QPen(QColor(55, 255, 120, alpha), width))
            painter.drawLine(cp, tp)
        painter.setPen(QPen(QColor(165, 255, 188, 245), 1.3))
        painter.drawLine(cp, tp)

        # Animated photon flow along the core: packets visibly travel from the
        # terminal lens toward the remote terminal (direction of transmit).
        flow = QPen(QColor(214, 255, 226, 210), 2.2, Qt.CustomDashLine)
        flow.setDashPattern([5.0, 11.0])
        flow.setDashOffset(-self.frame_count * 1.7)
        painter.setPen(flow)
        painter.drawLine(cp, tp)

        # Visible beam-wander signature, driven by the actual turbulence sample.
        wx, wy = self.beam_wander_px
        if self.turbulence_enabled and (abs(wx) + abs(wy)) > 0.15:
            amp = min(20.0, 1.4 + 0.32 * math.hypot(wx, wy) + 0.10 * self.turbulence_strength)
            phase = self.frame_count * 0.46
            previous = None
            for i in range(31):
                t = i / 30.0
                wobble = math.sin(t * 10.0 + phase) * amp + math.sin(t * 23.0 - phase * 0.8) * amp * 0.28
                q = QPointF(cp.x() + dx * t + px * wobble, cp.y() + dy * t + py * wobble)
                if previous is not None:
                    painter.setPen(QPen(QColor(255, 215, 86, 175), 1.2, Qt.DashLine))
                    painter.drawLine(previous, q)
                previous = q

        # Direction ticks.
        painter.setPen(QPen(QColor(190, 255, 205, 175), 1.0))
        for t in (0.22, 0.50, 0.78):
            cx, cy = cp.x() + dx * t, cp.y() + dy * t
            tick = 4.5
            painter.drawLine(QPointF(cx - px * tick, cy - py * tick), QPointF(cx + px * tick, cy + py * tick))

    def _draw_beam_impact(self, painter, target_screen, beam_length):
        """Render the live beam impact at the receiving terminal.

        The impact point is placed at the aperture, displaced by the REAL
        pointing offset (the same px error the mission measures, mapped into
        the visualization frame). No invented physics: offset comes from
        set_beam_impact_data().
        """
        w, h = self.width(), self.height()
        aperture_r = max(26.0, min(w, h) * 0.085)
        ix, iy = target_screen.x(), target_screen.y()

        # Map the real px offset into screen displacement at the beacon.
        ox_px, oy_px = self.beam_impact_offset_px
        unit = max(beam_length / 26.0, 0.55)
        dx = ox_px * unit
        dy = oy_px * unit
        # Clamp so extreme errors stay visible on a ring instead of flying
        # off-canvas: the status text still reports the REAL value.
        max_disp = aperture_r * self.BEAM_IMPACT_CLAMP_FACTOR
        disp = math.hypot(dx, dy)
        if disp > max_disp > 0:
            k = max_disp / disp
            dx *= k
            dy *= k
        impact = QPointF(ix + dx, iy - dy)
        inside = self.beam_impact_state == "BEAM ALIGNED"

        # Receiving aperture ring around the beacon terminal.
        painter.setPen(QPen(QColor(120, 200, 255, 200), 2.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(target_screen, aperture_r, aperture_r)
        painter.setPen(QPen(QColor(120, 200, 255, 70), 1.0, Qt.DashLine))
        painter.drawEllipse(target_screen, aperture_r * 1.45, aperture_r * 1.45)

        # Offset vector from aperture center to the live impact point.
        if disp > 1.0:
            painter.setPen(QPen(QColor(255, 205, 84, 220), 1.6, Qt.DashLine))
            painter.drawLine(target_screen, impact)

        # Impact footprint: size grows with the real error magnitude.
        error = self.beam_impact_center_error_px
        footprint_r = max(3.5, min(aperture_r * 0.62, 4.0 + (error or 0.0) * 0.55))
        core = QColor(255, 205, 84, 210) if not inside else QColor(96, 255, 148, 220)
        glow = QColor(255, 205, 84, 60) if not inside else QColor(96, 255, 148, 70)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(impact, footprint_r * 2.0, footprint_r * 2.0)
        painter.setBrush(QBrush(core))
        painter.drawEllipse(impact, footprint_r, footprint_r)

        # Status banner: text from the REAL alignment state only.
        banner_w = 214
        bx, by = 18, 60
        painter.setBrush(QBrush(QColor(3, 10, 16, 225)))
        painter.setPen(QPen(QColor(80, 255, 125, 230) if inside else QColor(255, 170, 90, 230), 1.2))
        painter.drawRoundedRect(QRectF(bx, by, banner_w, 64), 8, 8)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(96, 255, 148) if inside else QColor(255, 176, 96))
        painter.drawText(bx + 12, by + 22, self.beam_impact_state)
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(237, 244, 248))
        if error is None:
            painter.drawText(bx + 12, by + 40, "CENTER ERROR: NO LOCK")
            painter.drawText(bx + 12, by + 55, "HANDOFF NOT READY")
        else:
            painter.drawText(bx + 12, by + 40, f"CENTER ERROR: {error:.1f} PX")
            ready_text = "HANDOFF READY" if self.beam_impact_link_ready else "HANDOFF NOT READY"
            painter.drawText(bx + 12, by + 55, ready_text)

    def _draw_pointer_axes(self, painter, camera_world, target_world):
        cp, _ = self._project(camera_world)
        tp, _ = self._project(target_world)
        dx, dy = tp.x() - cp.x(), tp.y() - cp.y()
        length = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        # Short axis showing where the communication terminal is aiming in the
        # independent view. The label is secondary: it yields to name plates.
        a1 = QPointF(cp.x() + ux * 46, cp.y() + uy * 46)
        painter.setPen(QPen(QColor(85, 188, 242, 200), 1.4, Qt.DashLine))
        painter.drawLine(cp, a1)
        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
        painter.setPen(QColor(107, 197, 241, 220))
        self._draw_secondary_label(painter, a1, "OPTICAL AXIS", QColor(107, 197, 241, 160))

    def _draw_motion_vector(self, painter, trail, current_pos, label):
        pts = list(trail) if trail else []
        if len(pts) < 8:
            return
        previous = pts[-8]
        p1, _ = self._project(previous)
        p2, _ = self._project(current_pos)
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        mag = math.hypot(dx, dy)
        if mag < 1.0:
            return
        ux, uy = dx / mag, dy / mag
        px, py = -uy, ux
        arrow_len = min(34.0, mag * 0.65)
        base = QPointF(p2.x() - ux * arrow_len, p2.y() - uy * arrow_len)
        left = QPointF(p2.x() - ux * 8 + px * 4.5, p2.y() - uy * 8 + py * 4.5)
        right = QPointF(p2.x() - ux * 8 - px * 4.5, p2.y() - uy * 8 - py * 4.5)
        painter.setPen(QPen(QColor(91, 255, 150, 205), 1.8))
        painter.drawLine(base, p2)
        painter.setBrush(QBrush(QColor(91, 255, 150, 210)))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([p2, left, right]))
        painter.setPen(QColor(116, 255, 160, 220))
        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
        self._draw_secondary_label(painter, base, label, QColor(116, 255, 160, 150))

    def _draw_disturbance_markers(self, painter, camera_world, target_world):
        cp, _ = self._project(camera_world)
        tp, _ = self._project(target_world)

        if self.platform_enabled and self.platform_strength > 0:
            # Platform displacement arrow from nominal origin.
            px, py = self.platform_offset_px
            scale = 0.75
            ref = QPointF(cp.x() - px * scale, cp.y() + py * scale)
            painter.setPen(QPen(QColor(255, 114, 136, 190), 1.4, Qt.DashLine))
            painter.drawLine(ref, cp)
            painter.setBrush(QBrush(QColor(255, 114, 136, 180)))
            painter.drawEllipse(ref, 3.0, 3.0)
            painter.setPen(QColor(255, 140, 156, 225))
            painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
            painter.drawText(int(cp.x() + 10), int(cp.y() + 18), "PLATFORM MOTION")

        if self.jitter_enabled and self.jitter_strength > 0:
            # Camera vibration envelope.
            for i in range(4):
                rr = 18 + i * 7 + 2.5 * math.sin(self.frame_count * 0.75 + i)
                painter.setPen(QPen(QColor(255, 207, 84, 120 - i * 16), 1.0))
                painter.drawEllipse(cp, rr, rr * 0.72)
            painter.setPen(QColor(255, 219, 100, 220))
            painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
            painter.drawText(int(cp.x() + 10), int(cp.y() + 34), "CAMERA JITTER")

        if self.turbulence_enabled and self.turbulence_strength > 0:
            wx, wy = self.beam_wander_px
            end = QPointF(tp.x() + wx * 0.9, tp.y() - wy * 0.9)
            painter.setPen(QPen(QColor(255, 205, 70, 185), 1.2, Qt.DashLine))
            painter.drawLine(tp, end)
            painter.setBrush(QBrush(QColor(255, 205, 70, 180)))
            painter.drawEllipse(end, 3.0, 3.0)
            painter.setPen(QColor(255, 218, 102, 220))
            painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
            painter.drawText(int(tp.x() + 9), int(tp.y() + 22), "BEAM WANDER")

    # ---------------------------------- HUD --------------------------------------
    def _draw_hud(self, painter):
        w, h = self.width(), self.height()
        state_color = QColor(70, 255, 125) if self.state in {"TRACKING", "RE-ACQUIRING", "PREDICTING"} else QColor(255, 204, 80)
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(state_color)
        painter.drawText(18, 26, f"● LIVE  |  FSOC 3D DIGITAL TWIN  |  PAT: {self.state}")
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(173, 194, 208))
        painter.drawText(18, 43, f"t = {self.sim_time:06.2f} s   •   frame {self.frame_count:05d}   •   motion: {self.motion_pattern.upper()}")

        painter.setPen(QColor(112, 218, 160))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.drawText(max(20, w - 310), 26, "LOCAL INERTIAL VIEW  •  USER CONTROLLED")
        painter.setPen(QColor(150, 172, 185))
        painter.setFont(QFont("Segoe UI", 6))
        painter.drawText(max(20, w - 310), 42, "mouse drag: orbit   •   wheel: zoom   •   F11: full screen")
        painter.setPen(QColor(120, 145, 160, 190))
        painter.setFont(QFont("Segoe UI", 6))
        painter.drawText(max(20, w - 310), 56, f"VIEW YAW {self.view_yaw:+.1f}°  |  VIEW PITCH {self.view_pitch:+.1f}°  |  ZOOM {self.zoom:.2f}×")
        painter.setPen(QColor(122, 214, 255))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.drawText(max(20, w - 310), 72, f"VIEW: {self._view_mode}")
        # PAT orientation is shown as READ-ONLY system data, never as the POV.
        pat_pan, pat_tilt = self.camera_angles
        painter.setPen(QColor(150, 172, 185, 200))
        painter.drawText(max(20, w - 310), 86, f"PAT (READ-ONLY):  PAN {pat_pan:+.2f}°   TILT {pat_tilt:+.2f}°")
        painter.setPen(QColor(120, 145, 160, 190))
        painter.drawText(max(20, w - 310), 100, "GREEN BEAM: ILLUSTRATIVE CONE • PHOTON FLOW TOWARD TARGET")
        active = []
        if self.platform_enabled and self.platform_strength > 0:
            active.append("PLATFORM")
        if self.jitter_enabled and self.jitter_strength > 0:
            active.append("JITTER")
        if self.turbulence_enabled and self.turbulence_strength > 0:
            active.append("TURBULENCE")
        if self.noise_name != "None":
            active.append("NOISE")
        painter.setPen(QColor(246, 205, 92) if active else QColor(120, 145, 160, 190))
        painter.drawText(max(20, w - 310), 114, "DISTURBANCES: " + (" + ".join(active) if active else "NONE ACTIVE"))
        # Environment readout (visual-only preset; not a physics claim).
        painter.setPen(QColor(186, 160, 255))
        painter.drawText(max(20, w - 310), 128, f"ENV: {self.environment_name}")

        box_h = 112
        box = QRectF(14, h - box_h - 14, w - 28, box_h)
        painter.setBrush(QBrush(QColor(3, 10, 16, 236)))
        painter.setPen(QPen(QColor(44, 67, 80, 190), 1))
        painter.drawRoundedRect(box, 9, 9)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(237, 244, 248))
        painter.drawText(27, h - box_h + 4, f"TARGET  {self.designated_id}   •   {self.motion_pattern.upper()}   •   RANGE {self.range_m:.0f} m")
        painter.setPen(QColor(116, 255, 155))
        painter.drawText(27, h - box_h + 23, "BRIGHT BEACON  •  DESIGNATED COMMUNICATION TARGET")

        disturbance = []
        if self.platform_enabled and self.platform_strength > 0:
            disturbance.append(f"PLATFORM ±{self.platform_strength:.0f}px")
        if self.jitter_enabled and self.jitter_strength > 0:
            disturbance.append(f"JITTER ±{self.jitter_strength:.0f}px")
        if self.turbulence_enabled and self.turbulence_strength > 0:
            disturbance.append(f"BEAM-WANDER MODEL ±{self.turbulence_strength:.0f}px")
        if self.noise_name != "None":
            disturbance.append(f"SENSOR NOISE {self.noise_name}")
        dist_text = "  •  ".join(disturbance) if disturbance else "NOMINAL — NO ACTIVE DISTURBANCES"
        painter.setPen(QColor(246, 205, 92) if disturbance else QColor(144, 170, 183))
        painter.drawText(27, h - box_h + 42, f"DISTURBANCE STATE  •  {dist_text}")
        painter.setPen(QColor(135, 158, 171))
        painter.setFont(QFont("Segoe UI", 6))
        painter.drawText(27, h - box_h + 60, "ATMOSPHERIC RAIN / FOG / HAZE ARE NOT RENDERED IN THE TWO-SATELLITE SPACE SCENE")
        painter.drawText(27, h - box_h + 77, "GREEN BEAM IS A VISUALIZATION OF THE OPTICAL LINK PATH  •  VIEW IS NOT THE CAMERA POV")

        ready_text = "HANDOFF READY" if self.link_ready else "ALIGNMENT IN PROGRESS"
        ready_color = QColor(80, 255, 125) if self.link_ready else QColor(255, 198, 80)
        painter.setPen(ready_color)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(w - 190, h - 44, ready_text)
        painter.setPen(QColor(116, 143, 156))
        painter.setFont(QFont("Segoe UI", 6))
        painter.drawText(w - 190, h - 27, "coarse optical handoff")

    # ---------------------------------- paint ------------------------------------
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Fresh label-collision budget every frame (draw order = priority:
        # name plates claim first, secondary annotations yield to them).
        self._label_boxes = []
        self._draw_background(painter)
        self._draw_environment_layers(painter)
        self._draw_starfield(painter)
        # Slow celestial drift: bodies creep along their orbit arc over minutes
        # (draw-only; phases seeded per environment build).
        frame = self.frame_count
        for body in self._celestial:
            bx, by, bz = body["pos"]
            phase = body.get("drift_phase", 0.0)
            drift = frame * 0.0012
            body["pos"] = (
                bx + math.cos(phase) * drift * 0.06,
                by + math.sin(phase) * drift * 0.03,
                bz,
            )
            self._draw_celestial_body(painter, body)

        # Draw current synchronized camera and target in a fixed local inertial frame.
        # Scene offsets/separation apply ONLY to where objects are drawn here.
        camera_world = self._vis_camera_pos()
        raw_camera = self.camera_position
        entries = []
        for target in self.world_targets:
            tid = str(target.get("id", ""))
            raw_pos = self._world_position_from_raw(target)
            pos = self._vis_target_pos(raw_pos)
            entries.append((tid, bool(target.get("visible", True)), pos, tid == self.designated_id, float(target.get("range_m", self.range_m)), raw_pos))
        entries.sort(key=lambda item: sum(v * v for v in item[2]), reverse=True)

        # Trails / motion are deliberately prominent.
        for tid, visible, pos, designated, _r, _raw in entries:
            trail = self._target_trails.get(tid)
            if trail:
                self._draw_trail(painter, trail, (72, 255, 130) if designated else (128, 152, 168))

        if any(abs(v) > 1e-9 for v in self.scene_camera_offset):
            ox, oy, oz = self.scene_camera_offset
            camera_trail_pts = deque(
                ((x + ox, y + oy, z + oz) for x, y, z in self._camera_trail),
                maxlen=max(len(self._camera_trail), 1),
            )
        else:
            camera_trail_pts = self._camera_trail
        self._draw_trail(painter, camera_trail_pts, (90, 180, 238))

        designated_entry = None
        for tid, visible, pos, designated, _r, raw_pos in entries:
            self._draw_satellite(
                painter,
                pos,
                name=f"{tid}  •  {'BRIGHT BEACON' if designated else 'DECOY / OTHER TARGET'}",
                designated=designated,
                camera_unit=False,
            )
            if designated and visible:
                designated_entry = (pos, raw_pos)

        # Aim unit vector of the optical terminal in the visualization frame:
        # toward the designated beacon when a live beam exists, otherwise along
        # the PAT camera boresight derived from the actual pan/tilt (read-only).
        if designated_entry is not None:
            aim = designated_entry[0]
        else:
            aim = self._boresight_direction(raw_camera)
        aim_unit = self._aim_unit(raw_camera, aim)

        self._draw_satellite(
            painter,
            camera_world,
            name="TRACKING SATELLITE  •  CAMERA / OPTICAL TERMINAL",
            designated=False,
            camera_unit=True,
            aim_unit=aim_unit,
        )

        if designated_entry is not None:
            designated_pos, designated_raw = designated_entry
            # Engineering line-of-sight between TRUE synchronized positions.
            cp_screen, _ = self._project(raw_camera)
            tp_screen, _ = self._project(designated_raw)
            painter.setPen(QPen(QColor(84, 177, 218, 90), 0.8, Qt.DotLine))
            painter.drawLine(cp_screen, tp_screen)

            self._draw_scene_axes(painter)
            self._draw_beam(painter, camera_world, designated_pos, aim_unit)
            self._draw_pointer_axes(painter, camera_world, designated_pos)
            # Secondary annotation: the terminal label draws late (after name
            # plates claimed space) at the real lens position with a leader
            # line — placed by priority instead of hard-coded offsets.
            if self.terminal_anchor_screen is not None:
                painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
                self._draw_secondary_label(
                    painter, QPointF(self.terminal_anchor_screen), "OPTICAL TERMINAL",
                    QColor(120, 226, 250, 160),
                )
            self._draw_motion_vector(
                painter,
                self._target_trails.get(self.designated_id),
                designated_pos,
                f"VELOCITY VECTOR • {self.motion_pattern.upper()}",
            )
            self._draw_disturbance_markers(painter, camera_world, designated_pos)

            # Big, readable motion vector at the beacon.
            tp, _ = self._project(designated_pos)  # defined for prediction marker regardless of trail length
            trail = self._target_trails.get(self.designated_id)
            if trail and len(trail) >= 6:
                old = list(trail)[-6]
                op, _ = self._project(old)
                painter.setPen(QPen(QColor(92, 255, 145, 170), 2.0, Qt.DashLine))
                painter.drawLine(op, tp)
                painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                painter.setPen(QColor(102, 255, 150, 225))
                painter.drawText(int(tp.x() + 10), int(tp.y() - 12), f"TARGET MOTION • {self.motion_pattern.upper()}")

            # Disturbance motion rings around camera platform.
            cp, _ = self._project(camera_world)
            if self.platform_enabled and self.platform_strength > 0:
                r = 15 + min(12, self.platform_strength * 0.45)
                painter.setPen(QPen(QColor(255, 112, 135, 160), 1.1))
                painter.drawEllipse(cp, r, r * 0.72)
            if self.jitter_enabled and self.jitter_strength > 0:
                for i in range(4):
                    r = 17 + i * 7 + 2 * math.sin(self.frame_count * 0.7 + i)
                    painter.setPen(QPen(QColor(255, 207, 80, 115 - i * 15), 1.0))
                    painter.drawEllipse(cp, r, r)

            # Kalman prediction: projected from the TRUE target position so the
            # prediction marker stays on the live-geometry ray.
            pred = self._projected_prediction(self.world_targets)
            if pred is not None:
                pp, _ = self._project(pred)
                painter.setPen(QPen(QColor(255, 220, 84, 195), 1.3, Qt.DashLine))
                painter.drawLine(tp, pp)
                painter.setBrush(QBrush(QColor(255, 220, 84, 225)))
                painter.setPen(QPen(QColor(255, 235, 145), 1.2))
                painter.drawEllipse(pp, 6.0, 6.0)
                painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
                painter.setPen(QColor(255, 223, 96, 235))
                painter.drawText(int(pp.x() + 9), int(pp.y() - 5), "KALMAN PREDICTION")

        self._draw_hud(painter)
        painter.end()

    # -------------------------------- interaction --------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # A user grab always aborts a cinematic focus flight — the
            # independent user POV has priority over the animation.
            if self._focus_anim is not None:
                self._stop_focus_animation()
                self._view_mode = "FULL SCENE"
            pos = event.position().toPoint()
            # Direct 3D manipulation: pressing ON a satellite grabs the object
            # itself (touch input synthesizes identical mouse events); any
            # other press orbits the viewpoint exactly as before.
            hit = self._hit_test_targets(pos)
            if hit is not None:
                target, sp = hit
                self._object_drag = {"id": str(target.get("id")), "screen": pos}
                self.setCursor(Qt.ClosedHandCursor)
                self.setFocus()
                event.accept()
                return
            self._dragging = True
            self._last_mouse = pos
            self.setFocus()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._object_drag is not None:
            self._emit_drag_delta(pos)
            self.update()
            event.accept()
            return
        if self._dragging:
            dx = pos.x() - self._last_mouse.x()
            dy = pos.y() - self._last_mouse.y()
            self.view_yaw += dx * 0.40
            self.view_pitch += dy * 0.28
            self.view_pitch = max(-70.0, min(70.0, self.view_pitch))
            self._last_mouse = pos
            self.update()
            event.accept()
            return
        # Discoverability: open-hand cursor while hovering a grabbable object.
        self.setCursor(Qt.OpenHandCursor if self._hit_test_targets(pos) else Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._object_drag is not None:
                self._object_drag = None
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self.zoom *= 1.0 + 0.12 * steps
        self.zoom = max(0.55, min(3.8, self.zoom))
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        window = self.window()
        if hasattr(window, "toggle_fullscreen"):
            window.toggle_fullscreen()
        else:
            window.showNormal() if window.isFullScreen() else window.showFullScreen()
        event.accept()


class DigitalTwinWindow(QMainWindow):
    """Standalone independent-view 3D scene synchronized to the main GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ASTRALINK — 3D Digital Twin | Independent Inertial View")
        self.resize(1320, 860)
        self.setMinimumSize(980, 700)

        # The window is a thin shell around the reusable TwinPane so the SAME
        # twin surface (scene + view controls + mission strip) can be embedded
        # directly inside the main application window.
        self.pane = TwinPane(show_fullscreen=True)
        host = parent
        while host is not None and not hasattr(host, "_apply_twin_mission"):
            host = host.parent()
        self.pane.bind_host(host)
        self.setCentralWidget(self.pane)

        # Backward-compatible aliases used by the regression harness. The
        # visualization-only scene sliders no longer exist in the hero UI.
        self.twin = self.pane.twin
        self.full_button = self.pane.fullscreen_button
        self.cam_x = self.cam_y = self.cam_z = None
        self.tgt_x = self.tgt_y = self.tgt_z = None
        self.sep_slider = None
        self.env_selector = self.pane.env_selector
        self._last_normal_geometry = None

    def set_state(self, **kwargs):
        self.pane.set_state(**kwargs)

    def set_beam_impact_data(self, center_error_px, offset_px=(0.0, 0.0), link_ready=False):
        self.pane.set_beam_impact_data(center_error_px, offset_px, link_ready)

    def toggle_beam_impact(self):
        self.pane.toggle_beam_impact()

    def reset_view(self):
        self.pane.reset_view()

    def reset_scene(self):
        self.pane.reset_scene()

    def focus_camera_terminal(self):
        self.pane.focus_camera_terminal()

    def focus_beacon_terminal(self):
        self.pane.focus_beacon_terminal()

    def focus_full_scene(self):
        self.pane.focus_full_scene()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            if self._last_normal_geometry is not None:
                self.setGeometry(self._last_normal_geometry)
            self.full_button.setText("FULL SCREEN  F11")
        else:
            self._last_normal_geometry = self.geometry()
            self.showFullScreen()
            self.full_button.setText("EXIT FULL SCREEN")

    def _apply_separation(self, scale):
        self.pane._apply_separation(scale)

    def _on_reset_scene(self):
        self.pane._on_reset_scene()

    def _sync_scene_sliders(self):
        self.pane._sync_scene_sliders()

    def keyPressEvent(self, event):
        # Standalone twin window: F11 toggles ITS OWN window full screen;
        # ESC leaves that full screen. The embedded twin's true app-level
        # full screen is handled by the host main window (Stage 9).
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)


class TwinPane(QWidget):
    """Reusable Digital-Twin surface: the synchronized 3D scene plus the view
    control bar (FULL SCENE / terminal focus modes / scene positioning) and a
    compact mission-control strip (target, motion, disturbances, pointing).

    The pane can be embedded inside the main window (mission-control page) or
    hosted standalone in :class:`DigitalTwinWindow` — both surfaces share the
    SAME live mission state through ``bind_host`` / ``set_state``.
    """

    modeChanged = Signal(str)

    def __init__(self, parent=None, show_fullscreen=False):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ---- HERO LAYOUT: the 3D scene owns 75-85% of the window ----------
        # TOP: one compact status row (title + live status + env + F11).
        # CENTER: the huge synchronized 3D viewport (stretch factor 1).
        # BOTTOM: one compact toolbar row + one essential-info strip.
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        title = QLabel("3D DIGITAL TWIN")
        title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        title.setStyleSheet("font-size: 12px; font-weight: 800; color: #dbe8f0; letter-spacing: 1px;")
        # Title absorbs the row's spare width so every other item (including
        # CLOSE) hugs its natural compact size instead of inflating equally.
        top_row.addWidget(title, 1)
        self.status_label = QLabel("● LIVE  •  SYNCHRONIZED")
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.status_label.setStyleSheet("font-size: 9px; font-weight: 700; color: #76d5a1;")
        top_row.addWidget(self.status_label)
        env_label = QLabel("ENVIRONMENT")
        env_label.setStyleSheet("font-size: 9px; font-weight: 700; color: #b49fff;")
        top_row.addWidget(env_label)
        self.env_selector = QComboBox()
        self.env_selector.addItems(list(DigitalTwinWidget.ENVIRONMENT_PRESETS))
        self.env_selector.setToolTip(
            "Visual environment preset (draw-only). Does not affect tracking or benchmark results."
        )
        self.env_selector.setMinimumWidth(140)
        top_row.addWidget(self.env_selector)
        self.close_button = QPushButton("✕  CLOSE")
        self.close_button.setObjectName("close_button")
        self.close_button.setToolTip("Close the Digital Twin window")
        self.close_button.clicked.connect(self._on_close_requested)
        top_row.addWidget(self.close_button)
        self._title_row_host = QWidget()
        self._title_row_host.setLayout(top_row)
        layout.addWidget(self._title_row_host)

        # ---- synchronized 3D scene (THE HERO — takes all remaining space) --
        self.twin = DigitalTwinWidget(self)
        layout.addWidget(self.twin, 1)

        # ---- COMPACT BOTTOM TOOLBAR (wrapping row) -------------------------
        # FlowLayout keeps this toolbar from pinning a ~1000 px minimum on the
        # whole window: the buttons wrap onto a second row on a narrow screen
        # instead of being cut off or forcing a horizontal scrollbar. The
        # buttons keep their exact size and order — only the wrapping is new.
        row1 = FlowLayout(h_spacing=6, v_spacing=4)

        def _bar_button(text, width):
            button = QPushButton(text)
            button.setMinimumWidth(width)
            return button

        self.fullscene_button = _bar_button("FULL SCENE", 92)
        self.focus_cam_button = _bar_button("FOCUS CAMERA TERMINAL", 168)
        self.focus_tgt_button = _bar_button("FOCUS BEACON TERMINAL", 168)
        self.return_button = _bar_button("RETURN TO FULL SCENE", 158)
        self.return_button.setEnabled(False)
        self.fullscreen_button = _bar_button("FULL SCREEN  F11", 140)
        self.beam_impact_button = _bar_button("⚡ BEAM IMPACT", 118)
        self.beam_impact_button.setToolTip(
            "Fly to the receiving aperture and show where the LIVE beam lands "
            "relative to it, with the real center error and alignment state."
        )
        reset_view_button = _bar_button("RESET VIEW", 105)
        for button in (self.fullscene_button, self.focus_cam_button, self.focus_tgt_button,
                       self.return_button, self.beam_impact_button, reset_view_button,
                       self.fullscreen_button):
            row1.addWidget(button)
        self._toolbar_row_host = QWidget()
        self._toolbar_row_host.setLayout(row1)
        layout.addWidget(self._toolbar_row_host)

        # ---- ESSENTIAL-INFO STRIP (compact; not a second dashboard) -------
        self.mission_strip = QFrame()
        self.mission_strip.setObjectName("twin_mission_strip")
        # Wrapping strip: the four info groups sit side by side whenever there
        # is room and wrap onto further rows when the pane narrows. This keeps
        # the strip from pinning a ~1090 px minimum on the whole window.
        strip = FlowLayout(self.mission_strip, h_spacing=8, v_spacing=4)
        strip.setContentsMargins(8, 4, 8, 4)
        target_group = self._strip_group("TARGET", [
            ("ID", "twin_designated"),
        ])
        strip.addWidget(target_group)
        # SEPARATION: compact [ - ] value [ + ] control INSIDE the TARGET
        # group (4 rows total — the same height MOTION already has, so the
        # strip and the hero canvas keep their exact previous geometry).
        # It mirrors the REAL beacon-range path on the main dashboard (same
        # slider, same handler, same simulator state) — no twin-only
        # distance variable.
        sep_row = QHBoxLayout()
        sep_row.setSpacing(4)
        self.sep_minus = QPushButton("−")
        self.sep_minus.setObjectName("twin_step_button")
        self.sep_minus.setFixedSize(20, 20)
        self.sep_minus.setToolTip("Decrease satellite separation (beacon range)")
        self.sep_value = QLabel("—")
        self.sep_value.setObjectName("twin_value")
        self.sep_value.setAlignment(Qt.AlignCenter)
        self.sep_value.setMinimumWidth(64)
        self.sep_plus = QPushButton("+")
        self.sep_plus.setObjectName("twin_step_button")
        self.sep_plus.setFixedSize(20, 20)
        self.sep_plus.setToolTip("Increase satellite separation (beacon range)")
        self.sep_minus.clicked.connect(lambda: self._on_separation_step(-1))
        self.sep_plus.clicked.connect(lambda: self._on_separation_step(+1))
        sep_title = QLabel("SEPARATION")
        sep_title.setObjectName("twin_group_title")
        tgrid = target_group.layout()
        tgrid.addWidget(sep_title, 2, 0, 1, 2)
        tgrid.addWidget(self.sep_minus, 3, 0)
        tgrid.addWidget(self.sep_value, 3, 1)
        tgrid.addWidget(self.sep_plus, 3, 2)
        # MOTION is read-only in the twin (no duplicate editing of the main
        # dashboard's control); _strip_group renders values as QLabels.
        strip.addWidget(self._strip_group("MOTION", [
            ("PATTERN", "twin_motion"),
            ("PAN", "twin_pan_label"),
            ("TILT", "twin_tilt_label"),
        ]))
        strip.addWidget(self._strip_group("POINTING", [
            ("CENTER ERR", "twin_center"), ("TRACK ERR", "twin_error"),
        ]))
        strip.addWidget(self._strip_group("LINK", [
            ("STATUS", "twin_handoff"), ("STATE", "twin_state"),
        ]))
        layout.addWidget(self.mission_strip)

        self.host = None
        self._syncing_strip = False
        self.fullscreen_button.setVisible(bool(show_fullscreen))
        self.fullscreen_button.clicked.connect(self._on_fullscreen_clicked)
        self.fullscene_button.clicked.connect(self.focus_full_scene)
        self.focus_cam_button.clicked.connect(self.focus_camera_terminal)
        self.focus_tgt_button.clicked.connect(self.focus_beacon_terminal)
        self.return_button.clicked.connect(self.focus_full_scene)
        self.beam_impact_button.clicked.connect(self.toggle_beam_impact)

        # Visualization-only scene controls (CAM/TGT OFFSET, SEPARATION) were
        # removed from the hero layout by design — they remain available as
        # engine methods and the harness verifies they never touch the real
        # controller. Nothing in the pane references the removed sliders.
        self.env_selector.currentTextChanged.connect(self._on_environment_changed)

        # Twin focus buttons drive both the animation and this bar's labels.
        self.twin.focus_requested.connect(self._on_focus_requested)
        self._sync_scene_sliders()
        self.setStyleSheet("""
            QWidget { background: transparent; }
            #twin_mission_strip {
                background: #04101a; border: 1px solid #1d3a4d; border-radius: 8px;
            }
            #twin_group { background: #081724; border: 1px solid #1b2f3f; border-radius: 6px; }
            #twin_group_title { font-size: 7px; font-weight: 800; color: #6f93a8; letter-spacing: 1px; }
            #twin_value { font-size: 9px; font-weight: 700; color: #d7e6ef; }
            #twin_check { font-size: 8px; color: #a9bfcc; }
            #twin_combo {
                background: #0c1c29; border: 1px solid #2c485c; border-radius: 5px;
                padding: 2px 4px; color: #dcebf4; font-size: 8px; min-height: 18px;
            }
            QComboBox QAbstractItemView { background: #0c1c29; color: #dcebf4; }
            QPushButton {
                background: #0b1822; border: 1px solid #284556; border-radius: 6px;
                padding: 7px 10px; color: #dbe8f0; font-weight: 700; font-size: 9px;
            }
            QPushButton:hover { background: #123043; }
            QPushButton:disabled { color: #5a7181; border-color: #22323e; }
            #twin_focus_active { background: #123a52; border-color: #3f89b5; color: #bfe6ff; }
            QSlider::groove:horizontal { height: 3px; background: #1d3547; border-radius: 1px; }
            QSlider::handle:horizontal { width: 9px; margin: -3px 0; border-radius: 4px; background: #58a6c9; }
            QLabel { font-size: 9px; color: #93aebb; }
        """)

    # ------------------------- strip construction -------------------------
    def _strip_group(self, title, rows):
        group = QFrame()
        group.setObjectName("twin_group")
        grid = QGridLayout(group)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(1)
        header = QLabel(title)
        header.setObjectName("twin_group_title")
        grid.addWidget(header, 0, 0, 1, max(2, len(rows)))
        for index, (label, attr) in enumerate(rows):
            name = QLabel(label)
            name.setStyleSheet("font-size: 7px; color: #7d99a8;")
            if attr.endswith("_check"):
                widget = QCheckBox()
                widget.setObjectName("twin_check")
                widget.setToolTip(f"Toggle {label.lower()} from inside the Digital Twin")
                widget.stateChanged.connect(self._strip_control_changed)
            elif attr.endswith("_combo") or attr in (
                "twin_designated",
            ):
                widget = QComboBox()
                widget.setObjectName("twin_combo")
                widget.setMinimumWidth(74)
                # The designated-target list is host-driven (live target IDs).
                widget.currentTextChanged.connect(self._strip_control_changed)
            else:
                widget = QLabel("—")
                widget.setObjectName("twin_value")
                # Live telemetry flows through these labels every tick. An
                # unbounded width let every value change resize the strip and
                # shift sibling groups — the observed Twin jitter. Ignored
                # policy plus a per-metric reserved minimum (measured from
                # the widest realistic value, not a blanket size) gives each
                # value a stable column: text changes never move the strip
                # again, and the strip stays narrow enough for 1920 px.
                _longest = {
                    "twin_pan": "-888.88°", "twin_tilt": "-888.88°",
                    "twin_center": "888.88 px", "twin_error": "888.88 px",
                    "twin_state": "RE-ACQUIRING", "twin_handoff": "NOT READY",
                }
                _fm = widget.fontMetrics()
                widget.setMinimumWidth(int(_fm.horizontalAdvance(_longest.get(attr, "888.88")) * 1.05) + 8)
                widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
                widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            setattr(self, attr, widget)
            grid.addWidget(name, 1, index)
            grid.addWidget(widget, 2, index)
        return group

    # ---------------------------- host binding ----------------------------
    def bind_host(self, host):
        """Bind the mission strip to the live mission (one source of truth).

        The host must expose ``_apply_twin_mission(section, key, value)`` to
        push strip edits into the real simulator/GUI and ``_twin_strip_state()``
        for the strip refresh. No binding means the strip stays display-only.
        """
        self.host = host
        self._refresh_strip_controls()

    # --------------------------- separation control ------------------------
    def _on_separation_step(self, direction):
        """SEPARATION [−]/[+]: one clean step through the REAL range path.

        The step lands on the main dashboard's Beacon Range slider (the same
        control, the same signal handler, the same simulator state), so
        radar, camera, strip, canvas and reports all move together — there
        is no twin-only distance variable.
        """
        host = self.host
        if host is None or not hasattr(host, "beacon_range_m"):
            return
        current = float(host.beacon_range_m)
        step = max(50.0, round(current * 0.1 / 50.0) * 50.0)
        new_value = max(100.0, min(10000.0, current + direction * step))
        slider_value = int(round(new_value / 100.0))
        slider_value = max(host.range_slider.minimum(),
                           min(host.range_slider.maximum(), slider_value))
        host.range_slider.setValue(slider_value)  # normal signal path
        self._sync_separation_display(slider_value * 100.0)

    def _sync_separation_display(self, range_m=None):
        host = self.host
        if host is None or not hasattr(host, "beacon_range_m"):
            return
        value = float(host.beacon_range_m if range_m is None else range_m)
        self.sep_value.setText(f"{value:.0f} m")

    def _strip_control_changed(self, _value=None):
        if self._syncing_strip or self.host is None:
            return
        self._syncing_strip = True
        try:
            # "motion" intentionally excluded: read-only in the twin strip.
            for key in ("count", "designated", "noise", "turbulence",
                        "jitter", "platform", "atmosphere"):
                self.host._apply_twin_mission("strip", key, self._strip_value(key))
        finally:
            self._syncing_strip = False

    def _strip_value(self, key):
        mapping = {
            "count": lambda: None,
            "designated": lambda: self.twin_designated.currentText(),
            # MOTION is read-only in the twin (display only); editing stays on
            # the main dashboard, so the strip never emits it.
            "motion": lambda: None,
        }
        getter = mapping.get(key)
        return getter() if getter else None

    def _refresh_strip_controls(self):
        if self.host is None:
            return
        state = self.host._twin_strip_state()
        self._syncing_strip = True
        try:
            options = state.get("designated_options")
            if options:
                self.twin_designated.blockSignals(True)
                self.twin_designated.clear()
                self.twin_designated.addItems([str(option) for option in options])
                self.twin_designated.blockSignals(False)
            combo_box = self.twin_designated
            value = state.get("designated")
            combo_box.blockSignals(True)
            if value is not None:
                index = combo_box.findText(str(value))
                if index >= 0:
                    combo_box.setCurrentIndex(index)
            combo_box.blockSignals(False)
            # MOTION is read-only in the twin strip: display the live motion
            # pattern the host reports instead of editing a duplicate selector.
            if state.get("motion"):
                self.twin_motion.setText(str(state.get("motion")))
        finally:
            self._syncing_strip = False

    # --------------------------- strip refresh ----------------------------
    def update_strip(self, **values):
        """Refresh read-only strip status (values pushed by the host each tick)."""
        if "separation" in values:
            self.sep_value.setText(str(values["separation"]))
        simple = {
            "center": "twin_center", "error": "twin_error",
            "state": "twin_state", "handoff": "twin_handoff",
            "designated": "twin_designated", "motion": "twin_motion",
            "pan": "twin_pan_label", "tilt": "twin_tilt_label",
        }
        for key, attr in simple.items():
            if key in values:
                widget = getattr(self, attr)
                if isinstance(widget, QComboBox):
                    index = widget.findText(str(values[key]))
                    if index >= 0 and widget.currentText() != str(values[key]):
                        widget.blockSignals(True)
                        widget.setCurrentIndex(index)
                        widget.blockSignals(False)
                else:
                    widget.setText(str(values[key]))

    # ---------------------------- focus plumbing --------------------------
    def _on_focus_requested(self, mode):
        self._view_label_sync()
        self.modeChanged.emit(mode)

    def toggle_beam_impact(self):
        """BEAM IMPACT view: fly to the receiver and render the live impact."""
        self.twin.toggle_beam_impact()
        self._view_label_sync()
        self.modeChanged.emit("BEAM IMPACT" if self.twin.beam_impact_enabled else "FULL SCENE")

    def _view_label_sync(self):
        mode = self.twin.view_mode
        self.return_button.setEnabled(mode != "FULL SCENE")
        beam_active = getattr(self.twin, "beam_impact_enabled", False)
        for button, active in (
            (self.fullscene_button, mode == "FULL SCENE" and not beam_active),
            (self.focus_cam_button, mode == "CAMERA TERMINAL"),
            (self.focus_tgt_button, mode == "BEACON TERMINAL"),
            (self.beam_impact_button, beam_active),
        ):
            button.setProperty("twin_focus_active", "true" if active else "false")
            button.style().unpolish(button)
            button.style().polish(button)

    def focus_camera_terminal(self):
        self.twin.focus_camera_terminal()
        self._view_label_sync()
        self.modeChanged.emit("CAMERA TERMINAL")

    def focus_beacon_terminal(self):
        self.twin.focus_beacon_terminal()
        self._view_label_sync()
        self.modeChanged.emit("BEACON TERMINAL")

    def focus_full_scene(self):
        self.twin.focus_full_scene()
        self._view_label_sync()
        self.modeChanged.emit("FULL SCENE")

    def reset_view(self):
        self.twin.reset_view()
        self._view_label_sync()
        self.modeChanged.emit("FULL SCENE")

    def reset_scene(self):
        self.twin.reset_scene()

    def set_state(self, **kwargs):
        self.twin.set_state(**kwargs)

    def set_beam_impact_data(self, center_error_px, offset_px=(0.0, 0.0), link_ready=False):
        """Forward the live pointing result to the 3D scene (display only)."""
        self.twin.set_beam_impact_data(center_error_px, offset_px, link_ready)

    def _apply_separation(self, scale):
        self.twin.set_scene_separation(scale)

    def _on_reset_scene(self):
        self.twin.reset_scene()

    def _on_environment_changed(self, name):
        """Apply the selected visual environment preset (draw-only)."""
        self.twin.set_environment(name)

    def sync_environment(self):
        """Reflect the twin's current environment in the selector without loops."""
        self.env_selector.blockSignals(True)
        self.env_selector.setCurrentText(self.twin.environment_name)
        self.env_selector.blockSignals(False)

    def _sync_scene_sliders(self):
        """No-op: the scene sliders were removed from the hero UI by design.
        Kept as a compatibility shim for the harness/window delegates."""

    def set_fullscreen_chrome(self, enabled):
        """Slim the pane to the full-screen overlay set.

        Hides the title row and the info strip so the synchronized 3D canvas
        owns the whole surface; keeps the view-action overlay (FULL SCENE /
        FOCUS… / BEAM IMPACT / RESET VIEW / EXIT) visible. The strip keeps
        syncing in the background — hiding it never detaches the twin from
        the mission.
        """
        enabled = bool(enabled)
        for widget in (self._title_row_host, self.mission_strip):
            if widget is not None:
                widget.setVisible(not enabled)
        self.fullscreen_button.setText("EXIT FULL SCREEN" if enabled else "FULL SCREEN  F11")

    def _on_close_requested(self):
        """CLOSE button: hide the standalone twin window; in the embedded pane
        return the main window to the radar/camera dashboard."""
        window = self.window()
        if isinstance(window, DigitalTwinWindow):
            window.close()
            return
        # Embedded: bind_host() stores the main window itself.
        main = getattr(self, "host", None)
        if main is not None and hasattr(main, "_switch_live_view"):
            main._switch_live_view(0)

    def _on_fullscreen_clicked(self):
        window = self.window()
        if hasattr(window, "toggle_fullscreen"):
            # Standalone twin window keeps its own window-level fullscreen.
            window.toggle_fullscreen()
        else:
            # Embedded in the main window: TRUE app fullscreen (Stage 9).
            host = getattr(self, "host", None)
            main = getattr(host, "window", None) if host is not None else None
            if main is not None and hasattr(main, "enter_twin_fullscreen"):
                if getattr(main, "_twin_fullscreen", False):
                    main.exit_twin_fullscreen()
                else:
                    main.enter_twin_fullscreen()
