"""Virtual FSOC camera scene with multiple moving optical targets.

The simulator supports 1–5 independently moving optical targets. The user
selects how many targets are active and designates exactly one target for PAT
tracking; all other active targets remain moving decoys. No obstacle/debris
objects are part of this simulator.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import config


TARGET_SIGNATURES = {
    "TGT-01": {"name": "GREEN-SQUARE", "color_bgr": (70, 255, 70), "shape": "square"},
    "TGT-02": {"name": "AMBER-DIAMOND", "color_bgr": (40, 180, 255), "shape": "diamond"},
    "TGT-03": {"name": "MAGENTA-CIRCLE", "color_bgr": (255, 80, 220), "shape": "circle"},
    "TGT-04": {"name": "CYAN-TRIANGLE", "color_bgr": (255, 230, 70), "shape": "triangle"},
    "TGT-05": {"name": "RED-CIRCLE", "color_bgr": (70, 70, 255), "shape": "circle"},
}
INITIAL_POSITIONS = [
    (60.0, 240.0),
    (210.0, 120.0),
    (420.0, 340.0),
    (530.0, 130.0),
    (330.0, 400.0),
]
RANGE_SCALES = [1.00, 0.82, 1.24, 0.68, 1.45]
SPEED_SCALES = [1.00, 0.82, 1.12, 0.68, 0.95]


class Simulator:
    def __init__(self):
        self.center_x = config.FRAME_WIDTH / 2.0
        self.center_y = config.FRAME_HEIGHT / 2.0
        self.refresh_camera_geometry()

        self.motion_pattern = "Straight Line"
        self.frame_count = 0
        self.speed_x = 4.0
        self.random_speed = 4.0
        self.speed_factor = 1.0
        self.beacon_range_m = 1000.0
        self.beacon_size_px = float(config.BEACON_SIZE)

        self.camera_pan = 0.0
        self.camera_tilt = 0.0
        self.beacon_visible = True
        self.loss_start = None
        self.loss_end = None

        self.noise_type = "None"
        self.noise_level = 10
        self.turbulence_enabled = False
        self.turbulence_strength_px = 0.0
        self.camera_jitter_enabled = False
        self.camera_jitter_px = 0.0
        self.platform_motion_enabled = False
        self.platform_motion_px = 0.0
        self.platform_motion_pattern = "Linear"
        self.atmosphere = "Clear"
        self.atmosphere_level = 0
        self.disturbance_preset = "Clear"

        self.last_camera_jitter = (0.0, 0.0)
        self.camera_jitter_offset = np.zeros(2, dtype=np.float32)
        self.last_platform_offset = (0.0, 0.0)
        self.last_turbulence_dx = None
        self.last_turbulence_dy = None
        self.turbulence_field_x = None
        self.turbulence_field_y = None
        self.last_platform_random = np.zeros(2, dtype=np.float32)
        self.platform_offset = np.zeros(2, dtype=np.float32)
        self.platform_velocity = np.zeros(2, dtype=np.float32)

        self.grid_y, self.grid_x = np.mgrid[0:config.FRAME_HEIGHT, 0:config.FRAME_WIDTH].astype(np.float32)
        self.rng = np.random.default_rng(42)
        rng = np.random.default_rng(42)
        self.star_world_x = rng.uniform(-900, 1540, 140)
        self.star_world_y = rng.uniform(-720, 1200, 140)
        self.star_brightness = rng.integers(20, 85, 140)

        self.target_count = 1
        self.designated_target_id = "TGT-01"
        self.targets = []
        self._initialize_targets()
        self._sync_legacy_primary_attributes()

    # ------------------------------------------------------------------
    # Multi-target configuration
    # ------------------------------------------------------------------
    def _initialize_targets(self):
        self.targets = []
        count = max(1, min(5, int(self.target_count)))
        # Scale initial positions to the active frame dimensions.
        position_scale_x = config.FRAME_WIDTH / 640.0
        position_scale_y = config.FRAME_HEIGHT / 480.0
        for index in range(count):
            target_id = f"TGT-{index + 1:02d}"
            px = INITIAL_POSITIONS[index][0] * position_scale_x
            py = INITIAL_POSITIONS[index][1] * position_scale_y
            if index == 0:
                px, py = 60.0, self.center_y
            style = TARGET_SIGNATURES[target_id]
            self.targets.append({
                "id": target_id,
                "role": "DESIGNATED" if target_id == self.designated_target_id else "DECOY",
                "x": float(px),
                "y": float(py),
                "visible": True,
                "speed_scale": float(SPEED_SCALES[index]),
                "range_m": float(self.beacon_range_m * RANGE_SCALES[index]),
                "phase": float(index) * 0.83,
                "random_x": float(px),
                "random_y": float(py),
                "signature": style["name"],
                "color_bgr": tuple(style["color_bgr"]),
                "shape": style["shape"],
            })

    def set_target_count(self, count):
        self.target_count = max(1, min(5, int(count)))
        ids = [f"TGT-{i:02d}" for i in range(1, self.target_count + 1)]
        if self.designated_target_id not in ids:
            self.designated_target_id = ids[0]
        self.frame_count = 0
        self._initialize_targets()
        self._sync_legacy_primary_attributes()

    def get_target_count(self):
        return len(self.targets)

    def get_target_ids(self):
        return [target["id"] for target in self.targets]

    def set_designated_target(self, target_id):
        target_id = str(target_id)
        if target_id not in self.get_target_ids():
            return False
        self.designated_target_id = target_id
        for target in self.targets:
            target["role"] = "DESIGNATED" if target["id"] == target_id else "DECOY"
        self._sync_legacy_primary_attributes()
        return True

    def get_designated_target_id(self):
        return self.designated_target_id

    def get_designated_target_signature(self):
        for target in self.targets:
            if target["id"] == self.designated_target_id:
                return target["signature"]
        return TARGET_SIGNATURES["TGT-01"]["name"]

    def get_designated_target_color(self):
        for target in self.targets:
            if target["id"] == self.designated_target_id:
                return target["color_bgr"]
        return TARGET_SIGNATURES["TGT-01"]["color_bgr"]

    # ------------------------------------------------------------------
    # Compatibility / runtime configuration
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Direct 3D manipulation (Digital Twin drag)
    # ------------------------------------------------------------------
    def move_target(self, target_id, dx_px, dy_px):
        """Move a target by image-coordinate deltas (Digital Twin drag).

        ONE mission state: the drag lands directly in the simulator's real
        target position, so radar, virtual camera and the twin all render
        the same moved object. Deltas are clamped to the frame so the
        object can never vanish or run away from the tracking optics.
        """
        target = next((t for t in self.targets if t["id"] == str(target_id)), None)
        if target is None:
            return False
        target["x"] = float(np.clip(target["x"] + float(dx_px), 30.0, config.FRAME_WIDTH - 30.0))
        target["y"] = float(np.clip(target["y"] + float(dy_px), 30.0, config.FRAME_HEIGHT - 30.0))
        target["random_x"] = target["x"]
        target["random_y"] = target["y"]
        if target["id"] == self.designated_target_id:
            self._sync_legacy_primary_attributes()
        return True

    def _sync_legacy_primary_attributes(self):
        target = next((t for t in self.targets if t["id"] == self.designated_target_id), None)
        if target is None and self.targets:
            target = self.targets[0]
            self.designated_target_id = target["id"]
        if target is not None:
            self.beacon_x = float(target["x"])
            self.beacon_y = float(target["y"])
            self.beacon_visible = bool(target["visible"])
            if self.designated_target_id == "TGT-01":
                # Preserve the user-configured base range for the primary target.
                target["range_m"] = float(self.beacon_range_m)

    def set_motion_pattern(self, pattern):
        self.motion_pattern = str(pattern)
        self.frame_count = 0
        self._initialize_targets()
        self._sync_legacy_primary_attributes()

    def set_noise_type(self, noise_type):
        self.noise_type = str(noise_type)

    def set_beacon_visibility(self, visible):
        visible = bool(visible)
        for target in self.targets:
            if target["id"] == self.designated_target_id:
                target["visible"] = visible
        self.beacon_visible = visible

    def set_beacon_loss(self, loss_start, loss_duration):
        self.loss_start = int(loss_start)
        self.loss_end = self.loss_start + int(loss_duration)

    def clear_beacon_loss(self):
        self.loss_start = None
        self.loss_end = None
        self.set_beacon_visibility(True)

    def update_camera(self, pan, tilt):
        self.camera_pan = float(pan)
        self.camera_tilt = float(tilt)

    def set_beacon_speed(self, speed_px_per_frame):
        speed = max(0.1, float(speed_px_per_frame))
        self.speed_x = speed if self.speed_x >= 0 else -speed
        self.random_speed = speed
        self.speed_factor = speed / 4.0

    def set_beacon_range(self, range_m):
        self.beacon_range_m = max(1.0, float(range_m))
        for index, target in enumerate(self.targets):
            target["range_m"] = self.beacon_range_m * RANGE_SCALES[index]

    def set_beacon_size(self, size_px):
        self.beacon_size_px = max(5.0, min(20.0, float(size_px)))
        config.BEACON_SIZE = int(round(self.beacon_size_px))

    def set_fov(self, horizontal_fov_deg):
        horizontal = max(1.0, float(horizontal_fov_deg))
        config.FOV_HORIZONTAL = horizontal
        config.FOV_VERTICAL = horizontal * (3.0 / 4.0)
        self.refresh_camera_geometry()

    def set_disturbances(self, turbulence_enabled=False, turbulence_strength_px=0.0,
                         camera_jitter_enabled=False, camera_jitter_px=0.0,
                         platform_motion_enabled=False, platform_motion_px=0.0,
                         platform_motion_pattern="Linear", atmosphere="Clear",
                         atmosphere_level=0, disturbance_preset=None):
        self.turbulence_enabled = bool(turbulence_enabled)
        self.turbulence_strength_px = max(0.0, min(20.0, float(turbulence_strength_px)))
        self.camera_jitter_enabled = bool(camera_jitter_enabled)
        self.camera_jitter_px = max(0.0, min(20.0, float(camera_jitter_px)))
        self.platform_motion_enabled = bool(platform_motion_enabled)
        self.platform_motion_px = max(0.0, min(20.0, float(platform_motion_px)))
        self.platform_motion_pattern = str(platform_motion_pattern)
        self.atmosphere = str(atmosphere)
        self.atmosphere_level = max(0, min(100, int(atmosphere_level)))
        if disturbance_preset is not None:
            self.disturbance_preset = str(disturbance_preset)

    def set_disturbance_preset(self, preset):
        """Apply a named deterministic disturbance profile.

        These presets are shared by the headless refinement tests and the GUI
        semantics.  They describe the input disturbance; the detector receives
        the resulting rendered frame without any artificial metric adjustment.
        """
        preset = str(preset)
        profiles = {
            "Clear": {
                "noise": "None", "noise_level": 0,
                "turbulence": False, "turbulence_px": 0,
                "jitter": False, "jitter_px": 0,
                "platform": False, "platform_px": 0,
                "atmosphere": "Clear", "atmosphere_level": 0,
            },
            "Sensor Noise": {
                "noise": "Gaussian", "noise_level": 20,
                "turbulence": False, "turbulence_px": 0,
                "jitter": False, "jitter_px": 0,
                "platform": False, "platform_px": 0,
                "atmosphere": "Clear", "atmosphere_level": 0,
            },
            "Vibration": {
                "noise": "None", "noise_level": 0,
                "turbulence": True, "turbulence_px": 10,
                "jitter": True, "jitter_px": 15,
                "platform": True, "platform_px": 15,
                "atmosphere": "Clear", "atmosphere_level": 0,
            },
            "Atmosphere": {
                "noise": "None", "noise_level": 0,
                "turbulence": True, "turbulence_px": 8,
                "jitter": False, "jitter_px": 0,
                "platform": False, "platform_px": 0,
                "atmosphere": "Fog", "atmosphere_level": 70,
            },
            # The combined profile is the sustained real-time composite used
            # for the PS performance check: sensor noise + camera jitter +
            # platform motion + atmospheric haze.  Full spatial turbulence is
            # exercised separately by the Vibration/Extreme PAT profiles
            # because dense remapping is deliberately treated as a stress test.
            "Full Combined": {
                "noise": "Gaussian", "noise_level": 8,
                "turbulence": False, "turbulence_px": 0,
                "jitter": True, "jitter_px": 8,
                "platform": True, "platform_px": 8,
                "atmosphere": "Haze", "atmosphere_level": 35,
            },
            "Extreme PAT": {
                "noise": "Gaussian", "noise_level": 20,
                "turbulence": True, "turbulence_px": 20,
                "jitter": True, "jitter_px": 20,
                "platform": True, "platform_px": 20,
                "atmosphere": "Fog", "atmosphere_level": 85,
            },
        }
        profile = profiles.get(preset, profiles["Clear"])
        self.disturbance_preset = preset if preset in profiles else "Clear"
        self.set_noise_type(profile["noise"])
        self.noise_level = int(profile["noise_level"])
        self.set_disturbances(
            turbulence_enabled=profile["turbulence"],
            turbulence_strength_px=profile["turbulence_px"],
            camera_jitter_enabled=profile["jitter"],
            camera_jitter_px=profile["jitter_px"],
            platform_motion_enabled=profile["platform"],
            platform_motion_px=profile["platform_px"],
            platform_motion_pattern="Linear",
            atmosphere=profile["atmosphere"],
            atmosphere_level=profile["atmosphere_level"],
            disturbance_preset=self.disturbance_preset,
        )
        return dict(profile)

    def get_disturbance_status(self):
        active = []
        if self.noise_type != "None" and self.noise_level > 0:
            active.append(f"noise:{self.noise_type}")
        if self.turbulence_enabled and self.turbulence_strength_px > 0:
            active.append(f"turbulence:{self.turbulence_strength_px:.0f}px")
        if self.camera_jitter_enabled and self.camera_jitter_px > 0:
            active.append(f"jitter:{self.camera_jitter_px:.0f}px")
        if self.platform_motion_enabled and self.platform_motion_px > 0:
            active.append(f"platform:{self.platform_motion_px:.0f}px")
        if self.atmosphere != "Clear" and self.atmosphere_level > 0:
            active.append(f"atmosphere:{self.atmosphere} {self.atmosphere_level}%")
        return {
            "preset": self.disturbance_preset,
            "active": active,
            "count": len(active),
        }

    def refresh_camera_geometry(self):
        self.px_per_degree_x = config.FRAME_WIDTH / config.FOV_HORIZONTAL
        self.px_per_degree_y = config.FRAME_HEIGHT / config.FOV_VERTICAL

    # ------------------------------------------------------------------
    # Target motion
    # ------------------------------------------------------------------
    def update_beacon(self):
        for target in self.targets:
            self._update_target(target)
        self.frame_count += 1
        self._update_visibility_from_loss_window()
        self._sync_legacy_primary_attributes()

    def _update_target(self, target):
        index = int(target["id"][-2:]) - 1
        scale = float(target["speed_scale"])
        pattern = self.motion_pattern
        if pattern == "Straight Line":
            direction = 1.0 if scale >= 0 else -1.0
            target["x"] += abs(self.speed_x) * scale
            if index == 0:
                target["y"] = self.center_y
            else:
                target["y"] = self.center_y + (index * 28.0 - 42.0) + 34.0 * math.sin(self.frame_count * 0.035 + target["phase"])
            if target["x"] >= config.FRAME_WIDTH - 45:
                target["x"] = config.FRAME_WIDTH - 45
                target["speed_scale"] = -abs(target["speed_scale"])
            elif target["x"] <= 45:
                target["x"] = 45.0
                target["speed_scale"] = abs(target["speed_scale"])
        elif pattern == "Circular":
            angle = self.frame_count * 0.025 * self.speed_factor * scale + target["phase"]
            rx = 145.0 + 24.0 * index
            ry = 90.0 + 14.0 * index
            target["x"] = self.center_x + rx * math.cos(angle)
            target["y"] = self.center_y + ry * math.sin(angle)
        elif pattern == "Figure 8":
            angle = self.frame_count * 0.025 * self.speed_factor * scale + target["phase"]
            rx = 195.0 + 24.0 * index
            ry = 110.0 + 12.0 * index
            target["x"] = self.center_x + rx * math.sin(angle)
            target["y"] = self.center_y + ry * math.sin(angle) * math.cos(angle)
        else:
            dx = target["random_x"] - target["x"]
            dy = target["random_y"] - target["y"]
            distance = math.hypot(dx, dy)
            if distance < 14:
                margin = 50
                target["random_x"] = float(self.rng.integers(margin, config.FRAME_WIDTH - margin))
                target["random_y"] = float(self.rng.integers(margin, config.FRAME_HEIGHT - margin))
            else:
                step = self.random_speed * abs(scale)
                target["x"] += dx / distance * step
                target["y"] += dy / distance * step

    def _update_visibility_from_loss_window(self):
        selected_visible = True
        if self.loss_start is not None and self.loss_end is not None:
            selected_visible = not (self.loss_start <= self.frame_count < self.loss_end)
        for target in self.targets:
            target["visible"] = True
            if target["id"] == self.designated_target_id:
                target["visible"] = selected_visible
        self.beacon_visible = selected_visible

    # ------------------------------------------------------------------
    # Geometry / truth
    # ------------------------------------------------------------------
    def world_to_image(self, world_x, world_y):
        return (
            self.center_x + (world_x - self.center_x) - self.camera_pan * self.px_per_degree_x,
            self.center_y + (world_y - self.center_y) - self.camera_tilt * self.px_per_degree_y,
        )

    def _disturbed_image_position(self, target):
        x, y = self.world_to_image(target["x"], target["y"])
        px, py = self.last_platform_offset
        jx, jy = self.last_camera_jitter
        shifted_x, shifted_y = x + px + jx, y + py + jy
        tx, ty = self._turbulence_at(shifted_x, shifted_y)
        return shifted_x + tx, shifted_y + ty

    def get_target_image_positions(self, visible_only=False):
        result=[]
        for target in self.targets:
            if visible_only and not target["visible"]:
                continue
            x,y=self._disturbed_image_position(target)
            result.append({
                "id":target["id"], "role":target["role"], "x":float(x), "y":float(y),
                "visible":bool(target["visible"]), "range_m":float(target["range_m"]),
                "signature":target["signature"], "color_bgr":target["color_bgr"], "shape":target["shape"]
            })
        return result

    def get_target_states(self):
        return self.get_target_image_positions(False)

    def get_target_angles(self):
        for target in self.targets:
            if target["id"] == self.designated_target_id:
                x,y=self._disturbed_image_position(target)
                return ((x-self.center_x)/self.px_per_degree_x, (y-self.center_y)/self.px_per_degree_y)
        return (0.0,0.0)

    def get_all_target_angles(self):
        out=[]
        for target in self.targets:
            x,y=self._disturbed_image_position(target)
            out.append({"id":target["id"],"role":target["role"],"pan":(x-self.center_x)/self.px_per_degree_x,"tilt":(y-self.center_y)/self.px_per_degree_y,"range_m":float(target["range_m"])})
        return out

    def get_digital_twin_state(self):
        """Return the single world-state snapshot consumed by the 3D twin.

        The returned target x/y values are the simulator's pre-camera world
        coordinates, so the 3D view is not a camera-POV visualization.
        Disturbance telemetry comes from the same frame that was rendered for
        detection, keeping radar/camera/twin synchronized to one time step.
        """
        beam_wander = (0.0, 0.0)
        if self.last_turbulence_dx is not None and self.targets:
            target = next((t for t in self.targets if t["id"] == self.designated_target_id), None)
            if target is None:
                target = self.targets[0]
            x, y = self.world_to_image(target["x"], target["y"])
            ix = int(np.clip(round(x), 0, config.FRAME_WIDTH - 1))
            iy = int(np.clip(round(y), 0, config.FRAME_HEIGHT - 1))
            beam_wander = (float(self.last_turbulence_dx[iy, ix]), float(self.last_turbulence_dy[iy, ix]))

        px, py = self.last_platform_offset
        jx, jy = self.last_camera_jitter
        camera_position = (float((px + jx) / 32.0), float(-(py + jy) / 32.0), 0.0)

        return {
            "motion_pattern": self.motion_pattern,
            "frame_count": int(self.frame_count),
            "camera_position": camera_position,
            "platform_offset_px": (float(px), float(py)),
            "jitter_px": (float(jx), float(jy)),
            "beam_wander_px": beam_wander,
            "targets": [
                {
                    "id": str(t["id"]),
                    "role": str(t["role"]),
                    "x": float(t["x"]),
                    "y": float(t["y"]),
                    "range_m": float(t["range_m"]),
                    "visible": bool(t["visible"]),
                    "signature": str(t["signature"]),
                    "shape": str(t["shape"]),
                }
                for t in self.targets
            ],
        }

    def get_ground_truth_image_position(self):
        for target in self.targets:
            if target["id"] == self.designated_target_id and target["visible"]:
                x,y=self._disturbed_image_position(target)
                margin=self.beacon_size_px/2.0
                if -margin <= x < config.FRAME_WIDTH + margin and -margin <= y < config.FRAME_HEIGHT + margin:
                    return float(x),float(y)
        return None

    def _current_platform_offset(self):
        """Return cumulative apparent scene offset caused by platform motion.

        The GUI value is the maximum per-frame displacement, matching the PS
        disturbance wording. The accumulated offset is bounded so the scene
        does not drift away indefinitely.
        """
        if not self.platform_motion_enabled or self.platform_motion_px <= 0:
            self.platform_offset[:] = 0.0
            self.platform_velocity[:] = 0.0
            return 0.0, 0.0

        step = float(self.platform_motion_px)
        limit = 120.0
        pattern = self.platform_motion_pattern
        phase = self.frame_count * 0.07

        if pattern == "Circular":
            delta_x = step * math.cos(phase)
            delta_y = step * math.sin(phase)
        elif pattern == "Figure 8":
            delta_x = step * math.cos(phase)
            delta_y = step * math.sin(2.0 * phase) * 0.7
        elif pattern == "Random":
            delta = self.rng.uniform(-step, step, 2)
            delta = delta.astype(np.float32)
            self.last_platform_random = 0.75 * self.last_platform_random + 0.25 * delta
            delta_x, delta_y = float(self.last_platform_random[0]), float(self.last_platform_random[1])
        else:
            # Linear / mandatory default: bounded platform velocity with a
            # smooth reversal instead of an instantaneous 40 px/frame jump.
            direction = 1.0 if int(self.frame_count / 120) % 2 == 0 else -1.0
            desired_vx = direction * step
            max_accel = max(0.5, 0.08 * step)
            self.platform_velocity[0] += float(
                np.clip(desired_vx - self.platform_velocity[0], -max_accel, max_accel)
            )
            self.platform_velocity[1] = 0.12 * step * math.sin(phase)
            delta_x = float(self.platform_velocity[0])
            delta_y = float(self.platform_velocity[1])

        self.platform_offset[0] += delta_x
        self.platform_offset[1] += delta_y
        self.platform_offset[0] = float(np.clip(self.platform_offset[0], -limit, limit))
        self.platform_offset[1] = float(np.clip(self.platform_offset[1], -limit, limit))
        return float(self.platform_offset[0]), float(self.platform_offset[1])

    def _new_camera_jitter(self):
        """Generate bounded, temporally correlated camera jitter.

        The configured value is the maximum displacement envelope (±px). A
        smoothed random walk is used so the disturbance resembles mechanical
        vibration instead of teleporting the image to an unrelated offset each
        frame.
        """
        if not self.camera_jitter_enabled or self.camera_jitter_px <= 0:
            self.camera_jitter_offset[:] = 0.0
            return 0.0, 0.0

        limit = float(self.camera_jitter_px)
        delta = self.rng.normal(0.0, max(1.0, 0.35 * limit), 2).astype(np.float32)
        self.camera_jitter_offset *= 0.72
        self.camera_jitter_offset += 0.28 * delta
        self.camera_jitter_offset = np.clip(
            self.camera_jitter_offset, -limit, limit
        )
        return float(self.camera_jitter_offset[0]), float(self.camera_jitter_offset[1])

    def _make_turbulence_field(self):
        strength = float(self.turbulence_strength_px)
        if not self.turbulence_enabled or strength <= 0:
            self.last_turbulence_dx = None
            self.last_turbulence_dy = None
            self.turbulence_field_x = None
            self.turbulence_field_y = None
            return None, None

        h, w = config.FRAME_HEIGHT, config.FRAME_WIDTH
        gh, gw = 24, 32
        small_x = self.rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32)
        small_y = self.rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32)
        small_x = cv2.GaussianBlur(small_x, (0, 0), 2.3)
        small_y = cv2.GaussianBlur(small_y, (0, 0), 2.3)
        dx = cv2.resize(small_x, (w, h), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(small_y, (w, h), interpolation=cv2.INTER_CUBIC)

        if self.turbulence_field_x is None:
            self.turbulence_field_x = dx
            self.turbulence_field_y = dy
        else:
            self.turbulence_field_x = 0.94 * self.turbulence_field_x + 0.06 * dx
            self.turbulence_field_y = 0.94 * self.turbulence_field_y + 0.06 * dy
        dx = self.turbulence_field_x
        dy = self.turbulence_field_y

        def normalize(field):
            max_abs = float(np.max(np.abs(field)))
            if max_abs < 1e-6:
                return np.zeros_like(field)
            return field * (strength / max_abs)

        self.last_turbulence_dx = normalize(dx)
        self.last_turbulence_dy = normalize(dy)
        return self.last_turbulence_dx, self.last_turbulence_dy

    def _turbulence_at(self, x, y):
        if self.last_turbulence_dx is None:
            return 0.0, 0.0
        ix = int(np.clip(round(x), 0, config.FRAME_WIDTH - 1))
        iy = int(np.clip(round(y), 0, config.FRAME_HEIGHT - 1))
        return float(self.last_turbulence_dx[iy, ix]), float(self.last_turbulence_dy[iy, ix])

    def get_ground_truth_image_position(self):
        """Return beacon centroid in the current *disturbed* camera feed."""
        if not self.beacon_visible:
            return None
        x, y = self.world_to_image(self.beacon_x, self.beacon_y)
        px, py = self.last_platform_offset
        jx, jy = self.last_camera_jitter
        shifted_x = x + px + jx
        shifted_y = y + py + jy
        tx, ty = self._turbulence_at(shifted_x, shifted_y)
        x = shifted_x + tx
        y = shifted_y + ty

        margin = self.beacon_size_px / 2.0
        if -margin <= x < config.FRAME_WIDTH + margin and -margin <= y < config.FRAME_HEIGHT + margin:
            return float(x), float(y)
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def create_background(self):
        h, w = config.FRAME_HEIGHT, config.FRAME_WIDTH
        frame = np.empty((h, w, 3), dtype=np.uint8)
        gradient = np.linspace(7, 24, h, dtype=np.uint8)
        frame[:, :, 0] = gradient[:, None]
        frame[:, :, 1] = gradient[:, None]
        frame[:, :, 2] = gradient[:, None]

        for world_x, world_y, brightness in zip(
            self.star_world_x, self.star_world_y, self.star_brightness
        ):
            sx, sy = self.world_to_image(world_x, world_y)
            ix, iy = int(round(sx)), int(round(sy))
            if 0 <= ix < w and 0 <= iy < h:
                value = int(brightness)
                frame[iy, ix] = (value, value, value)
        return frame

    def _render_beacon(self, frame, visible_x, visible_y):
        size = max(5.0, float(self.beacon_size_px))
        half = max(2, int(math.ceil(size / 2.0)))
        pad = 3
        x0 = max(0, int(math.floor(visible_x)) - half - pad)
        y0 = max(0, int(math.floor(visible_y)) - half - pad)
        x1 = min(frame.shape[1], int(math.ceil(visible_x)) + half + pad + 1)
        y1 = min(frame.shape[0], int(math.ceil(visible_y)) + half + pad + 1)
        if x0 >= x1 or y0 >= y1:
            return

        local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cx = int(round(visible_x)) - x0
        cy = int(round(visible_y)) - y0
        cv2.rectangle(local, (cx - half, cy - half), (cx + half, cy + half), 235, -1)
        local = cv2.GaussianBlur(local, (0, 0), sigmaX=0.9)

        # Mild distance-based attenuation; keeps default range bright while
        # allowing range tests to exercise the detector realistically.
        brightness_scale = math.sqrt(1000.0 / max(self.beacon_range_m, 1.0))
        brightness_scale = float(np.clip(brightness_scale, 0.45, 1.15))
        local = np.clip(local.astype(np.float32) * brightness_scale, 0, 255).astype(np.uint8)
        local = np.maximum(local, 8).astype(np.uint8)

        roi = frame[y0:y1, x0:x1]
        roi[:, :, 0] = np.maximum(roi[:, :, 0], local)
        roi[:, :, 1] = np.maximum(roi[:, :, 1], local)
        roi[:, :, 2] = np.maximum(roi[:, :, 2], local)

    def create_background(self):
        h, w = config.FRAME_HEIGHT, config.FRAME_WIDTH
        frame = np.empty((h, w, 3), dtype=np.uint8)
        gradient = np.linspace(7, 24, h, dtype=np.uint8)
        frame[:, :, 0] = gradient[:, None]
        frame[:, :, 1] = gradient[:, None]
        frame[:, :, 2] = gradient[:, None]

        for world_x, world_y, brightness in zip(
            self.star_world_x, self.star_world_y, self.star_brightness
        ):
            sx, sy = self.world_to_image(world_x, world_y)
            ix, iy = int(round(sx)), int(round(sy))
            if 0 <= ix < w and 0 <= iy < h:
                value = int(brightness)
                frame[iy, ix] = (value, value, value)
        return frame

    def _render_target(self, frame, visible_x, visible_y, size_px, color_bgr, shape):
        size=max(5.0,float(size_px))
        half=max(2,int(math.ceil(size/2.0)))
        cx,cy=int(round(visible_x)),int(round(visible_y))
        if cx < -half-4 or cy < -half-4 or cx >= frame.shape[1]+half+4 or cy >= frame.shape[0]+half+4:
            return
        brightness_scale = float(np.clip(math.sqrt(1000.0/max(self.beacon_range_m,1.0)),0.55,1.15))
        color=tuple(int(np.clip(c*brightness_scale,0,255)) for c in color_bgr)
        if shape=="square":
            cv2.rectangle(frame,(cx-half,cy-half),(cx+half,cy+half),color,-1)
            inner=max(1,half//3)
            cv2.rectangle(frame,(cx-inner,cy-inner),(cx+inner,cy+inner),(255,255,255),-1)
        elif shape=="diamond":
            pts=np.array([[cx,cy-half-1],[cx+half+1,cy],[cx,cy+half+1],[cx-half-1,cy]],np.int32)
            cv2.fillConvexPoly(frame,pts,color)
        elif shape=="circle":
            cv2.circle(frame,(cx,cy),half+1,color,-1)
            cv2.circle(frame,(cx,cy),max(1,half//3),(255,255,255),-1)
        elif shape=="triangle":
            pts=np.array([[cx,cy-half-2],[cx+half+2,cy+half],[cx-half-2,cy+half]],np.int32)
            cv2.fillConvexPoly(frame,pts,color)
        else:
            cv2.circle(frame,(cx,cy),half,color,-1)
        x0=max(0,cx-half-3); y0=max(0,cy-half-3); x1=min(frame.shape[1],cx+half+4); y1=min(frame.shape[0],cy+half+4)
        if x1>x0 and y1>y0:
            roi=frame[y0:y1,x0:x1].copy()
            blur=cv2.GaussianBlur(roi,(0,0),sigmaX=1.1)
            frame[y0:y1,x0:x1]=np.maximum(roi,(blur*0.20).astype(np.uint8))

    @staticmethod
    def _translate(frame, dx, dy):
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return frame
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(
            frame,
            matrix,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    def _apply_turbulence(self, frame):
        if self.last_turbulence_dx is None:
            return frame
        h, w = frame.shape[:2]
        # Positive field moves content in the positive screen direction.
        map_x = self.grid_x - self.last_turbulence_dx
        map_y = self.grid_y - self.last_turbulence_dy
        return cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    def apply_atmosphere(self, frame):
        level = max(0, min(100, int(self.atmosphere_level)))
        if self.atmosphere == "Clear" or level <= 0:
            return frame

        result = frame.astype(np.float32)
        alpha = level / 100.0

        if self.atmosphere == "Haze":
            # Haze primarily reduces contrast through a spatially uniform veil.
            # Keep blur out of this preset so the combined real-time profile
            # remains representative of a practical coarse-alignment loop.
            veil = np.full_like(result, 72.0)
            result = result * (1.0 - 0.34 * alpha) + veil * (0.34 * alpha)

        elif self.atmosphere == "Fog":
            # Spatially varying veil, but retain sufficient beacon/background
            # contrast for a meaningful tracking test.
            h, w = frame.shape[:2]
            y = np.linspace(0.20, 1.0, h, dtype=np.float32)[:, None]
            fog_alpha = (0.10 + 0.35 * y) * alpha
            fog = np.full_like(result, 112.0)
            result = result * (1.0 - fog_alpha[:, :, None]) + fog * fog_alpha[:, :, None]
            result = cv2.GaussianBlur(result, (0, 0), sigmaX=0.5 + 1.4 * alpha)

        elif self.atmosphere == "Rain":
            h, w = frame.shape[:2]
            count = int(8 + 90 * alpha)
            rain = result.copy()
            for _ in range(count):
                x = int(self.rng.integers(0, w))
                y = int(self.rng.integers(0, h))
                length = int(self.rng.integers(5, 16))
                cv2.line(rain, (x, y), (min(w - 1, x + 2), min(h - 1, y + length)), (125, 125, 125), 1)
            result = cv2.addWeighted(result, 0.93, rain, 0.07 + 0.12 * alpha, 0.0)

        elif self.atmosphere == "Low Light":
            gain = 1.0 - 0.60 * alpha
            result *= gain
            # A mild gamma correction preserves faint beacon contrast instead
            # of collapsing the whole feed into near-black pixels.
            gamma = 1.0 + 0.10 * alpha
            normalized = np.clip(result / 255.0, 0.0, 1.0)
            result = (normalized ** gamma) * 255.0

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_noise(self, frame):
        level = max(0, float(self.noise_level))
        if self.noise_type == "None" or level <= 0.0:
            return frame

        h, w = frame.shape[:2]

        if self.noise_type == "Salt & Pepper":
            amount = min(1.0, level / 100.0)
            mask = self.rng.random((h, w)) < amount
            salt = self.rng.random((h, w)) < 0.5
            frame[salt & mask] = 255
            frame[(~salt) & mask] = 0
            return frame

        if self.noise_type == "Gaussian":
            # Use one 2-D sensor-noise field across the three channels.  This
            # preserves the beacon's colour signature while avoiding the much
            # more expensive allocation of an H×W×3 random tensor.
            noise = self.rng.normal(0.0, level, (h, w)).astype(np.float32)
            noisy = frame.astype(np.float32) + noise[:, :, None]
            return np.clip(noisy, 0.0, 255.0).astype(np.uint8)

        if self.noise_type == "Poisson":
            # Use a coarse shot-noise field and interpolate it to sensor
            # resolution. This keeps the spatial character of photon noise
            # while avoiding a full-resolution Poisson sample on every channel.
            peak_counts = max(32.0, 1024.0 / (1.0 + 0.20 * level))
            gray_small = cv2.resize(frame[:, :, 0].astype(np.float32), (160, 120), interpolation=cv2.INTER_AREA)
            signal = gray_small / 255.0
            counts = self.rng.poisson(np.clip(signal, 0.0, 1.0) * peak_counts)
            noisy_small = np.clip(counts.astype(np.float32) / peak_counts * 255.0, 0.0, 255.0)
            gray_small = gray_small.astype(np.float32)
            delta_small = noisy_small - gray_small
            delta = cv2.resize(delta_small, (w, h), interpolation=cv2.INTER_LINEAR)
            noisy = np.clip(frame.astype(np.float32) + delta[:, :, None], 0.0, 255.0)
            return noisy.astype(np.uint8)

        return frame

    def get_frame(self):
        self.last_camera_jitter = self._new_camera_jitter()
        self.last_platform_offset = self._current_platform_offset()
        self._make_turbulence_field()
        frame = self.create_background()
        self.update_beacon()
        for target in self.targets:
            if not target["visible"]:
                continue
            x,y=self.world_to_image(target["x"],target["y"])
            self._render_target(frame,x,y,self.beacon_size_px,target["color_bgr"],target["shape"])
        frame=self._translate(frame,*self.last_platform_offset)
        frame=self._translate(frame,*self.last_camera_jitter)
        frame=self._apply_turbulence(frame)
        frame=self.apply_atmosphere(frame)
        return self.apply_noise(frame)
