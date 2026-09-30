"""Configurable pan/tilt controller for the virtual FSOC camera."""

from __future__ import annotations

import config


class CameraController:
    def __init__(self):
        self.pan = 0.0
        self.tilt = 0.0
        self.dead_zone_px = 0.25
        self.pan_limit_deg = 5.0
        self.tilt_limit_deg = 5.0
        self.gain = 2.0

    def set_angle_limits(self, pan_limit_deg, tilt_limit_deg=None):
        self.pan_limit_deg = max(1.0, float(pan_limit_deg))
        self.tilt_limit_deg = (
            self.pan_limit_deg
            if tilt_limit_deg is None
            else max(0.75, float(tilt_limit_deg))
        )
        self.pan = max(-self.pan_limit_deg, min(self.pan, self.pan_limit_deg))
        self.tilt = max(-self.tilt_limit_deg, min(self.tilt, self.tilt_limit_deg))

    def set_gain(self, gain):
        self.gain = max(0.1, min(2.0, float(gain)))

    def update(self, error_x, error_y):
        px_per_degree_x = config.FRAME_WIDTH / config.FOV_HORIZONTAL
        px_per_degree_y = config.FRAME_HEIGHT / config.FOV_VERTICAL

        desired_pan = self.pan
        desired_tilt = self.tilt

        if abs(error_x) > self.dead_zone_px:
            desired_pan += self.gain * error_x / px_per_degree_x
        if abs(error_y) > self.dead_zone_px:
            desired_tilt += self.gain * error_y / px_per_degree_y

        max_pan_step = config.MAX_PAN_SPEED / config.UPDATE_RATE
        max_tilt_step = config.MAX_TILT_SPEED / config.UPDATE_RATE

        pan_step = max(-max_pan_step, min(desired_pan - self.pan, max_pan_step))
        tilt_step = max(-max_tilt_step, min(desired_tilt - self.tilt, max_tilt_step))

        self.pan += pan_step
        self.tilt += tilt_step

        self.pan = max(-self.pan_limit_deg, min(self.pan, self.pan_limit_deg))
        self.tilt = max(-self.tilt_limit_deg, min(self.tilt, self.tilt_limit_deg))

        return self.pan, self.tilt
