"""Lightweight constant-velocity Kalman estimator for beacon tracking."""

from __future__ import annotations

import numpy as np

import config


class BeaconTracker:
    def __init__(self):
        self.nominal_dt = 1.0 / config.UPDATE_RATE
        self.measurement_variance = 1.00
        self.acceleration_variance = 90000.0

        self.state_vector = None
        self.covariance = None
        self._last_timestamp = None
        self.confirmed_measurements = 0

        self.x = None
        self.y = None
        self.previous_x = None
        self.previous_y = None
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.raw_velocity_x = 0.0
        self.raw_velocity_y = 0.0
        self._previous_measurement = None
        self._velocity_history = []
        self.position_uncertainty = None
        self.confidence = 0.0
        self.state = "SEARCHING"
        self.lost_frames = 0
        self.max_lost_frames = 10

    def _time_step(self, timestamp):
        if timestamp is None:
            return self.nominal_dt
        timestamp = float(timestamp)
        if self._last_timestamp is None:
            dt = self.nominal_dt
        else:
            dt = min(max(timestamp - self._last_timestamp, 1e-3), 0.5)
        self._last_timestamp = timestamp
        return dt

    def _predict(self, dt):
        transition = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        process_noise = self.acceleration_variance * np.array([
            [dt ** 4 / 4.0, 0.0, dt ** 3 / 2.0, 0.0],
            [0.0, dt ** 4 / 4.0, 0.0, dt ** 3 / 2.0],
            [dt ** 3 / 2.0, 0.0, dt ** 2, 0.0],
            [0.0, dt ** 3 / 2.0, 0.0, dt ** 2],
        ])
        self.state_vector = transition @ self.state_vector
        self.covariance = (
            transition @ self.covariance @ transition.T + process_noise
        )

    def _update_measurement(self, detection):
        measurement = np.asarray(detection, dtype=float)
        observation = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        measurement_noise = np.eye(2) * self.measurement_variance

        innovation = measurement - observation @ self.state_vector
        innovation_covariance = (
            observation @ self.covariance @ observation.T
            + measurement_noise
        )
        kalman_gain = np.linalg.solve(
            innovation_covariance,
            observation @ self.covariance,
        ).T

        self.state_vector += kalman_gain @ innovation

        identity = np.eye(4)
        residual = identity - kalman_gain @ observation
        self.covariance = (
            residual @ self.covariance @ residual.T
            + kalman_gain @ measurement_noise @ kalman_gain.T
        )

    def _publish(self):
        self.previous_x = self.x
        self.previous_y = self.y
        self.x = float(self.state_vector[0])
        self.y = float(self.state_vector[1])
        self.velocity_x = float(self.state_vector[2])
        self.velocity_y = float(self.state_vector[3])

        variance = max(
            0.0,
            float(self.covariance[0, 0] + self.covariance[1, 1]),
        )
        self.position_uncertainty = variance ** 0.5
        self.confidence = 1.0 / (1.0 + self.position_uncertainty)

    def _result(self):
        return int(round(self.x)), int(round(self.y)), self.state

    def update(self, detection, timestamp=None):
        dt = self._time_step(timestamp)

        if detection is None:
            if self.state_vector is None:
                self.state = "SEARCHING"
                return None
            self._predict(dt)
            self._publish()
            self.confirmed_measurements = 0
            self._velocity_history.clear()
            self.lost_frames += 1
            self.state = (
                "PREDICTING"
                if self.lost_frames <= self.max_lost_frames
                else "LOST"
            )
            return self._result()

        first = self.state_vector is None
        was_lost = self.state == "LOST"
        gap = self.lost_frames > 0

        if not first and self._previous_measurement is not None and dt > 1e-6:
            raw_vx = (float(detection[0]) - self._previous_measurement[0]) / dt
            raw_vy = (float(detection[1]) - self._previous_measurement[1]) / dt
            raw_vx = float(np.clip(raw_vx, -900.0, 900.0))
            raw_vy = float(np.clip(raw_vy, -900.0, 900.0))
            self.raw_velocity_x = raw_vx
            self.raw_velocity_y = raw_vy
            self._velocity_history.append((raw_vx, raw_vy))
            self._velocity_history = self._velocity_history[-5:]

        if first:
            mx, my = detection
            self.state_vector = np.array(
                [float(mx), float(my), 0.0, 0.0],
                dtype=float,
            )
            self.covariance = np.diag([
                self.measurement_variance,
                self.measurement_variance,
                2500.0,
                2500.0,
            ])
            self.confirmed_measurements = 1
            self.state = "ACQUIRING"
        else:
            self._predict(dt)
            self._update_measurement(detection)
            if was_lost or gap:
                self.confirmed_measurements = 1
                self.state = "RE-ACQUIRING"
            elif self.state in {"ACQUIRING", "RE-ACQUIRING"}:
                self.confirmed_measurements += 1
                self.state = (
                    "TRACKING"
                    if self.confirmed_measurements >= 2
                    else self.state
                )
            else:
                self.confirmed_measurements += 1
                self.state = "TRACKING"

        self.lost_frames = 0
        self._publish()

        # Use a short robust median of recent measured velocities to reduce
        # Kalman lag during sudden platform/turbulence motion. The filtered
        # estimate remains dominant, so this is feed-forward rather than a
        # replacement for the Kalman state.
        if self._velocity_history:
            history = np.asarray(self._velocity_history, dtype=float)
            median_vx = float(np.median(history[:, 0]))
            median_vy = float(np.median(history[:, 1]))
            self.velocity_x = 0.60 * self.velocity_x + 0.40 * median_vx
            self.velocity_y = 0.60 * self.velocity_y + 0.40 * median_vy

        self._previous_measurement = (float(detection[0]), float(detection[1]))
        return self._result()

    def reset(self):
        self.__init__()
