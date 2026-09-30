"""Lightweight learned beacon-candidate verifier.

This module intentionally has no external ML dependency.  A tiny logistic
model is trained once on synthetic candidate-shape features.  The detector
uses it as a second-stage verifier after ordinary computer-vision thresholding.
"""

from __future__ import annotations

import numpy as np


class BeaconAIVerifier:
    """Small synthetic-trained classifier for beacon-like blob candidates."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros(5, dtype=float)
        self.std = np.ones(5, dtype=float)
        self.weights = np.zeros(5, dtype=float)
        self.bias = 0.0
        self._train()

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _train(self):
        # Features:
        # [area/expected_area, aspect_ratio, mean_brightness,
        #  fill_ratio, contour_circularity]
        n = 900

        positive = np.column_stack([
            self.rng.normal(1.0, 0.25, n),
            self.rng.normal(1.0, 0.16, n),
            self.rng.normal(0.96, 0.05, n),
            self.rng.normal(0.88, 0.08, n),
            self.rng.normal(0.78, 0.10, n),
        ])

        negative = np.column_stack([
            self.rng.uniform(0.1, 6.0, n),
            self.rng.uniform(0.2, 3.5, n),
            self.rng.uniform(0.05, 1.0, n),
            self.rng.uniform(0.1, 1.0, n),
            self.rng.uniform(0.05, 1.1, n),
        ])

        positive[:, 0] = np.clip(positive[:, 0], 0.2, 2.5)
        positive[:, 1] = np.clip(positive[:, 1], 0.45, 1.8)
        positive[:, 2:] = np.clip(positive[:, 2:], 0.0, 1.2)

        x = np.vstack([positive, negative])
        y = np.concatenate([
            np.ones(n, dtype=float),
            np.zeros(n, dtype=float),
        ])

        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        xn = (x - self.mean) / self.std

        w = np.zeros(xn.shape[1], dtype=float)
        b = 0.0
        lr = 0.08
        l2 = 0.002

        for _ in range(320):
            p = self._sigmoid(xn @ w + b)
            grad_w = (xn.T @ (p - y)) / len(y) + l2 * w
            grad_b = float(np.mean(p - y))
            w -= lr * grad_w
            b -= lr * grad_b

        self.weights = w
        self.bias = b

    def score(self, features) -> float:
        features = np.asarray(features, dtype=float)
        xn = (features - self.mean) / self.std
        return float(self._sigmoid(xn @ self.weights + self.bias))


_default_verifier = BeaconAIVerifier()


def beacon_ai_score(features) -> float:
    """Return the learned beacon-likeness probability in [0, 1]."""
    return _default_verifier.score(features)
