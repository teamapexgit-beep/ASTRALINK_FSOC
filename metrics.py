"""Shared metric calculations for the FSOC application.

This module intentionally contains pure calculation helpers rather than a
second independent metric accumulator. GUI and MP4 benchmark code use the
same formulas so reported numbers cannot drift between execution paths.
"""
from __future__ import annotations

import math
from typing import Iterable


def average(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def maximum(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return max(vals, default=0.0)


def rmse(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else 0.0


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator > 0 else 0.0


def lock_retention(tracking_frames: int, evaluated_frames: int) -> float:
    return percentage(tracking_frames, evaluated_frames)


def target_loss(missed_frames: int, evaluated_frames: int) -> float:
    return percentage(missed_frames, evaluated_frames)


def average_reacquisition(times: Iterable[float]) -> float | None:
    vals = [float(v) for v in times]
    return sum(vals) / len(vals) if vals else None


def maximum_reacquisition(times: Iterable[float]) -> float | None:
    vals = [float(v) for v in times]
    return max(vals) if vals else None
