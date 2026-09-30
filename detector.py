"""Multi-candidate beacon detector with CV + lightweight AI verification."""
from __future__ import annotations

import math

import cv2
import numpy as np

import config
from ai_verifier import beacon_ai_score


_SIGNATURE_HUE_RANGES = {
    "GREEN-SQUARE": (45, 85),
    "AMBER-DIAMOND": (8, 35),
    "MAGENTA-CIRCLE": (135, 175),
    "CYAN-TRIANGLE": (80, 110),
    "RED-CIRCLE": None,
}


def _signature_score(roi_bgr, signature: str) -> float:
    if roi_bgr is None or roi_bgr.size == 0 or not signature:
        return 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat > 70) & (val > 70)
    if not np.any(mask):
        return 0.0

    hue = hsv[:, :, 0]
    wanted = _SIGNATURE_HUE_RANGES.get(signature)
    if wanted is None and signature == "RED-CIRCLE":
        matches = ((hue <= 8) | (hue >= 172)) & mask
    elif wanted is None:
        return 0.0
    else:
        lo, hi = wanted
        matches = (hue >= lo) & (hue <= hi) & mask
    return float(np.mean(matches[mask]))


def _shape_name(contour) -> str:
    perimeter = max(float(cv2.arcLength(contour, True)), 1e-6)
    approx = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
    n = len(approx)
    if 4 <= n <= 5:
        return "QUADRILATERAL"
    if n == 3:
        return "TRIANGLE"
    if n >= 7:
        return "ROUND"
    return "OTHER"


def _extract_candidates(cleaned, frame, threshold_value, expected_position=None, preferred_signature=None):
    """Convert a binary candidate mask into scored beacon candidates."""
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    expected_area = float(max(config.BEACON_SIZE, 1) ** 2)
    candidates = []
    height_total, width_total = frame.shape[:2]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if preferred_signature is None else None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 8.0 or area > max(900.0, expected_area * 9.0):
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        aspect = width / float(height)
        if not 0.25 <= aspect <= 3.5:
            continue

        roi = frame[y:y + height, x:x + width]
        if roi.size == 0:
            continue

        perimeter = max(float(cv2.arcLength(contour, True)), 1e-6)
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        fill_ratio = area / float(width * height)
        roi_gray = (
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if frame_gray is None
            else frame_gray[y:y + height, x:x + width]
        )
        brightness = float(roi_gray.mean()) / 255.0
        area_ratio = area / max(expected_area, 1.0)

        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-9:
            shape_cx = moments["m10"] / moments["m00"]
            shape_cy = moments["m01"] / moments["m00"]
        else:
            shape_cx = x + width * 0.5
            shape_cy = y + height * 0.5

        local = cv2.GaussianBlur(roi_gray, (3, 3), 0.65)
        baseline = float(np.percentile(local, 20))
        weights = np.clip(
            local.astype(np.float32) - baseline, 0.0, 255.0
        ).astype(np.uint8)
        m_w = cv2.moments(weights, binaryImage=False)
        weight_sum = float(m_w["m00"])
        if weight_sum > 1e-6:
            weighted_cx = x + float(m_w["m10"] / weight_sum)
            weighted_cy = y + float(m_w["m01"] / weight_sum)
        else:
            weighted_cx, weighted_cy = shape_cx, shape_cy

        ai_score = beacon_ai_score(
            [area_ratio, aspect, brightness, fill_ratio, circularity]
        )
        signature_score = _signature_score(roi, preferred_signature)
        shape_name = _shape_name(contour)

        signature_match = (
            preferred_signature is None or signature_score >= 0.55
        )
        if not signature_match:
            # Keep the candidate available for GUI decoy visualisation, but
            # let detect_beacon enforce the identity gate.
            pass

        shape_score = 1.0 / (1.0 + abs(area_ratio - 1.0))
        aspect_score = 1.0 / (1.0 + abs(aspect - 1.0))
        fill_score = min(1.0, fill_ratio / 0.65)
        total_score = (
            0.24 * shape_score
            + 0.12 * aspect_score
            + 0.10 * brightness
            + 0.09 * fill_score
            + 0.15 * ai_score
            + 0.30 * signature_score
        )

        if expected_position is not None:
            ex, ey = expected_position
            distance_px = math.hypot(weighted_cx - ex, weighted_cy - ey)
            proximity_score = math.exp(-distance_px / 35.0)
            total_score = 0.80 * total_score + 0.20 * proximity_score

        margin_penalty = 1.0
        if (
            x <= 0 or y <= 0
            or x + width >= width_total
            or y + height >= height_total
        ):
            margin_penalty = 0.92
        total_score *= margin_penalty

        candidates.append({
            "score": float(np.clip(total_score, 0.0, 1.0)),
            "x": float(weighted_cx),
            "y": float(weighted_cy),
            "bbox": (int(x), int(y), int(width), int(height)),
            "area": area,
            "ai_score": float(ai_score),
            "signature_score": float(signature_score),
            "shape": shape_name,
            "brightness": brightness,
            "aspect": aspect,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def detect_candidates(frame, return_details=False, expected_position=None, preferred_signature=None):
    """Return all plausible beacon candidates, with a fog local-contrast fallback."""
    if frame is None or frame.size == 0:
        empty = []
        details = {"candidate_count": 0, "threshold": 0, "mode": "none"}
        return (empty, details) if return_details else empty

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    expected_area = float(max(config.BEACON_SIZE, 1) ** 2)

    # Existing fast path. Keep this unchanged for normal conditions.
    if preferred_signature is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        colour_mask = (sat >= 55) & (val >= 28)
        bch, gch, rch = cv2.split(frame)

        if preferred_signature == "GREEN-SQUARE":
            colour_mask &= (
                (gch.astype(np.int16) > rch.astype(np.int16) + 20)
                & (gch.astype(np.int16) > bch.astype(np.int16) + 8)
            )
        elif preferred_signature == "AMBER-DIAMOND":
            colour_mask &= (
                (rch.astype(np.int16) > bch.astype(np.int16) + 45)
                & (gch.astype(np.int16) > bch.astype(np.int16) + 35)
            )
        elif preferred_signature == "MAGENTA-CIRCLE":
            colour_mask &= (
                (bch.astype(np.int16) > gch.astype(np.int16) + 35)
                & (rch.astype(np.int16) > gch.astype(np.int16) + 45)
            )
        elif preferred_signature == "CYAN-TRIANGLE":
            colour_mask &= (
                (bch.astype(np.int16) > rch.astype(np.int16) + 55)
                & (gch.astype(np.int16) > rch.astype(np.int16) + 35)
            )
        elif preferred_signature == "RED-CIRCLE":
            colour_mask &= (
                (rch.astype(np.int16) > gch.astype(np.int16) + 45)
                & (rch.astype(np.int16) > bch.astype(np.int16) + 35)
            )

        wanted = _SIGNATURE_HUE_RANGES.get(preferred_signature)
        if preferred_signature == "RED-CIRCLE":
            colour_mask &= (hue <= 10) | (hue >= 170)
        elif wanted is not None:
            colour_mask &= (hue >= wanted[0]) & (hue <= wanted[1])
        else:
            colour_mask[:] = False

        cleaned = cv2.morphologyEx(
            colour_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        threshold_value = 0
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered = cv2.GaussianBlur(gray, (3, 3), 0.65)
        otsu_value, _ = cv2.threshold(
            filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_value = int(np.clip(max(45.0, otsu_value), 45.0, 215.0))
        _, cleaned = cv2.threshold(
            filtered, threshold_value, 255, cv2.THRESH_BINARY
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    candidates = _extract_candidates(
        cleaned, frame, threshold_value,
        expected_position=expected_position,
        preferred_signature=preferred_signature,
    )
    mode = "global" if preferred_signature is None else "signature"

    # Fog fallback: use local contrast both when the global stage finds
    # nothing and when every global candidate is far outside the track gate.
    # The latter case is important: a bright fog/edge blob can otherwise make
    # the detector believe it has a valid candidate while the real beacon is
    # still locally visible near the predicted position.
    need_local_fallback = not candidates
    if candidates and expected_position is not None:
        ex, ey = expected_position
        gate_radius = max(80.0, float(config.BEACON_SIZE) * 8.0)
        need_local_fallback = not any(
            math.hypot(c["x"] - ex, c["y"] - ey) <= gate_radius
            for c in candidates
        )

    if need_local_fallback:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        local_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        local_contrast = cv2.subtract(gray, local_bg)

        top_hat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (17, 17)
        )
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, top_hat_kernel)
        contrast_map = cv2.max(top_hat, local_contrast)

        peak = float(np.percentile(contrast_map, 99.2))
        fallback_threshold = max(6.0, min(42.0, peak * 0.55))
        _, fallback_binary = cv2.threshold(
            contrast_map,
            fallback_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        fallback_binary = cv2.morphologyEx(
            fallback_binary, cv2.MORPH_OPEN, kernel
        )
        fallback_binary = cv2.morphologyEx(
            fallback_binary, cv2.MORPH_CLOSE, kernel
        )

        fallback_candidates = _extract_candidates(
            fallback_binary, frame, threshold_value,
            expected_position=expected_position,
            preferred_signature=preferred_signature,
        )
        if fallback_candidates:
            # Local-contrast blobs are more susceptible to tiny noise. Keep
            # the learned/shape score primary; brightness is only a tiebreaker.
            for item in fallback_candidates:
                item["score"] = float(
                    0.72 * item["score"]
                    + 0.28 * min(1.0, item["brightness"] * 1.30)
                )
            fallback_candidates.sort(key=lambda item: item["score"], reverse=True)

            if not candidates:
                candidates = fallback_candidates
                mode = "fog-local-contrast"
            elif expected_position is not None:
                ex, ey = expected_position
                gate_radius = max(80.0, float(config.BEACON_SIZE) * 8.0)
                nearby_fallback = [
                    c for c in fallback_candidates
                    if math.hypot(c["x"] - ex, c["y"] - ey) <= gate_radius
                ]
                if nearby_fallback:
                    candidates = nearby_fallback
                    mode = "fog-local-contrast"

    if not return_details:
        return candidates
    return candidates, {
        "candidate_count": len(candidates),
        "threshold": threshold_value,
        "mode": mode,
    }


def detect_beacon(frame, return_details=False, expected_position=None, preferred_signature=None):
    candidates, pipeline = detect_candidates(
        frame,
        return_details=True,
        expected_position=expected_position,
        preferred_signature=preferred_signature,
    )
    if not candidates:
        details = {
            "confidence": 0.0,
            "ai_score": 0.0,
            "signature_score": 0.0,
            "candidate_count": 0,
            "threshold": pipeline.get("threshold", 0),
            "mode": pipeline.get("mode", "global"),
            "candidates": [],
        }
        return (None, details) if return_details else None

    if preferred_signature is not None:
        eligible = [c for c in candidates if c["signature_score"] >= 0.55]
    else:
        eligible = candidates

    # During an established track, use a spatial gate around the predicted
    # position. In heavy fog the global threshold can produce bright edge/star
    # blobs; selecting one hundreds of pixels away would cause a catastrophic
    # track jump. The gate is deliberately generous for the PS motion limits.
    if eligible and expected_position is not None:
        ex, ey = expected_position
        gate_radius = max(80.0, float(config.BEACON_SIZE) * 8.0)
        nearby = [
            c for c in eligible
            if math.hypot(c["x"] - ex, c["y"] - ey) <= gate_radius
        ]
        if nearby:
            eligible = nearby
        else:
            # A distant candidate is more dangerous than a temporary miss:
            # reject it while the tracker has a valid prediction. This prevents
            # fog/atmosphere blobs from causing a hundreds-of-pixels track jump.
            eligible = []

    if not eligible:
        details = {
            "confidence": 0.0,
            "ai_score": 0.0,
            "signature_score": 0.0,
            "candidate_count": len(candidates),
            "threshold": pipeline.get("threshold", 0),
            "mode": pipeline.get("mode", "global"),
            "bbox": None,
            "shape": None,
            "candidates": candidates,
            "identity_rejected": True,
        }
        return (None, details) if return_details else None

    eligible.sort(key=lambda item: item["score"], reverse=True)
    best = eligible[0]
    result = (int(round(best["x"])), int(round(best["y"])))
    details = {
        "confidence": float(best["score"]),
        "ai_score": float(best["ai_score"]),
        "signature_score": float(best["signature_score"]),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "threshold": pipeline.get("threshold", 0),
        "mode": pipeline.get("mode", "global"),
        "bbox": best["bbox"],
        "shape": best["shape"],
        "candidates": candidates,
        "identity_rejected": False,
    }
    return (result, details) if return_details else result
