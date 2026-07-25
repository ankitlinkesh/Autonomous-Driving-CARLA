"""Robust classical-computer-vision lane detection.

The public API remains ``LaneDetector(config).detect(image)`` and the original
LaneResult fields are preserved. Additional result fields are optional extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from config.config import LaneConfig


@dataclass(frozen=True)
class LaneResult:
    """Lane geometry and the annotated/debug image for one frame."""

    image: np.ndarray
    lane_center: float | None
    lane_offset: float | None
    lane_angle: float | None
    left_line: tuple[int, int, int, int] | None
    right_line: tuple[int, int, int, int] | None
    lane_confidence: float = 0.0
    debug_image: np.ndarray | None = None


@dataclass
class _SideFit:
    """Internal polynomial fit for one lane boundary."""

    coefficients: np.ndarray
    points: np.ndarray
    segment_count: int
    support: float


class LaneDetector:
    """Detect straight, curved, and dashed lane boundaries with temporal smoothing."""

    def __init__(self, config: LaneConfig) -> None:
        self.config = config
        self._left_coefficients: np.ndarray | None = None
        self._right_coefficients: np.ndarray | None = None
        self._left_missing = 0
        self._right_missing = 0
        self._last_roi: np.ndarray | None = None

    def detect(self, image: np.ndarray) -> LaneResult:
        """Run lane detection and return geometry plus the debug-annotated frame."""
        if image is None or image.size == 0:
            raise ValueError("Lane detection requires a non-empty image")
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = max(3, int(self.config.blur_kernel) | 1)
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)
        if self.config.dashed_line_support:
            close_kernel = max(1, int(self.config.dash_morphology_kernel))
            morphology = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morphology)
        roi_polygon = self._adaptive_roi(width, height)
        self._last_roi = roi_polygon
        masked = self._region_of_interest(edges, roi_polygon)
        lines = cv2.HoughLinesP(
            masked,
            self.config.hough_rho,
            np.deg2rad(self.config.hough_theta_deg),
            self.config.hough_threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )
        left_fit, right_fit = self._fit_sides(lines, width, height)
        left_coefficients = self._smooth_side(left_fit, "left")
        right_coefficients = self._smooth_side(right_fit, "right")
        left_curve = self._curve_points(left_coefficients, width, height)
        right_curve = self._curve_points(right_coefficients, width, height)
        left_line = self._line_endpoints(left_curve)
        right_line = self._line_endpoints(right_curve)
        lane_center, lane_offset, lane_angle = self._metrics(left_curve, right_curve, width, height)
        confidence = self._confidence(left_fit, right_fit, left_curve, right_curve, width, height)
        overlay = self._draw_overlay(
            image,
            roi_polygon,
            lines,
            left_curve,
            right_curve,
            lane_center,
            lane_offset,
            lane_angle,
            confidence,
        )
        return LaneResult(
            overlay,
            lane_center,
            lane_offset,
            lane_angle,
            left_line,
            right_line,
            confidence,
            overlay.copy(),
        )

    def _adaptive_roi(self, width: int, height: int) -> np.ndarray:
        """Build a trapezoid and shift its top toward the tracked vanishing point."""
        top_y = int(height * self.config.roi_top_ratio)
        bottom_y = int(height * self.config.roi_bottom_ratio)
        center_x = width * 0.5
        if self.config.adaptive_roi and self._left_coefficients is not None and self._right_coefficients is not None:
            top_center = float((np.polyval(self._left_coefficients, top_y) + np.polyval(self._right_coefficients, top_y)) / 2)
            adjustment = np.clip((top_center - center_x) * self.config.roi_adaptation_gain, -width * 0.15, width * 0.15)
            center_x += float(adjustment)
        top_half = width * self.config.roi_top_width_ratio * 0.5
        bottom_half = width * self.config.roi_bottom_width_ratio * 0.5
        polygon = np.array(
            [[
                (int(max(0, center_x - bottom_half)), bottom_y),
                (int(min(width - 1, center_x + bottom_half)), bottom_y),
                (int(min(width - 1, center_x + top_half)), top_y),
                (int(max(0, center_x - top_half)), top_y),
            ]],
            dtype=np.int32,
        )
        return polygon

    @staticmethod
    def _region_of_interest(edges: np.ndarray, polygon: np.ndarray | None = None) -> np.ndarray:
        """Mask edges outside the configured/adaptive road trapezoid."""
        if polygon is None:
            height, width = edges.shape
            polygon = np.array([[(0, height), (width, height), (width, int(height * 0.6)), (0, int(height * 0.6))]])
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _fit_sides(self, lines: np.ndarray | None, width: int, height: int) -> tuple[_SideFit | None, _SideFit | None]:
        """Aggregate Hough segments, supporting dashed markings and curved fits."""
        left_points: list[tuple[int, int]] = []
        right_points: list[tuple[int, int]] = []
        left_lengths: list[float] = []
        right_lengths: list[float] = []
        if lines is not None:
            for line in lines[:, 0]:
                x1, y1, x2, y2 = map(int, line)
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 or dy == 0:
                    continue
                slope = dy / dx
                magnitude = abs(slope)
                if magnitude < self.config.min_slope or magnitude > self.config.max_slope:
                    continue
                midpoint_x = (x1 + x2) * 0.5
                if slope < 0 and midpoint_x < width * self.config.side_split_ratio:
                    left_points.extend([(x1, y1), (x2, y2)])
                    left_lengths.append(float(np.hypot(dx, dy)))
                elif slope > 0 and midpoint_x > width * (1.0 - self.config.side_split_ratio):
                    right_points.extend([(x1, y1), (x2, y2)])
                    right_lengths.append(float(np.hypot(dx, dy)))
        return (
            self._fit_side(left_points, left_lengths, height),
            self._fit_side(right_points, right_lengths, height),
        )

    def _fit_side(self, points: list[tuple[int, int]], lengths: list[float], height: int) -> _SideFit | None:
        if not points:
            return None
        values = np.asarray(points, dtype=np.float64)
        y_values = values[:, 1]
        x_values = values[:, 0]
        degree = min(max(1, self.config.curve_degree), len(points) - 1)
        try:
            coefficients = np.polyfit(y_values, x_values, degree, w=np.repeat(np.maximum(lengths, 1.0), 2))
        except (np.linalg.LinAlgError, ValueError):
            return None
        support = float(np.clip((y_values.max() - y_values.min()) / max(height * 0.45, 1.0), 0.0, 1.0))
        return _SideFit(coefficients, values, len(lengths), support)

    def _smooth_side(self, fit: _SideFit | None, side: str) -> np.ndarray | None:
        previous = self._left_coefficients if side == "left" else self._right_coefficients
        missing = self._left_missing if side == "left" else self._right_missing
        if fit is not None:
            coefficients = fit.coefficients
            if self.config.temporal_smoothing and previous is not None and len(previous) == len(coefficients):
                alpha = float(np.clip(self.config.smoothing_factor, 0.0, 1.0))
                coefficients = alpha * coefficients + (1.0 - alpha) * previous
            if side == "left":
                self._left_coefficients, self._left_missing = coefficients, 0
            else:
                self._right_coefficients, self._right_missing = coefficients, 0
            return coefficients
        if previous is not None and missing < self.config.max_missing_frames:
            if side == "left":
                self._left_missing += 1
            else:
                self._right_missing += 1
            return previous
        if side == "left":
            self._left_coefficients, self._left_missing = None, missing + 1
        else:
            self._right_coefficients, self._right_missing = None, missing + 1
        return None

    def _curve_points(self, coefficients: np.ndarray | None, width: int, height: int) -> np.ndarray | None:
        if coefficients is None:
            return None
        y_top = int(height * self.config.roi_top_ratio)
        y_bottom = int(height * self.config.roi_bottom_ratio)
        y_values = np.linspace(y_bottom, y_top, max(2, self.config.curve_samples))
        x_values = np.clip(np.polyval(coefficients, y_values), 0, width - 1)
        return np.column_stack((x_values, y_values)).astype(np.int32)

    @staticmethod
    def _line_endpoints(curve: np.ndarray | None) -> tuple[int, int, int, int] | None:
        if curve is None or len(curve) < 2:
            return None
        bottom = curve[0]
        top = curve[-1]
        return int(bottom[0]), int(bottom[1]), int(top[0]), int(top[1])

    @staticmethod
    def _metrics(left: np.ndarray | None, right: np.ndarray | None, width: int, height: int) -> tuple[float | None, float | None, float | None]:
        if left is None or right is None:
            return None, None, None
        lane_center = float((left[0, 0] + right[0, 0]) * 0.5)
        image_center = width * 0.5
        offset = image_center - lane_center
        lane_center_top = float((left[-1, 0] + right[-1, 0]) * 0.5)
        angle = float(np.degrees(np.arctan2(height - left[-1, 1], lane_center_top - lane_center)))
        return lane_center, float(offset), angle

    def _confidence(
        self,
        left_fit: _SideFit | None,
        right_fit: _SideFit | None,
        left_curve: np.ndarray | None,
        right_curve: np.ndarray | None,
        width: int,
        height: int,
    ) -> float:
        if left_curve is None and right_curve is None:
            return 0.0
        if left_fit is None or right_fit is None:
            return float(self.config.single_side_confidence)
        segment_score = min(1.0, (left_fit.segment_count + right_fit.segment_count) / max(self.config.confidence_segments_target, 1))
        support_score = (left_fit.support + right_fit.support) * 0.5
        bottom_width = float(right_curve[0, 0] - left_curve[0, 0])
        width_score = float(np.clip((bottom_width - self.config.expected_lane_width_min) / max(self.config.expected_lane_width_max - self.config.expected_lane_width_min, 1.0), 0.0, 1.0))
        slope_left = np.polyval(np.polyder(self._left_coefficients), height * self.config.roi_bottom_ratio) if self._left_coefficients is not None else 0.0
        slope_right = np.polyval(np.polyder(self._right_coefficients), height * self.config.roi_bottom_ratio) if self._right_coefficients is not None else 0.0
        parallel_score = float(np.clip(1.0 - abs(float(slope_left + slope_right)) / max(self.config.parallelism_tolerance, 1e-6), 0.0, 1.0))
        return float(np.clip(0.30 * segment_score + 0.30 * support_score + 0.20 * width_score + 0.20 * parallel_score, 0.0, 1.0))

    def _draw_overlay(
        self,
        image: np.ndarray,
        roi_polygon: np.ndarray,
        raw_lines: np.ndarray | None,
        left_curve: np.ndarray | None,
        right_curve: np.ndarray | None,
        lane_center: float | None,
        lane_offset: float | None,
        lane_angle: float | None,
        confidence: float,
    ) -> np.ndarray:
        """Render lane curves, ROI, raw segments, and diagnostic metrics."""
        output = image.copy()
        if not self.config.debug_visualization:
            return output
        if self.config.debug_draw_roi:
            cv2.polylines(output, roi_polygon, True, (255, 120, 0), 2)
        if self.config.debug_draw_raw_lines and raw_lines is not None:
            for line in raw_lines[:, 0]:
                x1, y1, x2, y2 = map(int, line)
                cv2.line(output, (x1, y1), (x2, y2), (80, 80, 255), 1)
        lane_layer = output.copy()
        for curve, color in ((left_curve, (0, 255, 0)), (right_curve, (0, 255, 0))):
            if curve is not None and len(curve) > 1:
                cv2.polylines(lane_layer, [curve.reshape(-1, 1, 2)], False, color, self.config.debug_line_thickness)
        if left_curve is not None and right_curve is not None:
            polygon = np.vstack((left_curve, right_curve[::-1])).reshape(-1, 1, 2)
            cv2.fillPoly(lane_layer, [polygon], (0, 100, 0))
        alpha = float(np.clip(self.config.debug_overlay_alpha, 0.0, 1.0))
        output = cv2.addWeighted(lane_layer, alpha, output, 1.0 - alpha, 0.0)
        metrics = f"lane conf: {confidence:.2f}"
        if lane_offset is not None:
            metrics += f" | offset: {lane_offset:.1f}px"
        if lane_angle is not None:
            metrics += f" | angle: {lane_angle:.1f} deg"
        cv2.putText(output, metrics, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return output

    @staticmethod
    def _average_lines(lines: np.ndarray | None, width: int, height: int) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        """Compatibility helper retaining the original line-averaging API."""
        candidates: list[tuple[float, float, float]] = []
        if lines is not None:
            for line in lines[:, 0]:
                x1, y1, x2, y2 = map(int, line)
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.35:
                    continue
                intercept = y1 - slope * x1
                candidates.append((slope, intercept, abs(x2 - x1)))
        result: list[tuple[int, int, int, int] | None] = []
        for items in ([item for item in candidates if item[0] < 0], [item for item in candidates if item[0] > 0]):
            if not items:
                result.append(None)
                continue
            weights = np.asarray([item[2] for item in items], dtype=np.float64)
            slope = float(np.average([item[0] for item in items], weights=weights))
            intercept = float(np.average([item[1] for item in items], weights=weights))
            y_bottom, y_top = height, int(height * 0.60)
            x_bottom = int((y_bottom - intercept) / slope)
            x_top = int((y_top - intercept) / slope)
            result.append((max(0, min(width - 1, x_bottom)), y_bottom, max(0, min(width - 1, x_top)), y_top))
        return result[0], result[1]