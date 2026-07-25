"""Explainable rule-based driving decisions."""

from __future__ import annotations

from dataclasses import dataclass

from config.config import ControlConfig
from src.object_detection import Detection


@dataclass(frozen=True)
class DrivingDecision:
    steering: float
    target_speed_kmh: float
    brake: float
    reason: str


class DecisionModule:
    """Combine lane geometry and object detections into a safe command intent."""

    def __init__(self, config: ControlConfig) -> None:
        self.config = config

    def decide(self, lane_offset: float | None, detections: list[Detection], frame_width: int) -> DrivingDecision:
        close_pedestrian = self._has_close_object(detections, {"person"}, self.config.pedestrian_distance_pixels, frame_width)
        close_vehicle = self._has_close_object(detections, {"car", "bus", "truck", "motorcycle"}, self.config.vehicle_distance_pixels, frame_width)
        if close_pedestrian:
            return DrivingDecision(0.0, 0.0, 1.0, "pedestrian detected in stopping zone")
        speed = self.config.reduced_speed_kmh if close_vehicle else self.config.target_speed_kmh
        reason = "vehicle ahead; reduced speed" if close_vehicle else "lane following"
        steering = 0.0 if lane_offset is None else max(-self.config.max_steering, min(self.config.max_steering, lane_offset * self.config.steering_gain))
        return DrivingDecision(steering, speed, 0.0, reason)

    @staticmethod
    def _has_close_object(detections: list[Detection], labels: set[str], distance_pixels: int, frame_width: int) -> bool:
        center_x = frame_width / 2
        for detection in detections:
            if detection.label.lower() not in labels:
                continue
            x1, y1, x2, y2 = detection.box
            box_center = (x1 + x2) / 2
            box_height = y2 - y1
            if abs(box_center - center_x) < frame_width * 0.30 and box_height >= distance_pixels:
                return True
        return False

