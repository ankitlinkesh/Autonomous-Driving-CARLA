"""Deterministic modular driving-behaviour state machine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from config.config import ControlConfig
from src.object_detection import Detection


class DrivingState(str, Enum):
    """Supported high-level driving behaviours."""

    CRUISE = "Cruise"
    LANE_CENTERING = "Lane Centering"
    FOLLOW_VEHICLE = "Follow Vehicle"
    SLOW_DOWN = "Slow Down"
    EMERGENCY_BRAKE = "Emergency Brake"
    RECOVERY = "Recovery"


@dataclass(frozen=True)
class DrivingDecision:
    """Stable downstream decision contract plus state-machine observability."""

    # Preserve the original positional fields and order.
    steering: float
    target_speed_kmh: float
    brake: float
    reason: str
    current_state: str = DrivingState.CRUISE.value
    transition: str = ""


@dataclass(frozen=True)
class _Hazards:
    emergency: bool
    medium_risk: bool
    follow_vehicle: bool
    lane_centering: bool
    follow_distance_m: float | None = None


class DecisionModule:
    """Convert lane/object observations into deterministic driving behaviour."""

    def __init__(self, config: ControlConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger
        self.state = DrivingState.CRUISE
        self._recovery_frames = 0
        self._previous_speed_kmh: float | None = None

    @property
    def current_state(self) -> str:
        """Return the current state label for overlays and external observers."""
        return self.state.value

    def decide(
        self,
        lane_offset: float | None,
        detections: list[Detection],
        frame_width: int,
        lane_confidence: float | None = None,
    ) -> DrivingDecision:
        """Advance the state machine and return steering, speed, and brake intent.

        ``lane_confidence`` is optional to preserve callers using the original
        three-input interface. When omitted, lane geometry is treated as usable.
        """
        hazards = self._evaluate_hazards(lane_offset, lane_confidence, detections, frame_width)
        next_state = self._next_state(hazards)
        previous_state = self.state
        self.state = next_state
        transition = ""
        if next_state != previous_state:
            transition = f"{previous_state.value} -> {next_state.value}"
            if self.config.state_logging_enabled and self.logger is not None:
                self.logger.info("Driving state transition: %s", transition)

        steering = self._steering(lane_offset, lane_confidence)
        if next_state == DrivingState.EMERGENCY_BRAKE:
            steering = 0.0
        speed, brake, reason = self._command_for_state(next_state, steering, hazards)
        return DrivingDecision(
            steering=steering,
            target_speed_kmh=speed,
            brake=brake,
            reason=reason,
            current_state=next_state.value,
            transition=transition,
        )

    def _evaluate_hazards(
        self,
        lane_offset: float | None,
        lane_confidence: float | None,
        detections: list[Detection],
        frame_width: int,
    ) -> _Hazards:
        emergency = any(self._is_emergency_detection(detection, frame_width) for detection in detections)
        medium_risk = any(str(detection.collision_risk).upper() == "MEDIUM" for detection in detections)
        follow_distances = [
            detection.estimated_distance
            for detection in detections
            if self._is_vehicle_ahead(detection, frame_width)
            and detection.estimated_distance is not None
            and detection.estimated_distance <= self.config.follow_vehicle_distance_m
        ]
        follow_vehicle = bool(follow_distances)
        if not follow_vehicle:
            follow_vehicle = self._has_close_legacy_vehicle(detections, frame_width)
        confidence_usable = lane_confidence is None or lane_confidence >= self.config.minimum_lane_confidence
        lane_centering = (
            lane_offset is not None
            and abs(lane_offset) > self.config.lane_offset_threshold_pixels
            and confidence_usable
        )
        return _Hazards(
            emergency=emergency,
            medium_risk=medium_risk,
            follow_vehicle=follow_vehicle,
            lane_centering=lane_centering,
            follow_distance_m=min(follow_distances) if follow_distances else None,
        )

    def _next_state(self, hazards: _Hazards) -> DrivingState:
        if hazards.emergency:
            self._recovery_frames = 0
            return DrivingState.EMERGENCY_BRAKE
        if hazards.medium_risk:
            self._recovery_frames = 0
            return DrivingState.SLOW_DOWN
        if hazards.follow_vehicle:
            self._recovery_frames = 0
            return DrivingState.FOLLOW_VEHICLE
        if self.state in {
            DrivingState.EMERGENCY_BRAKE,
            DrivingState.SLOW_DOWN,
            DrivingState.FOLLOW_VEHICLE,
        }:
            self._recovery_frames = 1
            return DrivingState.RECOVERY
        if self.state == DrivingState.RECOVERY:
            self._recovery_frames += 1
            if self._recovery_frames < self.config.recovery_hold_frames:
                return DrivingState.RECOVERY
        if hazards.lane_centering:
            return DrivingState.LANE_CENTERING
        return DrivingState.CRUISE

    def _command_for_state(
        self,
        state: DrivingState,
        steering: float,
        hazards: _Hazards,
    ) -> tuple[float, float, str]:
        if state == DrivingState.EMERGENCY_BRAKE:
            self._previous_speed_kmh = 0.0
            return 0.0, 1.0, "emergency hazard detected"
        if state == DrivingState.FOLLOW_VEHICLE:
            target = self.config.follow_speed_kmh
            if hazards.follow_distance_m is not None:
                target += (hazards.follow_distance_m - self.config.follow_distance_target_m) * self.config.follow_speed_gain
            target = max(0.0, min(self.config.target_speed_kmh, target))
            return self._smooth_speed(target), 0.0, "following vehicle at safe distance"
        if state == DrivingState.SLOW_DOWN:
            return self._smooth_speed(self.config.slow_down_speed_kmh), 0.0, "medium collision risk; slowing down"
        if state == DrivingState.RECOVERY:
            return self._smooth_speed(self.config.recovery_speed_kmh), 0.0, "recovering after hazard"
        if state == DrivingState.LANE_CENTERING:
            return self._smooth_speed(self.config.lane_centering_speed_kmh), 0.0, "lane offset above threshold"
        return self._smooth_speed(self.config.target_speed_kmh), 0.0, "cruising in lane"

    def _smooth_speed(self, target: float) -> float:
        alpha = max(0.0, min(1.0, self.config.speed_smoothing_factor))
        if self._previous_speed_kmh is None:
            speed = target
        else:
            speed = self._previous_speed_kmh + alpha * (target - self._previous_speed_kmh)
        self._previous_speed_kmh = speed
        return round(speed, 3)

    def _steering(self, lane_offset: float | None, lane_confidence: float | None) -> float:
        if lane_offset is None:
            return 0.0
        if lane_confidence is not None and lane_confidence < self.config.minimum_lane_confidence:
            return 0.0
        return max(-self.config.max_steering, min(self.config.max_steering, lane_offset * self.config.steering_gain))

    def _is_emergency_detection(self, detection: Detection, frame_width: int) -> bool:
        if str(detection.collision_risk).upper() == "HIGH":
            return True
        label = detection.label.lower()
        if label not in {item.lower() for item in self.config.emergency_classes} or not self._is_ahead(detection, frame_width):
            return False
        distance = detection.estimated_distance
        if label == "person":
            threshold = self.config.emergency_pedestrian_distance_m
        else:
            threshold = self.config.emergency_obstacle_distance_m
        if distance is not None:
            return distance <= threshold
        return detection.box[3] - detection.box[1] >= self.config.pedestrian_distance_pixels if label == "person" else detection.box[3] - detection.box[1] >= self.config.vehicle_distance_pixels

    def _is_vehicle_ahead(self, detection: Detection, frame_width: int) -> bool:
        return detection.label.lower() in {item.lower() for item in self.config.vehicle_classes} and self._is_ahead(detection, frame_width)

    @staticmethod
    def _is_ahead(detection: Detection, frame_width: int) -> bool:
        if detection.relative_position.lower() == "centre":
            return True
        center_x = (detection.box[0] + detection.box[2]) / 2
        return abs(center_x - frame_width / 2) < frame_width * 0.30

    def _has_close_legacy_vehicle(self, detections: list[Detection], frame_width: int) -> bool:
        return any(
            self._is_vehicle_ahead(detection, frame_width)
            and detection.estimated_distance is None
            and detection.box[3] - detection.box[1] >= self.config.vehicle_distance_pixels
            for detection in detections
        )
