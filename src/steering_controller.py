"""Independent steering computation, smoothing, and rate limiting."""

from __future__ import annotations


class SteeringController:
    """Compute bounded steering from decision and lane context."""

    def __init__(self, smoothing_factor: float = 0.35, rate_limit: float = 0.12, lane_offset_gain: float = 0.003, min_lane_confidence: float = 0.25) -> None:
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self.rate_limit = max(0.0, rate_limit)
        self.lane_offset_gain = lane_offset_gain
        self.min_lane_confidence = min_lane_confidence
        self._previous = 0.0

    def update(self, requested_steering: float, lane_offset: float | None = None, lane_confidence: float | None = None, driving_state: str = "Cruise") -> float:
        """Return smoothed steering using lane offset during lane centering."""
        if driving_state == "Emergency Brake":
            self._previous = 0.0
            return 0.0
        raw = requested_steering
        if lane_offset is not None and (lane_confidence is None or lane_confidence >= self.min_lane_confidence):
            if driving_state == "Lane Centering":
                raw = lane_offset * self.lane_offset_gain
        elif lane_confidence is not None and lane_confidence < self.min_lane_confidence:
            raw = 0.0
        raw = _clamp(raw, -1.0, 1.0)
        smoothed = self._previous + self.smoothing_factor * (raw - self._previous)
        value = _rate_limit(smoothed, self._previous, self.rate_limit)
        self._previous = value
        return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _rate_limit(desired: float, previous: float, limit: float) -> float:
    return _clamp(max(previous - limit, min(previous + limit, desired)), -1.0, 1.0)
