"""Independent brake computation, smoothing, and rate limiting."""

from __future__ import annotations


class BrakeController:
    """Compute bounded brake commands with an immediate emergency override."""

    def __init__(self, smoothing_factor: float = 0.35, rate_limit: float = 0.25, release_rate_limit: float = 0.10) -> None:
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self.rate_limit = max(0.0, rate_limit)
        self.release_rate_limit = max(0.0, release_rate_limit)
        self._previous = 0.0

    def update(self, requested_brake: float, driving_state: str = "Cruise") -> float:
        """Return brake in [0, 1], with immediate full emergency braking."""
        if driving_state == "Emergency Brake":
            self._previous = 1.0
            return 1.0
        desired = _clamp(requested_brake, 0.0, 1.0)
        smoothed = self._previous + self.smoothing_factor * (desired - self._previous)
        limit = self.rate_limit if smoothed >= self._previous else self.release_rate_limit
        value = _rate_limit(smoothed, self._previous, limit)
        self._previous = value
        return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _rate_limit(desired: float, previous: float, limit: float) -> float:
    return _clamp(max(previous - limit, min(previous + limit, desired)), 0.0, 1.0)
