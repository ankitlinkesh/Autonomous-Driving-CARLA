"""Independent speed-to-throttle conversion and rate limiting."""

from __future__ import annotations


class ThrottleController:
    """Convert target speed into a bounded, smoothed throttle command."""

    def __init__(self, max_speed_kmh: float = 50.0, smoothing_factor: float = 0.35, increase_rate: float = 0.08, decrease_rate: float = 0.15, state_speed_scale: dict[str, float] | None = None) -> None:
        self.max_speed_kmh = max(1e-6, max_speed_kmh)
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self.increase_rate = max(0.0, increase_rate)
        self.decrease_rate = max(0.0, decrease_rate)
        self.state_speed_scale = dict(state_speed_scale or {})
        self._previous = 0.0

    def update(self, target_speed_kmh: float, brake: float, driving_state: str = "Cruise") -> float:
        """Return throttle in [0, 1], immediately suppressed while braking."""
        scale = self.state_speed_scale.get(driving_state, 1.0)
        desired = max(0.0, target_speed_kmh) * max(0.0, scale) / self.max_speed_kmh
        if brake > 0.0 or driving_state == "Emergency Brake":
            self._previous = 0.0
            return 0.0
        desired = _clamp(desired, 0.0, 1.0)
        smoothed = self._previous + self.smoothing_factor * (desired - self._previous)
        limit = self.increase_rate if smoothed >= self._previous else self.decrease_rate
        value = _rate_limit(smoothed, self._previous, limit)
        self._previous = value
        return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _rate_limit(desired: float, previous: float, limit: float) -> float:
    return _clamp(max(previous - limit, min(previous + limit, desired)), 0.0, 1.0)
