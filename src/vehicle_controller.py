"""Controller composition, backend abstraction, and structured diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from config.config import ControlConfig
from src.brake_controller import BrakeController
from src.steering_controller import SteeringController
from src.throttle_controller import ThrottleController


@dataclass(frozen=True)
class ControllerDiagnostics:
    """Structured actuator outputs and controller status."""

    steering: float
    throttle: float
    brake: float
    status: str
    driving_state: str
    target_speed_kmh: float


@dataclass(frozen=True)
class VehicleCommand:
    """Normalized command sent to the active vehicle backend."""

    steering: float
    throttle: float
    brake: float
    diagnostics: ControllerDiagnostics | None = field(default=None, compare=False)


class VehicleBackend(Protocol):
    """Backend contract implemented by print, CARLA, or hardware adapters."""

    def apply(self, command: VehicleCommand) -> None:
        ...


class PrintBackend:
    """Safe development backend that emits commands without controlling a vehicle."""

    def apply(self, command: VehicleCommand) -> None:
        print(f"command steering={command.steering:.3f} throttle={command.throttle:.3f} brake={command.brake:.3f}")


class VehicleController:
    """Compose independent steering, throttle, and brake controllers."""

    def __init__(self, backend: VehicleBackend | None = None, config: ControlConfig | None = None, steering_controller: SteeringController | None = None, throttle_controller: ThrottleController | None = None, brake_controller: BrakeController | None = None) -> None:
        self.backend = backend or PrintBackend()
        self.config = config or ControlConfig()
        self.steering_controller = steering_controller or SteeringController(self.config.steering_smoothing_factor, self.config.steering_rate_limit, self.config.controller_lane_offset_gain, self.config.controller_min_lane_confidence)
        self.throttle_controller = throttle_controller or ThrottleController(self.config.controller_max_speed_kmh, self.config.throttle_smoothing_factor, self.config.throttle_increase_rate, self.config.throttle_decrease_rate, self.config.controller_state_speed_scale)
        self.brake_controller = brake_controller or BrakeController(self.config.brake_smoothing_factor, self.config.brake_rate_limit, self.config.brake_release_rate_limit)
        self.last_diagnostics: ControllerDiagnostics | None = None

    @property
    def diagnostics(self) -> ControllerDiagnostics | None:
        """Expose the most recent structured controller diagnostics."""
        return self.last_diagnostics

    def send(self, steering: float, target_speed_kmh: float, brake: float, max_speed_kmh: float = 50.0, driving_state: str = "Cruise", lane_offset: float | None = None, lane_confidence: float | None = None) -> VehicleCommand:
        """Generate and apply a command while retaining the legacy call shape."""
        if max_speed_kmh == 50.0 and self.config.controller_max_speed_kmh != 50.0:
            max_speed_kmh = self.config.controller_max_speed_kmh
            self.throttle_controller.max_speed_kmh = max(1e-6, max_speed_kmh)
        brake_command = self.brake_controller.update(brake, driving_state)
        steering_command = self.steering_controller.update(steering, lane_offset, lane_confidence, driving_state)
        throttle_command = self.throttle_controller.update(target_speed_kmh, brake_command, driving_state)
        status = _controller_status(driving_state, throttle_command, brake_command)
        diagnostics = ControllerDiagnostics(steering_command, throttle_command, brake_command, status, driving_state, target_speed_kmh)
        command = VehicleCommand(steering_command, throttle_command, brake_command, diagnostics)
        self.last_diagnostics = diagnostics
        self.backend.apply(command)
        return command


class CarlaBackend:
    """Future adapter boundary. No CARLA behavior is faked when the API is absent."""

    def __init__(self, vehicle: object) -> None:
        self.vehicle = vehicle

    def apply(self, command: VehicleCommand) -> None:
        try:
            import carla  # type: ignore
        except ImportError as exc:
            raise RuntimeError("CARLA Python API is unavailable; install it before using CarlaBackend.") from exc
        control = carla.VehicleControl(throttle=command.throttle, steer=command.steering, brake=command.brake)
        self.vehicle.apply_control(control)


def _controller_status(driving_state: str, throttle: float, brake: float) -> str:
    if driving_state == "Emergency Brake" or brake >= 0.99:
        return "EMERGENCY_BRAKE"
    if throttle <= 0.001 and brake <= 0.001:
        return "COASTING"
    return "ACTIVE"
