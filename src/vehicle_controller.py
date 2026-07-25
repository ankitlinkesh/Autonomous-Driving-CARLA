"""Vehicle command abstractions; CARLA integration is intentionally optional."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VehicleCommand:
    steering: float
    throttle: float
    brake: float


class VehicleBackend(Protocol):
    """Backend contract implemented by print, CARLA, or hardware adapters."""

    def apply(self, command: VehicleCommand) -> None:
        ...


class PrintBackend:
    """Safe development backend that emits commands without controlling a vehicle."""

    def apply(self, command: VehicleCommand) -> None:
        print(f"command steering={command.steering:.3f} throttle={command.throttle:.3f} brake={command.brake:.3f}")


class VehicleController:
    """Translate a driving decision into normalized actuator values."""

    def __init__(self, backend: VehicleBackend | None = None) -> None:
        self.backend = backend or PrintBackend()

    def send(self, steering: float, target_speed_kmh: float, brake: float, max_speed_kmh: float = 50.0) -> VehicleCommand:
        command = VehicleCommand(float(max(-1.0, min(1.0, steering))),
                                 0.0 if brake > 0 else float(max(0.0, min(1.0, target_speed_kmh / max_speed_kmh))),
                                 float(max(0.0, min(1.0, brake))))
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

