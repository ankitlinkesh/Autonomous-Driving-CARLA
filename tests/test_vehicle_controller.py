from config.config import ControlConfig
from src.vehicle_controller import (
    BrakeController,
    ControllerDiagnostics,
    SteeringController,
    ThrottleController,
    VehicleCommand,
    VehicleController,
)


def test_steering_smoothing_and_rate_limit() -> None:
    controller = SteeringController(smoothing_factor=1.0, rate_limit=0.1)
    first = controller.update(1.0)
    second = controller.update(1.0)
    assert first == 0.1
    assert second == 0.2
    assert -1.0 <= second <= 1.0


def test_throttle_increase_is_rate_limited() -> None:
    controller = ThrottleController(max_speed_kmh=50.0, smoothing_factor=1.0, increase_rate=0.1, decrease_rate=0.2)
    first = controller.update(50.0, 0.0, "Cruise")
    second = controller.update(50.0, 0.0, "Cruise")
    assert first == 0.1
    assert second == 0.2


def test_emergency_brake_is_immediate_and_throttle_is_zero() -> None:
    brake = BrakeController(smoothing_factor=0.1, rate_limit=0.1)
    assert brake.update(1.0, "Emergency Brake") == 1.0
    throttle = ThrottleController()
    assert throttle.update(30.0, 1.0, "Emergency Brake") == 0.0


def test_vehicle_controller_outputs_diagnostics_and_limits() -> None:
    class Backend:
        def __init__(self) -> None:
            self.command = None

        def apply(self, command: VehicleCommand) -> None:
            self.command = command

    backend = Backend()
    controller = VehicleController(backend=backend, config=ControlConfig())
    command = controller.send(4.0, 100.0, 0.0, driving_state="Cruise")

    assert backend.command == command
    assert isinstance(command.diagnostics, ControllerDiagnostics)
    assert command.diagnostics.status == "ACTIVE"
    assert -1.0 <= command.steering <= 1.0
    assert 0.0 <= command.throttle <= 1.0
    assert 0.0 <= command.brake <= 1.0

    emergency = controller.send(0.0, 30.0, 1.0, driving_state="Emergency Brake")
    assert emergency.steering == 0.0
    assert emergency.throttle == 0.0
    assert emergency.brake == 1.0
    assert emergency.diagnostics.status == "EMERGENCY_BRAKE"
