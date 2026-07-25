from config.config import ControlConfig
from src.decision_module import DecisionModule
from src.object_detection import Detection


def test_pedestrian_causes_emergency_braking() -> None:
    module = DecisionModule(ControlConfig())
    pedestrian = Detection((500, 400, 700, 650), "person", 0.9, 0)
    decision = module.decide(0.0, [pedestrian], 1280)
    assert decision.current_state == "Emergency Brake"
    assert decision.brake == 1.0
    assert decision.target_speed_kmh == 0.0


def test_state_machine_transitions_are_deterministic() -> None:
    config = ControlConfig(recovery_hold_frames=2)
    module = DecisionModule(config)

    cruise = module.decide(0.0, [], 1280, lane_confidence=0.9)
    centering = module.decide(100.0, [], 1280, lane_confidence=0.9)
    vehicle = Detection((560, 350, 720, 550), "car", 0.9, 2, 7, 18.0, "Centre", "LOW")
    follow = module.decide(0.0, [vehicle], 1280, lane_confidence=0.9)
    medium_vehicle = Detection((560, 350, 720, 550), "car", 0.9, 2, 7, 10.0, "Centre", "MEDIUM")
    slow = module.decide(0.0, [medium_vehicle], 1280, lane_confidence=0.9)
    high_person = Detection((560, 400, 720, 700), "person", 0.99, 0, 8, 4.0, "Centre", "HIGH")
    emergency = module.decide(0.0, [high_person], 1280, lane_confidence=0.9)
    recovery = module.decide(0.0, [], 1280, lane_confidence=0.9)
    cruise_after_recovery = module.decide(0.0, [], 1280, lane_confidence=0.9)

    assert cruise.current_state == "Cruise"
    assert centering.current_state == "Lane Centering"
    assert vehicle.object_id == 7
    assert follow.current_state == "Follow Vehicle"
    assert slow.current_state == "Slow Down"
    assert emergency.current_state == "Emergency Brake"
    assert recovery.current_state == "Recovery"
    assert cruise_after_recovery.current_state == "Cruise"
    assert emergency.brake == 1.0
