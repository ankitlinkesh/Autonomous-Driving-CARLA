from config.config import ControlConfig
from src.decision_module import DecisionModule
from src.object_detection import Detection


def test_pedestrian_causes_braking() -> None:
    module = DecisionModule(ControlConfig())
    pedestrian = Detection((500, 400, 700, 650), "person", 0.9, 0)
    decision = module.decide(0.0, [pedestrian], 1280)
    assert decision.brake == 1.0
    assert decision.target_speed_kmh == 0.0

