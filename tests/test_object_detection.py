from config.config import DetectionConfig
from src.object_detection import Detection, YOLOv8Detector


def _detector() -> YOLOv8Detector:
    detector = YOLOv8Detector.__new__(YOLOv8Detector)
    detector.config = DetectionConfig()
    detector.target_classes = set()
    detector._tracks = []
    detector._next_object_id = 1
    return detector


def test_tracking_metadata_and_id_stability() -> None:
    detector = _detector()
    first = detector._add_tracking_metadata([Detection((500, 300, 620, 600), "person", 0.9, 0)], 1280)
    detector._add_tracking_metadata([], 1280)
    second = detector._add_tracking_metadata([Detection((505, 305, 625, 605), "person", 0.88, 0)], 1280)

    assert first[0].object_id == second[0].object_id
    assert second[0].class_name == "person"
    assert second[0].bounding_box == second[0].box
    assert {"object_id", "class_name", "bounding_box", "estimated_distance", "relative_position", "collision_risk"}.issubset(second[0].to_dict())
    assert second[0].estimated_distance is not None
    assert second[0].relative_position in {"Left", "Centre", "Right"}
    assert second[0].collision_risk in {"LOW", "MEDIUM", "HIGH"}


def test_risk_is_high_for_close_centre_person() -> None:
    detector = _detector()
    result = detector._add_tracking_metadata([Detection((570, 450, 710, 700), "person", 0.99, 0)], 1280)
    assert result[0].relative_position == "Centre"
    assert result[0].collision_risk == "HIGH"
