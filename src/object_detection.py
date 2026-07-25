"""YOLOv8 object detection with lightweight CPU-friendly object tracking.

Distance values in this module are monocular approximations derived from
bounding-box height. They are useful for relative safety rules and overlays,
but are not a substitute for calibrated depth or sensor fusion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # Optional at import time for tests and documentation builds.
    YOLO = None  # type: ignore[assignment,misc]

from config.config import DetectionConfig


@dataclass(frozen=True)
class Detection:
    """A detection retaining legacy fields and exposing structured metadata."""

    # Keep these first and in the original order for positional compatibility.
    box: tuple[int, int, int, int]
    label: str
    confidence: float
    class_id: int
    object_id: int = 0
    estimated_distance: float | None = None
    relative_position: str = "centre"
    collision_risk: str = "LOW"

    @property
    def class_name(self) -> str:
        """Structured alias for the original ``label`` field."""
        return self.label

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """Structured alias for the original ``box`` field."""
        return self.box

    def to_dict(self) -> dict[str, object]:
        """Serialize both the new structured schema and legacy aliases."""
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "estimated_distance": self.estimated_distance,
            "relative_position": self.relative_position,
            "collision_risk": self.collision_risk,
            "box": self.box,
            "label": self.label,
            "class_id": self.class_id,
        }


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]
    inference_time_ms: float
    fps: float


@dataclass
class _Track:
    object_id: int
    box: tuple[int, int, int, int]
    label: str
    distance: float | None = None
    missed_frames: int = 0


class YOLOv8Detector:
    """Reusable YOLOv8 detector with persistent IDs and safety metadata."""

    def __init__(self, config: DetectionConfig) -> None:
        if YOLO is None:
            raise ImportError("Ultralytics is not installed. Install requirements.txt first.")
        model_path = Path(config.model_path)
        self.model = YOLO(str(model_path))
        self.config = config
        self.target_classes = {name.lower() for name in config.target_classes}
        self._tracks: list[_Track] = []
        self._next_object_id = 1

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run YOLO inference and return tracked, structured detections."""
        start = time.perf_counter()
        kwargs = {
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "verbose": False,
        }
        if self.config.device != "auto":
            kwargs["device"] = self.config.device
        result = self.model.predict(image, **kwargs)[0]
        detections = self._decode_detections(result)
        detections = self._add_tracking_metadata(detections, image.shape[1])
        elapsed = time.perf_counter() - start
        return DetectionResult(detections, elapsed * 1000, 1.0 / elapsed if elapsed else 0.0)

    def _decode_detections(self, result: object) -> list[Detection]:
        boxes = result.boxes  # type: ignore[attr-defined]
        detections: list[Detection] = []
        names = result.names  # type: ignore[attr-defined]
        for box, confidence, class_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
        ):
            class_index = int(class_id)
            label = str(names[class_index])
            if self.target_classes and label.lower() not in self.target_classes:
                continue
            detections.append(Detection(tuple(map(int, box)), label, float(confidence), class_index))
        return detections

    def _add_tracking_metadata(self, detections: list[Detection], frame_width: int) -> list[Detection]:
        if not self.config.tracking_enabled:
            return [self._with_metadata(detection, frame_width, 0, None) for detection in detections]

        unmatched_tracks = set(range(len(self._tracks)))
        tracked: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            match_index = self._best_track_match(detection, unmatched_tracks, frame_width)
            if match_index is None:
                track = _Track(self._next_object_id, detection.box, detection.label)
                self._next_object_id += 1
                self._tracks.append(track)
                match_index = len(self._tracks) - 1
            else:
                track = self._tracks[match_index]
                track.box = detection.box
                track.label = detection.label
                track.missed_frames = 0
                unmatched_tracks.discard(match_index)
            tracked.append(self._with_metadata(detection, frame_width, track.object_id, track))

        for index in unmatched_tracks:
            self._tracks[index].missed_frames += 1
        self._tracks = [
            track for track in self._tracks
            if track.missed_frames <= self.config.tracker_max_missed_frames
        ]
        return tracked

    def _best_track_match(
        self,
        detection: Detection,
        available: set[int],
        frame_width: int,
    ) -> int | None:
        best_index: int | None = None
        best_score = 0.0
        for index in available:
            track = self._tracks[index]
            if not self.config.tracker_class_agnostic and track.label.lower() != detection.label.lower():
                continue
            overlap = _iou(track.box, detection.box)
            center_distance = _center_distance(track.box, detection.box) / max(frame_width, 1)
            if overlap < self.config.tracker_iou_threshold and center_distance > self.config.tracker_max_center_distance_ratio:
                continue
            score = overlap + max(0.0, 1.0 - center_distance / max(self.config.tracker_max_center_distance_ratio, 1e-6)) * 0.1
            if score > best_score:
                best_index, best_score = index, score
        return best_index

    def _with_metadata(
        self,
        detection: Detection,
        frame_width: int,
        object_id: int,
        track: _Track | None,
    ) -> Detection:
        distance = self._estimate_distance(detection, track)
        if track is not None:
            track.distance = distance
        return Detection(
            detection.box,
            detection.label,
            detection.confidence,
            detection.class_id,
            object_id,
            distance,
            self._relative_position(detection.box, frame_width),
            self._collision_risk(detection.label, distance, self._relative_position(detection.box, frame_width)),
        )

    def _estimate_distance(self, detection: Detection, track: _Track | None) -> float | None:
        if not self.config.distance_enabled:
            return None
        pixel_height = max(1, detection.box[3] - detection.box[1])
        object_height = self.config.distance_reference_heights_m.get(
            detection.label.lower(), self.config.distance_default_object_height_m
        )
        raw_distance = self.config.distance_focal_length_pixels * object_height / pixel_height
        raw_distance = float(np.clip(raw_distance, self.config.distance_min_m, self.config.distance_max_m))
        if track is None or track.distance is None:
            return round(raw_distance, 2)
        alpha = float(np.clip(self.config.distance_smoothing_factor, 0.0, 1.0))
        return round(alpha * raw_distance + (1.0 - alpha) * track.distance, 2)

    def _relative_position(self, box: tuple[int, int, int, int], frame_width: int) -> str:
        center_ratio = ((box[0] + box[2]) / 2) / max(frame_width, 1)
        if center_ratio < self.config.relative_position_left_ratio:
            return "Left"
        if center_ratio > self.config.relative_position_right_ratio:
            return "Right"
        return "Centre"

    def _collision_risk(self, label: str, distance: float | None, position: str) -> str:
        if distance is None or position.lower() not in {item.lower() for item in self.config.collision_relevant_positions}:
            return "LOW"
        normalized_label = label.lower()
        high_classes = {item.lower() for item in self.config.collision_high_classes}
        medium_classes = {item.lower() for item in self.config.collision_medium_classes}
        if normalized_label in high_classes and position.lower() == "centre" and distance <= self.config.collision_high_distance_m:
            return "HIGH"
        if normalized_label in medium_classes and distance <= self.config.collision_medium_distance_m:
            return "MEDIUM"
        return "LOW"


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _center_distance(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    return float(np.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))
