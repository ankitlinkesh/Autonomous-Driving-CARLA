"""Typed configuration loading for the autonomous driving pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InputConfig:
    source: int | str = 0
    image_width: int = 1280
    image_height: int = 720
    display: bool = True
    save_outputs: bool = False


@dataclass(frozen=True)
class DetectionConfig:
    model_path: str = "models/yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "auto"
    target_classes: tuple[str, ...] = field(default_factory=tuple)
    tracking_enabled: bool = True
    tracker_iou_threshold: float = 0.30
    tracker_max_missed_frames: int = 8
    tracker_max_center_distance_ratio: float = 0.20
    tracker_class_agnostic: bool = False
    distance_enabled: bool = True
    distance_focal_length_pixels: float = 700.0
    distance_default_object_height_m: float = 1.7
    distance_min_m: float = 1.0
    distance_max_m: float = 100.0
    distance_smoothing_factor: float = 0.35
    distance_reference_heights_m: dict[str, float] = field(default_factory=lambda: {
        "car": 1.5,
        "bus": 3.2,
        "truck": 3.4,
        "motorcycle": 1.4,
        "person": 1.7,
    })
    relative_position_left_ratio: float = 0.33
    relative_position_right_ratio: float = 0.67
    collision_high_distance_m: float = 8.0
    collision_medium_distance_m: float = 20.0
    collision_relevant_positions: tuple[str, ...] = ("left", "centre", "right")
    collision_high_classes: tuple[str, ...] = ("person", "car", "bus", "truck", "motorcycle")
    collision_medium_classes: tuple[str, ...] = ("person", "car", "bus", "truck", "motorcycle")


@dataclass(frozen=True)
class LaneConfig:
    # Original fields retained in their original order for positional compatibility.
    canny_low: int = 50
    canny_high: int = 150
    blur_kernel: int = 5
    hough_threshold: int = 20
    min_line_length: int = 30
    max_line_gap: int = 100
    roi_top_ratio: float = 0.58
    offset_threshold_pixels: int = 50
    hough_rho: float = 1.0
    hough_theta_deg: float = 1.0
    min_slope: float = 0.35
    max_slope: float = 4.0
    side_split_ratio: float = 0.52
    roi_bottom_ratio: float = 0.98
    roi_top_width_ratio: float = 0.20
    roi_bottom_width_ratio: float = 0.95
    adaptive_roi: bool = True
    roi_adaptation_gain: float = 0.25
    dashed_line_support: bool = True
    dash_morphology_kernel: int = 3
    curve_degree: int = 2
    curve_samples: int = 24
    temporal_smoothing: bool = True
    smoothing_factor: float = 0.35
    max_missing_frames: int = 5
    confidence_segments_target: int = 8
    single_side_confidence: float = 0.25
    expected_lane_width_min: float = 120.0
    expected_lane_width_max: float = 1100.0
    parallelism_tolerance: float = 1.5
    debug_visualization: bool = True
    debug_draw_roi: bool = True
    debug_draw_raw_lines: bool = False
    debug_line_thickness: int = 5
    debug_overlay_alpha: float = 0.30


@dataclass(frozen=True)
class ControlConfig:
    # Legacy fields retained in their original order for positional compatibility.
    target_speed_kmh: float = 30.0
    reduced_speed_kmh: float = 12.0
    pedestrian_distance_pixels: int = 180
    vehicle_distance_pixels: int = 140
    steering_gain: float = 0.003
    max_steering: float = 1.0
    lane_offset_threshold_pixels: float = 50.0
    minimum_lane_confidence: float = 0.25
    lane_centering_speed_kmh: float = 24.0
    follow_vehicle_distance_m: float = 22.0
    follow_distance_target_m: float = 14.0
    follow_speed_kmh: float = 16.0
    follow_speed_gain: float = 1.0
    slow_down_speed_kmh: float = 10.0
    recovery_speed_kmh: float = 8.0
    recovery_hold_frames: int = 5
    emergency_obstacle_distance_m: float = 5.0
    emergency_pedestrian_distance_m: float = 12.0
    emergency_classes: tuple[str, ...] = ("person", "car", "bus", "truck", "motorcycle")
    vehicle_classes: tuple[str, ...] = ("car", "bus", "truck", "motorcycle")
    speed_smoothing_factor: float = 0.35
    state_logging_enabled: bool = True


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    log_dir: str
    input: InputConfig
    detection: DetectionConfig
    lane: LaneConfig
    control: ControlConfig


def _resolve_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def load_config(path: str | Path = "config/settings.yaml") -> AppConfig:
    """Load YAML settings and convert them into immutable typed sections."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    detection_raw = dict(raw.get("detection", {}))
    detection_raw["target_classes"] = tuple(detection_raw.get("target_classes", []))
    detection_raw["collision_relevant_positions"] = tuple(
        detection_raw.get("collision_relevant_positions", ("left", "centre", "right"))
    )
    detection_raw["collision_high_classes"] = tuple(
        detection_raw.get("collision_high_classes", ("person", "car", "bus", "truck", "motorcycle"))
    )
    detection_raw["collision_medium_classes"] = tuple(
        detection_raw.get("collision_medium_classes", ("person", "car", "bus", "truck", "motorcycle"))
    )
    control_raw = dict(raw.get("control", {}))
    control_raw["emergency_classes"] = tuple(
        control_raw.get("emergency_classes", ("person", "car", "bus", "truck", "motorcycle"))
    )
    control_raw["vehicle_classes"] = tuple(
        control_raw.get("vehicle_classes", ("car", "bus", "truck", "motorcycle"))
    )
    return AppConfig(
        project_name=raw.get("project", {}).get("name", "Autonomous Driving"),
        log_dir=raw.get("project", {}).get("log_dir", "logs"),
        input=InputConfig(**{**raw.get("input", {}), "source": _resolve_source(raw.get("input", {}).get("source", 0))}),
        detection=DetectionConfig(**detection_raw),
        lane=LaneConfig(**raw.get("lane", {})),
        control=ControlConfig(**control_raw),
    )
