"""Structured runtime logging for reproducible experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def create_logger(log_dir: str = "logs", name: str = "autonomous_driving") -> logging.Logger:
    """Create a console and file logger, avoiding duplicate handlers."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(path / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    return logger


def log_frame(logger: logging.Logger, frame_number: int, objects: list[Any], steering: float, inference_time_ms: float, fps: float) -> None:
    """Log frame metrics plus structured tracking metadata as JSON.

    Legacy lists of string labels are still accepted.
    """
    structured_objects = []
    for obj in objects:
        if isinstance(obj, str):
            structured_objects.append(obj)
        else:
            structured_objects.append({
                "object_id": getattr(obj, "object_id", 0),
                "class_name": getattr(obj, "class_name", getattr(obj, "label", "object")),
                "estimated_distance": getattr(obj, "estimated_distance", None),
                "collision_risk": getattr(obj, "collision_risk", "LOW"),
                "relative_position": getattr(obj, "relative_position", "Centre"),
            })
    payload: dict[str, Any] = {"frame": frame_number, "object_count": len(objects), "detected_objects": structured_objects,
                               "steering_angle": round(steering, 4),
                               "inference_time_ms": round(inference_time_ms, 2), "fps": round(fps, 2)}
    logger.info(json.dumps(payload))
