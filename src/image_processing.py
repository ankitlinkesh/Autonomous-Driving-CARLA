"""Reusable OpenCV image processing helpers."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to an explicit width and height."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR pixels to float32 values in the range [0, 1]."""
    return image.astype(np.float32) / 255.0


def crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop an image using clipped coordinates."""
    height, width = image.shape[:2]
    return image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]


def draw_fps(image: np.ndarray, fps: float, origin: tuple[int, int] = (20, 30)) -> np.ndarray:
    """Draw a readable FPS label onto a copy of an image."""
    output = image.copy()
    return draw_text(output, f"FPS: {fps:.1f}", origin)


def draw_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw text with a dark outline for visibility."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
    return image


def draw_boxes(image: np.ndarray, boxes: Iterable[dict]) -> np.ndarray:
    """Draw legacy or structured detections with tracking and risk metadata."""
    output = image.copy()
    for detection in boxes:
        box = detection.get("bounding_box", detection.get("box"))
        x1, y1, x2, y2 = map(int, box)
        class_name = detection.get("class_name", detection.get("label", "object"))
        confidence = float(detection.get("confidence", 0.0))
        object_id = int(detection.get("object_id", 0))
        distance = detection.get("estimated_distance")
        position = detection.get("relative_position", "Centre")
        risk = str(detection.get("collision_risk", "LOW")).upper()
        risk_colors = {"LOW": (0, 200, 0), "MEDIUM": (0, 180, 255), "HIGH": (0, 0, 255)}
        color = risk_colors.get(risk, (255, 160, 0))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        distance_text = "n/a" if distance is None else f"{float(distance):.1f}m"
        text = f"ID {object_id} | {class_name} {confidence:.2f} | {distance_text} | {position} | {risk}"
        draw_text(output, text, (x1, max(20, y1 - 8)), color)
    return output
