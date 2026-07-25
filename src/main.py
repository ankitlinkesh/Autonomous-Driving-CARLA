"""Command-line entry point for the perception-to-control pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2

from config.config import load_config
from src.decision_module import DecisionModule
from src.image_processing import draw_boxes, draw_fps, draw_text, resize
from src.lane_detection import LaneDetector
from src.logger import create_logger, log_frame
from src.object_detection import YOLOv8Detector
from src.vehicle_controller import VehicleController


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff"}


def _detect_input_mode(source: int | str) -> str:
    """Classify a configured source as webcam, video, or image."""
    if isinstance(source, int):
        return "webcam"
    return "image" if Path(source).suffix.lower() in IMAGE_EXTENSIONS else "video"


def _process_frame(
    frame: Any,
    frame_number: int,
    input_mode: str,
    config: Any,
    logger: Any,
    lane_detector: LaneDetector,
    detector: YOLOv8Detector,
    decision_module: DecisionModule,
    controller: VehicleController,
):
    """Run the existing perception/control pipeline and return its final frame."""
    frame = resize(frame, config.input.image_width, config.input.image_height)
    lane = lane_detector.detect(frame)
    detections = detector.detect(frame)
    decision = decision_module.decide(
        lane.lane_offset,
        detections.detections,
        frame.shape[1],
        lane_confidence=lane.lane_confidence,
    )
    command = controller.send(
        decision.steering,
        decision.target_speed_kmh,
        decision.brake,
        driving_state=decision.current_state,
        lane_offset=lane.lane_offset,
        lane_confidence=lane.lane_confidence,
    )
    rendered = draw_boxes(lane.image, [d.to_dict() for d in detections.detections])
    rendered = draw_fps(rendered, detections.fps)
    lane_info = "lane offset: unavailable" if lane.lane_offset is None else f"lane offset: {lane.lane_offset:.1f}px"
    draw_text(rendered, lane_info, (20, 60))
    draw_text(rendered, f"decision: {decision.reason}", (20, 90), (0, 220, 255))

    controller_status = command.diagnostics.status if command.diagnostics is not None else "UNKNOWN"
    cv2.rectangle(rendered, (15, 105), (470, 370), (25, 25, 25), -1)
    overlay_lines = [
        f"Mode: {input_mode.title()}",
        f"Objects: {len(detections.detections)}",
        f"FPS: {detections.fps:.1f}",
        f"Steering: {command.steering:.2f}",
        f"Throttle: {command.throttle:.2f}",
        f"Brake: {command.brake:.2f}",
        f"State: {decision.current_state}",
        f"Controller: {controller_status}",
        f"Decision: {decision.reason}",
    ]
    for index, line in enumerate(overlay_lines):
        draw_text(rendered, line, (25, 130 + index * 24), (235, 235, 235))

    log_frame(
        logger,
        frame_number,
        detections.detections,
        command.steering,
        detections.inference_time_ms,
        detections.fps,
    )
    return rendered


def _save_output(rendered: Any, output_path: Path, logger: Any) -> None:
    """Save one rendered output and log success or failure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rendered):
        logger.warning("Failed to save output: %s", output_path)
    else:
        logger.info("Saved annotated output: %s", output_path)


def run(config_path: str = "config/settings.yaml") -> None:
    """Run the pipeline over one image or a video/webcam stream."""
    config = load_config(config_path)
    logger = create_logger(config.log_dir)
    lane_detector = LaneDetector(config.lane)
    detector = YOLOv8Detector(config.detection)
    decision_module = DecisionModule(config.control, logger=logger)
    controller = VehicleController(config=config.control)
    source = config.input.source
    input_mode = _detect_input_mode(source)
    logger.info("Detected input mode: %s (source=%s)", input_mode, source)
    logger.info("save_outputs enabled: %s", config.input.save_outputs)

    if input_mode == "image":
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"Unable to read image source with cv2.imread(): {source}")
        logger.info("Loaded image resolution: width=%d height=%d", image.shape[1], image.shape[0])
        rendered = _process_frame(image, 1, input_mode, config, logger, lane_detector, detector, decision_module, controller)
        if config.input.save_outputs:
            _save_output(rendered, Path("assets/outputs") / "annotated_image.jpg", logger)
        if rendered is None:
            logger.error("Rendered image is None; skipping imshow()")
        elif not config.input.display:
            logger.info("imshow() skipped because display is disabled")
        else:
            logger.info("Reached cv2.imshow() for final image")
            cv2.imshow("Autonomous Driving Pipeline", rendered)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        logger.error("cv2.VideoCapture() could not open %s source: %s", input_mode, source)
        raise RuntimeError(f"Unable to open {input_mode} source: {source}")
    logger.info("Opened cv2.VideoCapture() for %s source", input_mode)
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.info("Input stream ended or returned no frame")
                break
            frame_number += 1
            rendered = _process_frame(frame, frame_number, input_mode, config, logger, lane_detector, detector, decision_module, controller)
            if config.input.save_outputs:
                _save_output(rendered, Path("assets/outputs") / f"frame_{frame_number:06d}.jpg", logger)
            if rendered is None:
                logger.error("Rendered frame is None; skipping imshow()")
                continue
            if not config.input.display:
                logger.info("imshow() skipped because display is disabled")
                continue
            logger.info("Reached cv2.imshow() for %s frame %d", input_mode, frame_number)
            cv2.imshow("Autonomous Driving Pipeline", rendered)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested with key q")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Released VideoCapture and destroyed OpenCV windows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CARLA-ready perception pipeline")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
