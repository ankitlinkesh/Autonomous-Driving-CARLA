# System Design

## Problem Statement

Autonomous vehicles must convert camera observations into safe, explainable driving actions. This project builds the perception-stage foundation for that problem and keeps simulator integration isolated so the same modules can be tested on recorded media.

## Objectives

1. Build a reproducible camera-to-command pipeline.
2. Detect lane geometry and relevant road users.
3. Produce interpretable decisions and normalized actuator commands.
4. Provide configuration, logging, tests, and extension points for CARLA.

## System Architecture

The pipeline is a synchronous frame loop. Each frame is resized, passed independently through the lane and object detectors, enriched by a lightweight tracker, merged by the decision module, and sent to a controller backend. Logging records frame number, object count, persistent IDs, distance estimates, collision risks, steering, inference latency, and FPS.

## Workflow

1. Open an image, video, or webcam stream.
2. Resize the frame to configured dimensions.
3. Detect lane boundaries using grayscale, blur, Canny, ROI masking, probabilistic Hough lines, and slope-weighted averaging.
4. Run YOLOv8 and normalize its boxes into project-level `Detection` objects.
5. Match detections to recent tracks using class-aware IoU/centroid gating. IDs survive short missed-detection intervals configured by `tracker_max_missed_frames`.
6. Estimate approximate distance from focal length, assumed object height, and image-space box height. Classify each object as Left, Centre, or Right and compute a configurable LOW/MEDIUM/HIGH collision risk.
7. Apply safety-first rules: brake for a close pedestrian, reduce speed for a close vehicle, and steer from lane offset.
8. Convert the intent to bounded steering, throttle, and brake values.
9. Render and log the result.

## Module Descriptions

- `config/config.py`: loads validated, typed settings from YAML.
- `src/image_processing.py`: reusable OpenCV drawing and image helpers.
- `src/lane_detection.py`: classical lane geometry estimation.
- `src/object_detection.py`: optional-at-import YOLOv8 inference adapter, lightweight persistent tracker, monocular distance estimator, relative-position classifier, and collision-risk rules.
- `src/decision_module.py`: deterministic safety rules.
- `src/vehicle_controller.py`: backend-neutral command interface and explicit CARLA boundary.
- `src/logger.py`: console/file logging with structured frame events including object IDs, distances, risks, and counts.
- `src/main.py`: composition root and command-line entry point.

## Future Work

The object-distance model is intentionally a monocular approximation, not depth estimation. CARLA integration should add a sensor/input adapter and instantiate `CarlaBackend` with a real CARLA vehicle. It must not alter lane, detector, decision, or logging contracts. Future evaluation should report lane offset error, object precision/recall, tracking ID switches, distance error against ground truth, braking latency, FPS, and scenario-level collision outcomes.
