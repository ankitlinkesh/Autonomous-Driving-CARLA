# Project Report

## Abstract

This project implements a modular autonomous-driving pipeline for an academic Machine Learning PBL. It combines classical OpenCV lane detection with YOLOv8 object detection and a transparent rule-based decision layer. The current scope is perception and command generation from camera media, with a controlled interface for future CARLA integration.

## Motivation

A staged architecture makes it possible to evaluate perception independently of a simulator, compare algorithms, and replace one component without rewriting the complete application. Safety-related behavior is visible in code and configuration, which is useful for academic review.

## Methodology

Frames are resized and analyzed in parallel paths. Lane geometry uses edge detection and a region-of-interest Hough transform. YOLOv8 provides object classes and confidence values. The decision module evaluates object proximity and lateral lane offset, then the controller converts the result into normalized actuator commands.

## Expected Outcomes

- A working baseline for lane and road-user perception.
- Explainable control behavior suitable for demonstrations.
- Reusable interfaces for recorded data, webcam input, and eventual CARLA sensors.
- Logs that support latency and FPS analysis.

## Limitations

The lane detector assumes visible lane markings and a forward-facing camera. The rule system is a baseline, not a certified autonomous-driving policy. Distance is estimated from image-space box size rather than calibrated depth. YOLO performance depends on the checkpoint and dataset domain.

## Evaluation Plan

Evaluate lane center/offset against annotated frames, detector precision/recall and mAP, end-to-end FPS, decision latency, and safety outcomes across daylight, rain, occlusion, and traffic-density scenarios. Once CARLA is connected, use repeatable routes and traffic seeds.

## Conclusion

The repository provides a clean perception-stage baseline that can grow toward a simulator-backed autonomy stack without pretending that CARLA is available when it is not.

