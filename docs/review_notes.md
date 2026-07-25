# Review Notes: Questions and Answers

## Why use both OpenCV and YOLOv8?

Lane markings are a structured geometric feature that can be detected efficiently with classical vision. YOLOv8 handles varied semantic objects such as people, vehicles, traffic lights, and signs. The combination is computationally practical and easy to explain in a PBL presentation.

## Is CARLA already implemented?

No. CARLA is an explicit future integration boundary. The repository does not fake simulator behavior. `CarlaBackend` raises a clear error if the Python API is absent, and a real adapter can later implement the same `VehicleBackend` contract.

## How are safety decisions prioritized?

A close pedestrian has highest priority and commands full braking. A close vehicle reduces target speed. Otherwise, lane offset contributes a bounded steering command. These rules are simple, deterministic, and easy to test.

## How should performance be improved?

Use a smaller model or GPU inference, batch only when latency permits, track lanes across frames, calibrate distance using depth or camera geometry, and profile each stage separately.

## What evidence should be added for final submission?

Include annotated sample frames, detector metrics, lane-offset plots, FPS/latency tables, test output, configuration snapshots, and a short video or CARLA scenario replay when available.

