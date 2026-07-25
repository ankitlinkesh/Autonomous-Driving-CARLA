# Dataset Guide

This directory is reserved for datasets and dataset metadata. Keep large or restricted datasets out of Git history.

Recommended layout:

```text
datasets/
├── raw/          # Original downloads or CARLA recordings
├── processed/    # Deterministic, generated data
└── metadata/     # Class maps, splits, and collection notes
```

For YOLO training, store images and labels according to the Ultralytics dataset format and document the source, license, and split policy.

