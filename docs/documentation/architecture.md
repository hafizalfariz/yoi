# Architecture

This document describes the YOI system architecture in production terms: service boundaries, runtime responsibilities, and data paths.

## Audience and Use

- Audience: Solution architects, senior engineers, and reviewers.
- Use this document when validating module boundaries, runtime responsibilities, and deployment-level design decisions.

## Purpose

- Provide a shared technical view of the runtime architecture.
- Clarify module ownership and data boundaries.
- Align deployment and operations with system design.

## Scope

- Runtime processing from input source to output artifacts.
- Configuration lifecycle from Builder to Engine execution.
- Deployment topology for CPU, GPU, and Builder profiles.

## Technical Layers

### Ingestion Layer

- Accepts input from local video files or RTSP streams.
- Produces a normalized frame stream for inference.
- Uses active YAML configuration from `configs/app`.

### Inference Layer

- Runs object detection with Ultralytics YOLO.
- Supports ONNX Runtime execution path.
- Produces detection candidates per frame.

### Tracking Layer

- Uses ByteTrack as primary tracker.
- Uses centroid-based fallback when required.
- Supports optional lightweight ReID continuity.

### Analytics Layer

- Computes line crossing events.
- Computes region crowd metrics.
- Computes dwell time metrics.
- Produces frame-level and event-level summaries.

### Output and Streaming Layer

- Writes annotated outputs and structured artifacts.
- Exports JSON, CSV, and alert payloads.
- Publishes RTSP via FFmpeg to MediaMTX.

## Runtime Modules

- `src/app/`: Startup, validation, config resolution, and runtime orchestration.
- `yoi/`: Core modules for inference, tracking, analytics, streaming, and utilities.
- `config_builder/`: Configuration authoring service and UI.
- `docker/` and `docker-compose.yml`: Build/runtime definitions.
- `tests/` and `config_builder/tests/`: Coverage for configuration and runtime behavior.

## Execution Summary

1. Runtime starts and resolves active configuration.
2. Input source emits frames for inference and tracking.
3. Analytics computes events and derived metrics.
4. Output subsystem writes artifacts and publishes stream.
5. Runtime exits through graceful shutdown path.

## Data Paths

- `input/`: Source video inputs.
- `models/`: Model files and metadata.
- `configs/app/`: Active YAML configurations.
- `logs/`: Runtime and service logs.
- `output/`: Processed artifacts and media outputs.

## Related Documents

- [Main README](../../README.md)
- [Runtime Flow](runtime-flow.md)
- [Configuration Builder](configuration-builder.md)
- [Operations](operations.md)
