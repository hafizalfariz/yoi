# Configuration Builder

This service provides controlled authoring of YOI YAML runtime configuration files.

## Audience and Use

- Audience: Operations engineers, QA engineers, and system integrators.
- Use this document when creating, validating, and handing off runtime configuration safely.

## Purpose

- Build valid runtime configuration from structured inputs.
- Manage feature parameters consistently.
- Save generated YAML directly to engine configuration path.

## Operational Scope

- UI-assisted parameter authoring for runtime features.
- API-driven build, parse, and save operations.
- Immediate handoff to engine through `configs/app` output.

## Supported Features

- `line_cross`
- `region_crowd`
- `dwell_time`

Each feature supports threshold, scheduling, and tracking parameter tuning.

## Multi-Video Inference Input

- Builder supports multi-video inference payload using `video_files`.
- Runtime executes those files sequentially in the order provided.
- Recommended payload pattern:
	- `source_mode: inference`
	- `video_source: input/14.mp4` (first file)
	- `video_files: [input/14.mp4, input/15.mp4]`

Generated YAML keeps `video_source` as first entry and writes the full `video_files` list.

## Service Lifecycle

- Start service: `docker compose --profile builder up -d config-builder`
- Stop service: `docker compose --profile builder stop config-builder`
- Stop and remove profile resources: `docker compose --profile builder down`

Default endpoint: `http://localhost:8032`

## Standard Usage

1. Open the Builder UI.
2. Select feature type and draw geometry.
3. Tune thresholds and tracking parameters.
4. Build and review YAML output.
5. Save configuration for runtime use.

## Operational Sequence

1. Start Builder service.
2. Build or adjust feature configuration.
3. Save YAML to runtime configuration path.
4. Hand off configuration to runtime execution.

## Runtime Handoff

- Saved YAML is written to `configs/app/<config_name>.yaml`.
- Engine configuration resolver can consume the file directly.
- No manual copy step is required between Builder and Engine.

## Save Behavior

When save is executed:

- YAML is generated from current payload.
- File is written to `configs/app/<config_name>.yaml`.
- Save metadata and download URL are returned.

This keeps Builder output immediately consumable by the engine runtime.

## Output Layout Expectation

For `annotated_video` output mode with file/video input, runtime writes results under:

- `output/<config_name>/<video_name>_<timestamp>/...`

This makes multi-video results grouped by config first, then by processed video.

For RTSP input (`input.source_type: rtsp`), runtime routes artifacts under:

- `logs/<config_name>/<stream_name>_<timestamp>/...`

This routing is handled by engine runtime automatically and does not require a `logs` block in builder output.

## Builder Output Notes

- Builder no longer emits a top-level `logs:` section in generated YAML.
- Runtime defaults handle logs directories internally.
- Existing legacy payload fields for logs are ignored by builder output.

## API Surface

- `GET /health`
- `GET /api/features`
- `GET /api/feature-params`
- `GET /api/feature-params/{feature}`
- `POST /api/feature-params/{feature}`
- `DELETE /api/feature-params/{feature}`
- `POST /api/parse`
- `POST /api/build`
- `POST /api/save`
- `GET /api/download/{filename}`

## Related Documents

- [Main README](../../README.md)
- [Architecture](architecture.md)
- [Runtime Flow](runtime-flow.md)
- [Operations](operations.md)
