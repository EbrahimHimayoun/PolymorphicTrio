# CHAMELEON — offline design resizing

CHAMELEON rearranges an existing design for a new canvas. It preserves element
IDs/count/order, text, font families, colors and image assets while changing
positions, dimensions and font sizes. Stage one includes this engine, Dockerfile
and README. No model, API key or inference service is needed.

## Build and run

From this directory, with Docker running:

```sh
docker build --platform linux/amd64 -t chameleon-submission .
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD/input:/app/input:ro" \
  -v "$PWD/output:/app/output" \
  chameleon-submission
```

The build downloads pinned Python dependencies and the pinned `cr-renderer`
source. Runtime is offline: fonts, assets and dependencies are inside the image.
The Dockerfile verifies the official renderer during the build.

Each `input/*.json` contains `source` and `target_canvas`. The runner writes
`output/<task>_output.json` and `output/<task>_output.png`. Three samples are
included. The JSON interface is documented in
[the design schema](chameleon_design_schema_v0.9.json).

## Approach

`engine/resize.py` implements `resize(source, target_canvas)`. It infers roles
and related groups from geometry and typography, measures text with the official
renderer's fonts/wrapping, and compares stacks, columns and banner compositions.
Candidate scoring considers readability, hierarchy, product visibility, balance,
alignment and badge spacing. Grouping and positions adapt to each input and
target; there are no sample-ID lookups or fixed right-side coordinates.

A soft organization preference can only select candidates that do not increase
any text unit's preferred-size shortfall relative to the existing scorer's best
valid candidate. Bounds, text fit, image proportions, content preservation and
independent-unit collisions are checked after final repair. Bounded recovery
tries alternative placements; if none validates, the task fails explicitly and
the process returns nonzero while continuing other tasks.

## Verification and limitations

All 34 production tests passed natively and in offline Linux/amd64 Docker as
part of the verified pre-submission suite. To test the built image:

```sh
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD/tests:/app/tests:ro" \
  -v "$PWD/tools:/app/tools:ro" \
  -v "$PWD/input:/app/input:ro" \
  --entrypoint python chameleon-submission \
  -m unittest discover -s tests -v
```

Case three deliberately references absent artwork; its placeholder and render
warning are expected. Dense formats can still require small text. Search and
role inference are heuristic and do not guarantee an optimal design. Native and
Linux font metrics differ, so exact positions can differ across platforms.
Full-bleed backgrounds use the schema's `cover` behavior; foreground imagery
keeps its proportions. Original asset files are never regenerated.

Create the source ZIP and its SHA-256 checksum with
`python3 -m tools.package_submission`. Both files are written to `dist/`.
The archive includes the engine, Dockerfile, dependencies, assets/fonts, samples
and production tests. See [VALIDATION.md](VALIDATION.md) for measured checks and
remaining limitations.
