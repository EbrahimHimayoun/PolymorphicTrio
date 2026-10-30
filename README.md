# CHAMELEON — offline design resizing

CHAMELEON rearranges an existing design's text, images and shapes for a new
canvas size or aspect ratio. It moves and reflows content instead of stretching
the entire design. It preserves element IDs/count, text, font families, colors
and image assets, while adjusting placement, dimensions and font sizes.

The phase-one engine deliverable is a working `resize(source, target_canvas)`
implementation producing resized design JSON and a PNG through the official
renderer. Both artifacts are evaluated; structural validity alone does not
prove visual quality. The starter FAQ also describes a required viewer add-on;
this repository's engine does not implement that viewer. See [RULES.md](RULES.md)
and [FAQ.md](FAQ.md) for submission context.

```text
input/*.json  (source design + target canvas)
       |
       v
engine/main.py -> engine/resize.py -> official renderer
       |
       v
output/<task>_output.json
output/<task>_output.png
```

## Current approach

The engine is rule-based Python. No AI model, online API or model download is
required. Runtime uses local brand fonts/assets and installed dependencies.

It compares source-preserving layouts, vertical stacks, equal/unequal columns,
and visual/text/control lanes. Unequal widths account for measured text needs
and image proportions. Importance is inferred conservatively from source
font sizes, geometry and relationships, rather than IDs, asset names or notes.
Units receive different scales according to importance and space pressure.

Simple text/rectangle groups can compact or share rows, resizing padding and
background together. Clearly aligned rotated pairs can flex; uncertain groups
remain rigid. A text-growth pass explores available rectangles, measures the
complete wrapped text, and rejects growth that clips or creates new collisions.
Decorations use source overlap evidence or canvas-relative anchors.

`TEXT_FLOOR`, `Ctx.soft`, and the banner targets in `engine/resize.py` define
role-specific readability preferences. These are engineering preferences,
**not official limits**. Shortfalls influence scoring; they do not authorize
dropping content. The candidate search and greedy growth pass do not guarantee
an optimal layout.

Skia's glyph advances differ between the macOS and Linux builds even with the
same fonts and package versions. The engine measures with the local official
renderer. If the usual short-banner candidates miss the copy targets, it also
tries compact brand/action groups with wrapped badges. This preserves readable
font sizes without changing the renderer; identical pixel layouts across
platforms are not promised. Regression fixtures distinguish measured integer
and fractional glyph advances while enforcing the same readability and fit
checks on both.

A deterministic review controller checks the finished geometry after rounding
and text-fit repair, including preserved fields, bounds, proportions, measured
text fit, line limits and collisions between independent units. Existing valid
candidates retain score priority. If those fail, the controller tries up to six
rejected candidates with at most three collision-repair rounds each, translating
whole units without changing their content or relative group positions. It
then proceeds through the existing font-floor stages and fallback structures.
This is a geometry review loop; it does not use a model or inspect rendered pixels.

If no candidate passes, `resize` raises `LayoutValidationError` with diagnostics.
The CLI reports the failed task, continues other tasks, and exits nonzero. It
writes no new artifacts for that task; previous output files may still exist.
This avoids silently exporting a known-invalid fallback. Failure means the
bounded search found no valid result, not proof that no feasible layout exists.

## Repository layout

```text
engine/resize.py                   layout implementation
engine/main.py                     input/output orchestration and self-check
renderer/                          official converter and rendering integration
brand_kit/assets/                  original image assets
brand_kit/fonts/                   local brand fonts
input/                             task JSON files (three samples included)
output/                            generated JSON and PNG previews (ignored by git)
tests/                             invariant/allocation tests and baseline fixture
chameleon_design_schema_v0.9.json   design format
RULES.md                           evaluation rules
VALIDATION.md                      measured results and known limitations
Dockerfile                         container entrypoint
```

## Python setup and local run

Use **Python 3.10 or newer**; the Dockerfile uses 3.11 and local validation uses
3.12. From the repository root, create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m engine.main
```

Installation needs access to download dependencies, including `cr-renderer`
from GitHub (Git must be available). Once installed, the engine runs offline.
Do not install dependencies during an offline run.

Optional example for the existing local Python 3.12 environment:

```sh
source .venv312/bin/activate
python -m engine.main
```

Or invoke that interpreter without activation:

```sh
.venv312/bin/python -m engine.main
.venv312/bin/python -m unittest discover -s tests -v
```

The standard test command in an activated environment is:

```sh
python -m unittest discover -s tests -v
```

The runner processes every `*.json` task in `input/` and writes an
`_output.json` and `_output.png` pair to `output/`. Keep the JSON and inspect
the rendered PNG at its intended display size. Tests check independent rotated
bounds, preserved properties/assets, image proportions, renderer-compatible text
fit, overlaps, hierarchy, readability preferences and allocation regressions.
Detailed results are in [VALIDATION.md](VALIDATION.md).

## Renderer and self-check

`engine/main.py` calls `renderer/converter/convert.py`'s
`render_design_to_png()`, which uses `cr-renderer` and local brand fonts/assets.
If importing the official renderer fails, the runner falls back to labeled
boxes. That fallback checks plumbing only; do not use it to judge layout quality.

The runner's self-check flags missing required fields, changed IDs and elements
off canvas. It is not the official scorer and does not establish text readability
or layout quality. The test suite adds checks, but cannot certify every input.

The Level 3 input deliberately references an absent asset:
`solstice_holiday_badge_2025.png`. The official renderer draws a red-bordered
missing-asset placeholder. The engine preserves that element and its asset ID,
tries a visual lane suited to its proportions, and expands/repositions standalone
missing images into collision-free space when possible. It does not remove the
placeholder, invent replacement artwork, or alter the renderer's appearance.

## Docker build and offline run

From the repository root:

```sh
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --network none \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  my-chameleon-engine
```

Build-time dependency installation needs network access. `--network none` models
the offline scoring run: all runtime dependencies, fonts and assets must already
be in the image. See the latest entry in [VALIDATION.md](VALIDATION.md) for
Docker/Linux and macOS checks; older entries describe historical results.

To run the regression suite against the built image, mount only the tests and
sample inputs (the engine, fonts, and renderer still come from the image):

```sh
docker run --rm --platform linux/amd64 --network none \
  -v "$(pwd)/tests:/app/tests:ro" \
  -v "$(pwd)/input:/app/input:ro" \
  --entrypoint python my-chameleon-engine -m unittest discover -s tests -v
```

## Known limitations

Extreme formats can still require tiny text and image regions even though every
element fits. Preferred font sizes are soft targets, with shortfalls penalized
in scoring. Geometry-based roles and grouping are heuristic; sampled column
widths and greedy growth can miss better arrangements. Rotation checks use
conservative enclosing rectangles. The deliberately missing asset's official
placeholder remains visible, and its diagnostic label may be clipped within
small proportional boxes. JSON validity does not imply universal readability.

See [VALIDATION.md](VALIDATION.md) for actual font sizes, verified checks and
remaining limitations, and the [design schema](chameleon_design_schema_v0.9.json)
for the input/output contract.
