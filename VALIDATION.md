# CHAMELEON engine: validation

How to verify the engine builds and runs offline exactly as stage-one scoring
runs it, and how to validate layout quality on Levels 1 to 3. Each step lists
the command, what it proves and the result we recorded.

## 1. What changed and why

| File | Change |
|---|---|
| `engine/resize.py` | Full rewrite: renderer-exact typesetting, geometry-based role detection, lane/column reflow for wide canvases, rhythm-preserving stacks for tall ones, anchored layout for same-ratio resizes, validated candidate search. |
| `Dockerfile` | Pinned stack, offline env vars, build-time smoke test, dropped unused `libglib2.0-0`. |
| `requirements.txt` | Every package pinned; `cr-renderer` pinned to commit `a17e1fb`. |
| `tools/` (dev only, not in the image) | `validate.py` strict checker and format sweep, `synthetic.py` adversarial designs, `contact_sheet.py`, `debug_candidates.py`. |

**Root cause of the old Level 3 collisions.** The previous engine measured
glyph widths at 100px and scaled them down. At banner sizes skia hints advances
to whole pixels, so small text renders roughly 10% wider than predicted (for
example "Shop the Summer Line" in bold at 12px: 137px predicted, 153px real). It
also emitted fractional font sizes, which the renderer truncates with `int()`.
On the 728x90 sample, 4 of 8 text elements wrapped onto more lines than the
engine planned. The new engine measures with the renderer's own fonts, integer
size, wrap rule, style runs and ascent/descent, so every planned line break is
the rendered one.

| Level 3 output, strict validator | Problems |
|---|---|
| Previous engine | 11 (8 fractional font sizes; headline, CTA and ribbon text taller than their boxes, which is the visible headline/subhead collision) |
| New engine | 0 |

## 2. Docker: build and offline run

Run from the repo root on the reference machine (Apple Silicon builds the
amd64 image under emulation, which is slower but identical).

```bash
# 1. Build (network allowed: this is when pip installs everything)
docker build --platform linux/amd64 -t my-chameleon-engine .
```

Expect the build log to contain `smoke test ok`. That step imports the engine
and the real renderer, asserts the skia font stack is active, runs `resize()`
and renders a PNG. If skia or the renderer were broken, the build fails here
instead of the scoring run silently producing placeholder-box PNGs.

```bash
# 2. Confirm the image architecture
docker image inspect my-chameleon-engine --format '{{.Os}}/{{.Architecture}}'
# expected: linux/amd64

# 3. Prove --network none really has no network (this SHOULD fail)
docker run --rm --network none --entrypoint python my-chameleon-engine \
  -c "import socket; socket.create_connection(('pypi.org', 443), timeout=3)"
# expected: an error such as "Temporary failure in name resolution"

# 4. The scored run, exactly as stage one invokes it
rm -f output/*.json output/*.png
docker run --rm --network none \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  my-chameleon-engine
```

Expected output (timings vary; first task includes font loading):

```
[chameleon_sample_level1_warmup.json] OK in 0.6s -> ..._output.json, ..._output.png
[chameleon_sample_level2_core.json] OK in 0.1s -> ..._output.json, ..._output.png
[chameleon_sample_level3_advanced.json] OK in 0.5s -> ..._output.json, ..._output.png
```

Then open `output/*.png`. They must show real fonts and brand assets; labelled
boxes would mean the renderer import failed.

```bash
# 5. Optional: run the strict validator inside the same offline image
docker run --rm --network none \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/tools:/app/tools \
  --entrypoint python my-chameleon-engine -m tools.validate
# expected last line: 57/57 outputs pass all checks
```

### Offline hardening in the image

- All dependencies install at build time from fully pinned versions; the
  renderer is pinned to an exact commit, so an upstream change cannot alter
  rendering between your test and scoring.
- `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`: `cr-renderer` imports the
  Hugging Face libraries; these flags guarantee they never attempt a download.
  The vendored converter builds its `FontManager` from `brand_kit/fonts`, so the
  `fonts.pickle` download path is never used.
- `libgl1` and `libegl1` cover skia's native dependencies (verified with `ldd`
  on the wheel: libGL/libEGL plus the GLVND/X11 chain they pull in; freetype,
  fontconfig and libpng ship inside the wheel). `libglib2.0-0` was removed:
  nothing links it, and on newer Debian bases it only exists under a renamed
  t64 package, a needless build risk.
- The constant `FROM --platform=linux/amd64` triggers one Docker lint warning
  (`FromPlatformFlagConstDisallowed`). It is intentional: it guarantees an amd64
  image even if `--platform` is forgotten on an Apple Silicon host.

### What we ran (results)

Docker Hub (`registry-1.docker.io`) is blocked by our sandbox's network policy,
so `docker build` could not pull `python:3.11-slim` there; run steps 1 to 5 on
your machine. We reproduced the image as closely as possible without the base
image:

| Check | Result |
|---|---|
| `docker build --check .` (Dockerfile parse and lint) | Parses; only the intentional platform warning |
| Fresh Python 3.11.15 venv, `pip install --no-cache-dir -r requirements.txt` (the image's install step, same manylinux x86_64 wheels) | Installs cleanly; `pip check`: no broken requirements |
| The Dockerfile's smoke-test command, verbatim | `smoke test ok` |
| `python -m engine.main` inside a Linux network namespace with no interfaces (`unshare -n`, the mechanism behind `--network none`), image env vars set, Python audit hook logging every DNS lookup and socket connect | Exit 0; **0 network attempts**; 3 JSON + 3 real PNGs written; all self-checks OK |
| All three output JSONs from that offline Python 3.11 run vs an online Python 3.13 run | Identical |

The namespace test is a useful CI substitute wherever Docker is unavailable:

```bash
cd /path/to/app-copy   # same layout as /app: engine/, renderer/, brand_kit/, input/, output/
unshare -n env -i PATH=/usr/bin:/bin HOME=/root HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /path/to/py311-venv/bin/python -m engine.main      # needs root (or unshare -rn)
```

## 3. Level 1 to 3 validation

```bash
pip install -r requirements.txt
python -m engine.main                                  # the three sample tasks
python -m tools.validate                               # samples x 18 real-world formats
python -m tools.validate --render out/ && python -m tools.contact_sheet out/
```

`tools/validate.py` checks every output against the hard rules in RULES.md and
the engine's own quality bar:

- same ids, same count, same order, only `canvas` and `elements` at top level
- required fields present; no style property changed other than size-dependent
  ones (font size, corner radius, stroke width, letter spacing) and a
  background's `fit`
- every element's rotated bounding box inside the canvas
- images keep their aspect ratio (full-bleed backgrounds are cropped with
  `fit: cover`, never stretched)
- integer font sizes (the renderer truncates fractions)
- every text measured with the real renderer stack: no line wider than its box,
  no text taller than its box
- no collisions between independent units (layering inside a group, such as a
  button label on its shape, is allowed)

### Results

| Sweep | Result |
|---|---|
| 3 samples x (own target + 18 formats: IG post/portrait/story, YouTube, Facebook/Twitter/LinkedIn headers, leaderboard, mobile banners, medium rectangle, skyscrapers, billboard, Pinterest, 16:9, A4, 4:1) | **57/57 pass**, slowest 0.7s |
| 5 adversarial synthetic designs (`python -m tools.synthetic DIR`) x 19 targets | **94/95 pass**, slowest 3.7s (a 37-element design) |
| Same input twice | Identical output (deterministic) |
| Caller's `source` dict after `resize()` | Unmodified |
| skia unavailable (import blocked) | Falls back to Pillow metrics; valid output, no crash |

### Per-level expectations on the sample targets

**Level 1, 1080x1080 to 1080x1350.** Stack layout keeping the source's vertical
rhythm (logo near the top, headline and CTA where the source placed them),
type scaled 1.25x (headline 75px, CTA 37px). Check: nothing cramped, spacing
proportionate, logo centred.

**Level 2, 1080x1080 to 1600x400.** Lane columns: brand and headline on the
left, CTA centre-right, product on the right, "NEW FLAVOR" badge floating near
its original top position, style-run colouring intact (headline 82px on one
line, subhead 35px, CTA 32px). The corner-bleeding decorative circle is kept
on-canvas, so main.py reports no off-canvas warning.

**Level 3, 1080x1350 to 728x90.** All 17 elements kept, no overlaps, nothing
off-canvas:

| Role (inferred, never from ids) | Elements | Placement | Font |
|---|---|---|---|
| hero | product | full-height left column, 78x78 | |
| headline | headline | copy column, 3 lines | 11px |
| body | subhead | under headline, 2 lines | 7px |
| fine | legal_text | bottom of copy column, 1 line | 6px |
| logo | logo | right column top, 120x37 | |
| cta | cta_bg + cta_label | right column, compact pill | 9px |
| badge x3 | badge1-3 | one row under the CTA | 7px |
| tag | ribbon (rotated -12 deg) | top-right corner, rotation kept | 7px |
| stamp | holiday_watermark (missing asset) | free space beside the CTA; renderer draws its placeholder | |

At its own 1080x1350 size, Level 3 reproduces the source layout (anchored
candidate) except for two minimal fixes: the ribbon moves 0.6px back on-canvas
and the legal line moves 22px down to clear the watermark it overlapped in the
source.

## 4. Known limits

- **Extreme density.** Level 3 at 320x50 is valid but its text is 3 to 5px.
  The engine relaxes readability floors in stages only when nothing fits.
- **One synthetic failure.** 37 random, mutually overlapping elements squeezed
  into 320x50: no layout fits, so the least-colliding option (a scaled copy of
  the source) is used and its source overlaps remain.
- **Backgrounds.** A full-bleed background is resized to the canvas with
  `fit: cover` (cropped, never distorted). A checker comparing raw box aspect
  ratios for every image would flag it; there is no other way to fill a new
  canvas without going off-canvas.
- **Decorations.** Low-opacity shapes are clamped on-canvas, so a circle that
  bled off a corner in the source sits inside the corner instead. Any text
  layering that existed in the source is preserved; new text coverage is
  avoided.