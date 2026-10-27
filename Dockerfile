FROM --platform=linux/amd64 python:3.11-slim

# No bytecode litter, unbuffered logs, and the Hugging Face libraries that
# cr-renderer imports are told never to reach for the network (belt and
# braces: nothing in the pipeline calls them at run time).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# skia (the renderer backend) links libGL/libEGL and their GLVND/X11 chain,
# which libgl1 + libegl1 pull in; freetype/fontconfig/png ship inside the
# wheel. git is build-time only, for pip's pinned cr-renderer install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libegl1 \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Everything the engine needs at run time is baked in below. Stage-one scoring
# runs with --network none: no pip installs, API calls or downloads at run time.
COPY chameleon_design_schema_v0.9.json .
COPY brand_kit/ ./brand_kit/
COPY renderer/ ./renderer/
COPY engine/ ./engine/

# Build-time smoke test. If the real renderer or its font stack is broken, fail
# the BUILD -- otherwise main.py would silently fall back to placeholder-box
# PNGs and the engine to approximate (Pillow) text metrics at scoring time.
RUN python -c "\
from engine.resize import _ts, resize; \
assert _ts().kind == 'skia', 'renderer font stack unavailable'; \
from renderer.converter.convert import render_design_to_png; \
d = {'canvas': {'width': 400, 'height': 400}, 'elements': [{'id': 't', 'type': 'text', 'x': 20, 'y': 20, \
'width': 360, 'height': 60, 'content': 'Smoke test', 'font_family': 'Solstice Serif', 'font_size': 40, \
'color': 'rgba(0,0,0,1)'}]}; \
render_design_to_png(resize(d, {'width': 728, 'height': 90}), 'brand_kit/assets', '/tmp/smoke.png'); \
print('smoke test ok')" \
    && rm -f /tmp/smoke.png

# input/ and output/ are mounted as volumes at run time (see README) --
# not baked into the image.
RUN mkdir -p /app/input /app/output

ENTRYPOINT ["python", "-m", "engine.main"]