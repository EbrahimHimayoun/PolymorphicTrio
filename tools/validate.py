"""Strict local validator + render sweep for engine/resize.py.

Not part of the scored pipeline. Run from the repo root:

    python -m tools.validate                 # samples x format catalogue
    python -m tools.validate --render out/   # also write PNGs + contact sheets

Checks every output against the hard rules the judge describes (RULES.md):
ids preserved exactly once and in order, required fields, nothing off-canvas,
images kept in proportion, plus the engine's own quality bar: no collisions
between independent units and every text box measured with the renderer's
real fonts/wrap (no overflow, no extra lines).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

from engine.resize import analyse, measure, _el_box, _inter
from engine import resize as engine

REQUIRED_COMMON = ["id", "type", "x", "y", "width", "height"]
REQUIRED_BY_TYPE = {
    "text": ["content", "font_family", "font_size", "color"],
    "image": ["asset_id"],
    "shape": ["shape_type", "fill_color"],
}

# Common real-world formats (the brief: targets come from a catalogue of
# platform formats). The three sample targets are included.
FORMATS = {
    "instagram_post": (1080, 1080),
    "instagram_portrait": (1080, 1350),
    "story_reel": (1080, 1920),
    "youtube_thumbnail": (1280, 720),
    "facebook_cover": (820, 312),
    "twitter_header": (1500, 500),
    "linkedin_banner": (1584, 396),
    "leaderboard": (728, 90),
    "mobile_banner": (320, 50),
    "large_mobile_banner": (320, 100),
    "medium_rectangle": (300, 250),
    "wide_skyscraper": (160, 600),
    "half_page": (300, 600),
    "billboard": (970, 250),
    "pinterest_pin": (1000, 1500),
    "presentation_16x9": (1920, 1080),
    "a4_poster": (2480, 3508),
    "wide_banner_4x1": (1600, 400),
}


def check(source, out, target):
    probs = []
    if set(out) != {"canvas", "elements"}:
        probs.append(f"top-level keys {sorted(out)}")
    if out["canvas"] != {"width": target[0], "height": target[1]}:
        probs.append(f"canvas {out['canvas']}")
    src_ids = [e["id"] for e in source["elements"]]
    out_ids = [e.get("id") for e in out["elements"]]
    if src_ids != out_ids:
        probs.append("ids differ / reordered / duplicated")
    cw, ch = target
    an = analyse(source)
    by_id = {e["id"]: e for e in out["elements"]}
    for src in source["elements"]:
        e = by_id.get(src["id"])
        if e is None:
            continue
        for f in REQUIRED_COMMON + REQUIRED_BY_TYPE.get(e["type"], []):
            if f not in e:
                probs.append(f"{e['id']}: missing {f}")
        for f, v in src.items():
            if f in ("x", "y", "width", "height", "font_size", "corner_radius",
                     "stroke_width", "letter_spacing", "fit"):
                continue
            if e.get(f) != v:
                probs.append(f"{e['id']}: property '{f}' changed")
        x, y, w, h = _el_box(e)
        if x < -0.5 or y < -0.5 or x + w > cw + 0.5 or y + h > ch + 0.5:
            probs.append(f"{e['id']}: off-canvas {x:.1f},{y:.1f},{w:.1f},{h:.1f}")
        if e["type"] == "image" and src["height"] and e["height"]:
            if e["x"] == 0 and e["y"] == 0 and e["width"] == cw and e["height"] == ch and e.get("fit") == "cover":
                pass  # full-bleed background, cropped not stretched
            elif abs(e["width"] / e["height"] - src["width"] / src["height"]) > 0.01 * src["width"] / src["height"]:
                probs.append(f"{e['id']}: aspect changed")
        if e["type"] == "text":
            if e["font_size"] != int(e["font_size"]):
                probs.append(f"{e['id']}: non-integer font_size (renderer truncates)")
            fit = measure(e, e["width"], e["font_size"])
            if fit.max_w > e["width"] + 0.5:
                probs.append(f"{e['id']}: line overflows box ({fit.max_w:.1f} > {e['width']:.1f})")
            if fit.height > e["height"] + 0.5:
                probs.append(f"{e['id']}: text taller than box ({fit.height:.1f} > {e['height']:.1f})")
    # Collisions between independent units (layering inside a unit is fine).
    idx = {e["id"]: i for i, e in enumerate(out["elements"])}
    units = an.units
    for a in range(len(units)):
        for b in range(a + 1, len(units)):
            hit = None
            for i in units[a].ids:
                for j in units[b].ids:
                    ov = _inter(_el_box(out["elements"][i]), _el_box(out["elements"][j]))
                    if ov > 0.25:
                        hit = (out["elements"][i]["id"], out["elements"][j]["id"], ov)
            if hit:
                probs.append(f"collision {hit[0]} x {hit[1]} ({hit[2]:.1f}px2)")
    return probs


def stats(source, out):
    an = analyse(source)
    rows = []
    for u in an.units:
        if u.kind in ("text", "pill"):
            e = out["elements"][u.text_id]
            rows.append(f"{u.role}:{e['font_size']}")
    return " ".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="input")
    ap.add_argument("--render", default=None)
    ap.add_argument("--formats", default=None, help="comma list of format names")
    args = ap.parse_args()
    tasks = sorted(Path(args.inputs).glob("*.json"))
    formats = dict(FORMATS)
    if args.formats:
        formats = {k: v for k, v in FORMATS.items() if k in args.formats.split(",")}
    render = None
    if args.render:
        from renderer.converter.convert import render_design_to_png
        render = render_design_to_png
        Path(args.render).mkdir(parents=True, exist_ok=True)
    failures = 0
    total = 0
    worst = 0.0
    for t in tasks:
        task = json.loads(t.read_text())
        src = task["source"]
        targets = [("sample_target", (task["target_canvas"]["width"], task["target_canvas"]["height"]))]
        targets += list(formats.items())
        for name, (w, h) in targets:
            total += 1
            t0 = time.time()
            out = engine.resize(src, {"width": w, "height": h})
            dt = time.time() - t0
            worst = max(worst, dt)
            json.loads(json.dumps(out))  # serialisable
            probs = check(src, out, (w, h))
            status = "OK " if not probs else "BAD"
            failures += bool(probs)
            print(f"{status} {t.stem[:30]:30s} {name:20s} {w}x{h:<5d} {dt:5.2f}s  {stats(src, out)}")
            for p in probs:
                print("     -", p)
            if render:
                png = Path(args.render) / f"{t.stem}__{name}_{w}x{h}.png"
                render(out, "brand_kit/assets", str(png))
    print(f"\n{total - failures}/{total} outputs pass all checks; slowest {worst:.2f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
