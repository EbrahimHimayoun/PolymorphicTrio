"""CHAMELEON resize engine: role-aware reflow with renderer-compatible metrics.

resize(source, target_canvas) -> design   (same ids, same count, same order)

Pipeline
--------
1. Typesetting model. With Skia and the official font manager available, use
   the renderer's font lookup and space-only greedy wrap, integer font sizes,
   per-span advances, style runs, letter spacing and ascent/descent metrics.
   Measure each candidate size directly: hinting and rounding can change glyph
   advances, and macOS/Linux metrics can differ. Pillow provides an approximate
   fallback when the renderer stack cannot be initialized.

2. Analysis. The source is split into background(s), decorations, panels
   and layout *units*: a single text, a label+shape pill,
   an image, or a rigid group (anything else that shares a group_id or is
   geometrically contained). Roles use geometry, typography, grouping, text
   length and asset availability; IDs, notes and asset names are not read as
   semantic labels. Inferred roles include:
   headline, body, fine print, CTA, badge, tag (rotated pill), hero, logo,
   image, stamp (small corner image), group.

3. Candidate layouts.
   * Vertical stacks: source reading-order rows (corner items stay pinned
     to their side) and the poster lane order logo/visual/copy/action/fine.
     Spare height is distributed in proportion to the source's own vertical
     rhythm, so near-ratio resizes stay faithful.
   * Horizontal lanes for wide canvases (banners, leaderboards, headers):
     columns built from lanes -- brand, visual, copy, action -- in several
     orders, with fine print either under the copy or as a full-width strip.
     Small rotated tags and corner stamps float into measured free space.
   Stack and column searches vary a shared pressure value; banner candidates
   also vary zone sizes, copy splits, badge wrapping and hero position. Text
   uses role-specific elasticity and staged size floors; fitting helpers and
   last-resort repair can reduce sizes further. Banner copy can grow one pixel
   at a time within its available space and size caps.

4. Validation and scoring. Some candidates are pre-ranked before completion.
   Completed layouts are checked for bounds, independent-unit collisions using
   rotated enclosing boxes, measured text fit and foreground-image proportions.
   Scoring weighs readability, hierarchy, visual weight, coverage and balance.
   A source-informed organization penalty may reorder valid candidates only
   without increasing any text/pill unit's preferred-size shortfall relative
   to the best candidate under the base score. Finished geometry is checked
   again after repair. Bounded recovery translates whole colliding units;
   fallback structures use the same final gate. If none passes within the
   validator's tolerances, LayoutValidationError reports the remaining issues.

Content, font families, colours, asset references and z-order are preserved.
Geometry, integer font sizes and size-dependent style values may change.
Foreground rotation is preserved; full-canvas backgrounds are reset to zero
rotation and image backgrounds use fit='cover'. Original asset files are not
modified. Passing validation does not establish optimal layout or readability.
"""
from __future__ import annotations

import heapq
import itertools
import math
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_FONTS_DIR = _ROOT / 'brand_kit' / 'fonts'
_ASSETS_DIR = _ROOT / 'brand_kit' / 'assets'

# --------------------------------------------------------------------------
# Tunables (size values are in target-canvas pixels; weights are dimensionless)
# --------------------------------------------------------------------------

# How strongly each text role resists shrinking (text = cap * p ** exponent).
TEXT_ELASTICITY = {'headline': 1.0, 'cta': 1.0, 'body': 1.12, 'badge': 1.3,
                   'tag': 1.3, 'fine': 1.3, 'label': 1.1}
# Starting size floors, capped by source-derived limits and relaxed in stages.
# Text-fitting helpers and last-resort repair may go below these preferences.
TEXT_FLOOR = {'headline': 9, 'cta': 9, 'body': 7, 'badge': 7, 'tag': 7,
              'fine': 6, 'label': 7}
# Weight of each role in the layout score.
IMPORTANCE = {'divider': 0.2, 'headline': 3.0, 'cta': 2.2, 'hero': 1.8, 'logo': 1.6,
              'body': 1.5, 'group': 1.4, 'image': 1.0, 'badge': 0.8,
              'tag': 0.7, 'shape': 0.4, 'stamp': 0.35, 'fine': 0.5,
              'label': 1.0}

# A bounded preference, never a placement rule or a readability floor.
ORGANIZATION_WEIGHT = 2.0

# Extra breathing room for the official opaque missing-image diagnostic box.
MISSING_ASSET_GAP_FRACTION = 0.12  # capped at 12 target pixels

_BOLD_WEIGHTS = {'bold', 'semibold', 'extrabold', 'black'}
_FONT_WEIGHTS = {'thin', 'light', 'extralight', 'regular', 'medium',
                 'semibold', 'bold', 'extrabold', 'black'}


# ==========================================================================
# Geometry
# ==========================================================================

def _rot_box(x, y, w, h, deg):
    """Axis-aligned bounds of a w*h box rotated by `deg` about its centre."""
    if not deg:
        return x, y, w, h
    a = math.radians(deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    rw, rh = c * w + s * h, s * w + c * h
    return x + (w - rw) / 2, y + (h - rh) / 2, rw, rh


def _el_box(e):
    return _rot_box(e['x'], e['y'], e['width'], e['height'], e.get('rotation', 0) or 0)


def _union(boxes):
    boxes = list(boxes)
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _inter(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _hits(a, b, gap=0.0):
    return (a[0] < b[0] + b[2] + gap and b[0] < a[0] + a[2] + gap and
            a[1] < b[1] + b[3] + gap and b[1] < a[1] + a[3] + gap)


def _contains(outer, inner, tol=1.0):
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and
            inner[0] + inner[2] <= outer[0] + outer[2] + tol and
            inner[1] + inner[3] <= outer[1] + outer[3] + tol)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _alpha(color):
    """Alpha channel of an rgba()/rgb() string (1.0 when absent/unparseable)."""
    if not isinstance(color, str):
        return 1.0
    s = color.strip()
    if s.startswith('rgba(') and s.endswith(')'):
        parts = s[5:-1].split(',')
        if len(parts) == 4:
            try:
                return _clamp(float(parts[3]), 0.0, 1.0)
            except ValueError:
                return 1.0
    return 1.0


# ==========================================================================
# Typesetting -- a faithful model of the official renderer
# ==========================================================================

def _norm_weight(weight):
    if not weight:
        return 'regular'
    weight = str(weight).lower()
    return weight if weight in _FONT_WEIGHTS else 'regular'


def _is_italic(value):
    if isinstance(value, bool):
        return value
    return value == 'italic'


def _style_name(bold, italic):
    if bold and italic:
        return 'bolditalic'
    if bold:
        return 'bold'
    if italic:
        return 'italic'
    return 'regular'


class _Typesetter:
    """Glyph advances and vertical metrics, preferring the renderer's stack.

    Primary path: the brand-kit FontManager the official converter builds and
    Skia. If initialization fails, use Pillow and local brand fonts, rounding
    character advances up. That fallback approximates the renderer; rounding
    alone does not guarantee its measurements exceed Skia's.
    """

    def __init__(self):
        self.kind = 'pil'
        self._skia = None
        self._fm = None
        try:
            import skia  # type: ignore
            from renderer.converter.brand_fonts import load_brand_font_manager
            self._fm = load_brand_font_manager(str(_FONTS_DIR))
            self._skia = skia
            self.kind = 'skia'
        except Exception:  # renderer unavailable: estimate with Pillow
            self._skia = None
        self._faces = {}
        self._widths = {}
        self._metrics = {}

    # -- font objects --------------------------------------------------------
    def _face(self, family, weight, style):
        key = (family, weight, style)
        face = self._faces.get(key)
        if face is None:
            if self._skia is not None:
                import logging
                log = logging.getLogger('cr_renderer.fonts')
                level = log.level
                log.setLevel(logging.ERROR)  # fallback notices are expected here
                try:
                    data = self._fm.lookup(family, weight, style)
                finally:
                    log.setLevel(level)
                face = self._skia.Typeface.MakeFromData(data)
            else:
                face = self._pil_path(family, style)
            self._faces[key] = face
        return face

    @staticmethod
    def _pil_path(family, style):
        stem = ''.join(str(family).split())
        suffix = {'regular': 'Regular', 'bold': 'Bold', 'italic': 'Italic',
                  'bolditalic': 'BoldItalic'}[style]
        for candidate in (f'{stem}-{suffix}', f'{stem}-Regular'):
            for ext in ('.ttf', '.otf'):
                path = _FONTS_DIR / (candidate + ext)
                if path.is_file():
                    return str(path)
        fonts = sorted(_FONTS_DIR.glob('*.ttf'))
        return str(fonts[0]) if fonts else None

    # -- measurements --------------------------------------------------------
    def widths(self, family, weight, style, size, text):
        key = (family, weight, style, size, text)
        cached = self._widths.get(key)
        if cached is not None:
            return cached
        if not text:
            result = ()
        elif self._skia is not None:
            font = self._skia.Font(self._face(family, weight, style), size)
            glyphs = font.textToGlyphs(text)
            result = tuple(font.getWidths(glyphs))
            if len(result) != len(text):  # surrogate pairs etc.: per character
                result = tuple(sum(font.getWidths(font.textToGlyphs(c))) for c in text)
        else:
            from PIL import ImageFont
            path = self._face(family, weight, style)
            font = ImageFont.truetype(path, size) if path else ImageFont.load_default()
            result = tuple(math.ceil(font.getlength(c)) for c in text)
        self._widths[key] = result
        return result

    def metrics(self, family, weight, style, size):
        """(ascent, descent, cap_height) as positive pixel values."""
        key = (family, weight, style, size)
        cached = self._metrics.get(key)
        if cached is not None:
            return cached
        if self._skia is not None:
            m = self._skia.Font(self._face(family, weight, style), size).getMetrics()
            cap = m.fCapHeight or 0.7 * size
            result = (-m.fAscent, m.fDescent, cap)
        else:
            from PIL import ImageFont
            path = self._face(family, weight, style)
            font = ImageFont.truetype(path, size) if path else ImageFont.load_default()
            asc, desc = font.getmetrics()
            result = (float(asc), float(desc), 0.7 * size)
        self._metrics[key] = result
        return result


_TS = None


def _ts():
    global _TS
    if _TS is None:
        _TS = _Typesetter()
    return _TS


def _wrap(text, widths, max_width):
    """Match the official wrapper's greedy replacement of spaces by newlines."""
    if not text or max_width <= 0:
        return text
    chars = list(text)
    line_width = 0.0
    last_space = None
    for i, c in enumerate(chars):
        if c == '\n':
            line_width = 0.0
            last_space = None
            continue
        w = widths[i] if i < len(widths) else 0.0
        if line_width + w > max_width and last_space is not None:
            chars[last_space] = '\n'
            line_width = sum(widths[j] for j in range(last_space + 1, i + 1) if chars[j] != '\n')
            last_space = None
        else:
            line_width += w
        if c == ' ':
            last_space = i
    return ''.join(chars)


class TextFit:
    __slots__ = ('lines', 'max_w', 'height', 'asc', 'desc', 'cap', 'space_w')

    def __init__(self, lines, max_w, height, asc, desc, cap, space_w):
        self.lines, self.max_w, self.height = lines, max_w, height
        self.asc, self.desc, self.cap, self.space_w = asc, desc, cap, space_w


_FIT_CACHE = {}


def _text_key(e):
    runs = tuple((r.get('start'), r.get('end'), r.get('color'), r.get('font_weight'),
                  r.get('font_style')) for r in (e.get('style_runs') or []))
    return (str(e.get('content', '')), e.get('font_family', 'Montserrat'),
            e.get('font_weight'), e.get('font_style'), e.get('color'),
            float(e.get('line_height', 1.2) or 0), float(e.get('letter_spacing', 0) or 0),
            runs)


def measure(e, width, size):
    """Measure wrapped text using the renderer model, or the Pillow fallback."""
    size = int(size)
    key = (_text_key(e), round(float(width), 3), size)
    hit = _FIT_CACHE.get(key)
    if hit is not None:
        return hit
    ts = _ts()
    content = str(e.get('content', ''))
    family = e.get('font_family', 'Montserrat')
    weight = _norm_weight(e.get('font_weight'))
    bold = weight in _BOLD_WEIGHTS
    italic = _is_italic(e.get('font_style') or e.get('italic'))
    style = _style_name(bold, italic)
    n = len(content)

    wrapped = _wrap(content, ts.widths(family, weight, style, size, content), float(width))

    # Per-character style flags, exactly as the converter builds them
    # (style-run end indices are applied inclusively there).
    bflags = [bold] * n
    iflags = [style in ('italic', 'bolditalic')] * n
    colors = [e.get('color', 'rgba(0,0,0,1)')] * n
    for run in e.get('style_runs') or []:
        start = max(0, int(run.get('start', 0)))
        end = min(n - 1, int(run.get('end', n - 1)))
        if start > end:
            continue
        for i in range(start, end + 1):
            if 'color' in run:
                colors[i] = run['color']
            if 'font_weight' in run:
                bflags[i] = _norm_weight(run.get('font_weight')) in _BOLD_WEIGHTS
            if 'font_style' in run or 'italic' in run:
                iflags[i] = _is_italic(run.get('font_style') or run.get('italic'))

    variants = {}
    for b, it in set(zip(bflags, iflags)) or {(bold, False)}:
        rs = _style_name(b, it)
        rw = 'bold' if b else 'regular'
        variants[(b, it)] = (ts.widths(family, rw, rs, size, content), ts.metrics(family, rw, rs, size))
    ls = float(e.get('letter_spacing', 0) or 0)

    max_w = 0.0
    start = 0
    for line in wrapped.split('\n'):
        stop = start + len(line)
        width_sum, span_len, prev = 0.0, 0, None
        for i in range(start, stop):
            sig = (colors[i], bflags[i], iflags[i])
            if sig != prev:
                if span_len:
                    width_sum += ls * (span_len - 1)
                span_len, prev = 0, sig
            width_sum += variants[(bflags[i], iflags[i])][0][i]
            span_len += 1
        if span_len:
            width_sum += ls * (span_len - 1)
        max_w = max(max_w, width_sum)
        start = stop + 1

    asc = max(v[1][0] for v in variants.values())
    desc = max(v[1][1] for v in variants.values())
    cap = max(v[1][2] for v in variants.values())
    lh = float(e.get('line_height', 1.2) or 0) or 1.0
    lines = wrapped.count('\n') + 1
    height = (lines - 1) * size * lh + asc + desc
    space = ts.widths(family, weight, style, size, ' ')
    fit = TextFit(lines, max_w, height, asc, desc, cap, (space[0] if space else 0.3 * size) + ls)
    if len(_FIT_CACHE) > 200000:
        _FIT_CACHE.clear()
    _FIT_CACHE[key] = fit
    return fit


def natural_width(e, size):
    return measure(e, 1e9, size).max_w


def _shrink_wrap(e, width, size):
    """Try a tighter width while retaining the measured line count and fit."""
    fit = measure(e, width, size)
    tight = math.floor(min(width, fit.max_w + fit.space_w + 1.0) * 100) / 100
    if tight < width:
        alt = measure(e, tight, size)
        if alt.lines == fit.lines and alt.max_w <= tight + 1e-6:
            return tight, alt
    return width, fit


def _text_ok(e, width, size, max_height=None):
    fit = measure(e, width, size)
    if fit.max_w > width + 1e-6:
        return None
    if e.get('max_lines') and fit.lines > int(e['max_lines']):
        return None
    if max_height is not None and fit.height > max_height + 1e-6:
        return None
    return fit


# ==========================================================================
# Assets
# ==========================================================================

@lru_cache(maxsize=128)
def _asset_info(asset_id):
    """Return asset dimensions/alpha bounds, or None if missing or unreadable."""
    if not asset_id:
        return None
    path = _ASSETS_DIR / str(asset_id)
    if not path.is_file():
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            size = im.size
            bbox = None
            if im.mode in ('RGBA', 'LA', 'PA') or 'transparency' in im.info:
                bbox = im.convert('RGBA').getchannel('A').getbbox()
            return size[0], size[1], bbox or (0, 0, size[0], size[1])
    except Exception:
        return None


def _visible_fraction(e):
    """Estimate occupied image-box fractions from asset alpha bounds and fit.

    Missing/unreadable assets and empty alpha bounds use a full-box estimate.
    """
    info = _asset_info(e.get('asset_id'))
    w, h = max(e['width'], 1e-6), max(e['height'], 1e-6)
    if info is None:
        return 0.0, 0.0, 1.0, 1.0  # the renderer draws a full-box placeholder
    iw, ih, (bx0, by0, bx1, by1) = info
    fit = e.get('fit', 'contain')
    if fit == 'fill':
        return bx0 / iw, by0 / ih, bx1 / iw, by1 / ih
    scale = min(w / iw, h / ih) if fit != 'cover' else max(w / iw, h / ih)
    ox, oy = (w - iw * scale) / 2, (h - ih * scale) / 2
    return (_clamp((ox + bx0 * scale) / w, 0, 1), _clamp((oy + by0 * scale) / h, 0, 1),
            _clamp((ox + bx1 * scale) / w, 0, 1), _clamp((oy + by1 * scale) / h, 0, 1))


# ==========================================================================
# Analysis: units and roles
# ==========================================================================

class Unit:
    def __init__(self, uid, ids):
        self.uid = uid
        self.ids = ids
        self.kind = 'rigid'     # text | pill | image | rigid
        self.role = 'group'
        self.box = None         # source AABB
        self.text_id = None
        self.shape_id = None
        self.rotation = 0.0
        self.font = 0.0         # principal font size in the source
        self.corner = None      # 'tl' | 'tr' | 'bl' | 'br'
        self.side = 'center'    # horizontal side in the source
        self.order = 0
        self.floater = False
        self.lane = 'V'
        self.missing = False
        self.align = 'center'
        self.pill = None        # pill proportions
        self.visible = (0, 0, 1, 1)

    def __repr__(self):
        return f'Unit({self.uid},{self.role},{self.kind})'


class Analysis:
    pass


def _eff_alpha(e):
    a = float(e.get('opacity', 1) if e.get('opacity') is not None else 1)
    if e['type'] == 'shape':
        a *= _alpha(e.get('fill_color'))
    return a


def analyse(source):
    els = source['elements']
    sw, sh = float(source['canvas']['width']), float(source['canvas']['height'])
    canvas = (0.0, 0.0, sw, sh)
    n = len(els)
    boxes = [_el_box(e) for e in els]
    z = [e.get('z_index', i) if e.get('z_index') is not None else i for i, e in enumerate(els)]
    an = Analysis()
    an.sw, an.sh = sw, sh

    # ---- backgrounds: non-text, cover the canvas, painted below everything else
    backgrounds = []
    for i, e in enumerate(els):
        if e['type'] == 'text':
            continue
        if _inter(boxes[i], canvas) < 0.95 * sw * sh:
            continue
        others = [z[j] for j in range(n) if j != i and _inter(boxes[j], canvas) < 0.95 * sw * sh]
        if not others or z[i] <= min(others):
            backgrounds.append(i)
    bg = set(backgrounds)

    group_of = {}
    for i, e in enumerate(els):
        if e.get('group_id') is not None and i not in bg:
            group_of.setdefault(e['group_id'], []).append(i)

    # ---- decorations: ungrouped, faint or bleeding shapes that hold nothing
    def holds_something(i):
        return any(j != i and j not in bg and z[j] > z[i] and els[j]['type'] == 'text'
                   and _contains(boxes[i], boxes[j], 2) for j in range(n))

    decorations = []
    for i, e in enumerate(els):
        if i in bg or e['type'] != 'shape' or e.get('group_id') is not None:
            continue
        area = max(boxes[i][2] * boxes[i][3], 1e-6)
        outside = 1 - _inter(boxes[i], canvas) / area
        # A faint shape never "hosts" content even if its box happens to
        # enclose some; an opaque one only counts when it holds no text.
        if _eff_alpha(e) < 0.45:
            decorations.append(i)
        elif outside > 0.2 and e.get('shape_type') == 'ellipse' and not holds_something(i):
            decorations.append(i)
    deco = set(decorations)

    # ---- panels: large opaque ungrouped shapes that host other content
    panels = {}
    for i, e in enumerate(els):
        if i in bg or i in deco or e['type'] != 'shape' or e.get('group_id') is not None:
            continue
        if _eff_alpha(e) < 0.45:
            continue
        if boxes[i][2] * boxes[i][3] < 0.18 * sw * sh:
            continue
        kids = [j for j in range(n) if j != i and j not in bg and j not in deco and z[j] > z[i]
                and _contains(boxes[i], boxes[j], 2)]
        if kids:
            panels[i] = kids
    an.panels = panels

    # Inert: nothing visible to lay out (zero-size boxes, empty strings).
    inert = [i for i, e in enumerate(els) if i not in bg and i not in deco and i not in panels and (
        e['width'] <= 0 or e['height'] <= 0 or
        (e['type'] == 'text' and not str(e.get('content', '')).strip()))]
    an.inert = inert
    fg = [i for i in range(n) if i not in bg and i not in deco and i not in panels and i not in inert]

    # ---- units: union of group_id members and geometric containment
    parent = {i: i for i in fg}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for members in group_of.values():
        members = [m for m in members if m in parent]
        for m in members[1:]:
            join(members[0], m)
    for i in fg:
        if els[i]['type'] == 'text':
            continue
        for j in fg:
            if i == j or els[j].get('group_id') is not None or els[i].get('group_id') is not None:
                continue
            if z[j] <= z[i] or not _contains(boxes[i], boxes[j], 1):
                continue
            # A text inside a shape/image, or an image inside a shape (a frame).
            if els[j]['type'] == 'text' or (els[i]['type'] == 'shape' and els[j]['type'] == 'image'):
                if boxes[j][2] * boxes[j][3] <= 0.9 * boxes[i][2] * boxes[i][3] or els[j]['type'] == 'text':
                    join(i, j)
    groups = {}
    for i in fg:
        groups.setdefault(find(i), []).append(i)

    units = []
    for ids in groups.values():
        ids = sorted(ids)
        u = Unit(len(units), ids)
        u.box = _union(boxes[i] for i in ids)
        texts = [i for i in ids if els[i]['type'] == 'text']
        shapes = [i for i in ids if els[i]['type'] == 'shape']
        images = [i for i in ids if els[i]['type'] == 'image']
        u.font = max((float(els[i]['font_size']) for i in texts), default=0.0)
        if texts:
            aligns = [els[i].get('text_align', 'left') for i in texts]
            u.align = max(set(aligns), key=aligns.count)
        if len(ids) == 1 and texts and not (els[ids[0]].get('rotation') or 0):
            u.kind, u.text_id = 'text', ids[0]
        elif len(ids) == 1 and images:
            u.kind = 'image'
            u.rotation = els[ids[0]].get('rotation', 0) or 0
            u.missing = _asset_info(els[ids[0]].get('asset_id')) is None
            u.visible = _visible_fraction(els[ids[0]])
        elif len(ids) == 2 and len(texts) == 1 and len(shapes) == 1 and _pill_ok(els[texts[0]], els[shapes[0]]):
            u.kind, u.text_id, u.shape_id = 'pill', texts[0], shapes[0]
            u.rotation = els[shapes[0]].get('rotation', 0) or 0
            u.pill = _pill_proportions(els[texts[0]], els[shapes[0]])
        else:
            u.kind = 'rigid'
        units.append(u)

    _assign_roles(units, els, sw, sh)

    # ---- reading order: rows by vertical overlap, left to right within a row
    units.sort(key=lambda u: (u.box[1] + u.box[3] / 2, u.box[0]))
    rows = []
    for u in units:
        placed = False
        for row in rows:
            ry0 = min(v.box[1] for v in row)
            ry1 = max(v.box[1] + v.box[3] for v in row)
            ov = min(ry1, u.box[1] + u.box[3]) - max(ry0, u.box[1])
            if ov < 0.5 * min(u.box[3], ry1 - ry0):
                continue
            if any(min(v.box[0] + v.box[2], u.box[0] + u.box[2]) - max(v.box[0], u.box[0]) > 0 for v in row):
                continue
            row.append(u)
            placed = True
            break
        if not placed:
            rows.append([u])
    rows.sort(key=lambda r: min(v.box[1] for v in r))
    for row in rows:
        row.sort(key=lambda v: v.box[0])
    order = 0
    for row in rows:
        for u in row:
            u.order = order
            order += 1
    units.sort(key=lambda u: u.order)
    for k, u in enumerate(units):
        u.uid = k

    an.els = els
    an.boxes = boxes
    an.backgrounds = backgrounds
    an.decorations = decorations
    an.units = units
    an.rows = rows
    an.unit_of = {i: u for u in units for i in u.ids}
    return an


def _pill_ok(t, s):
    """A label centred on a rectangle that fits it like a button."""
    if s.get('shape_type', 'rectangle') != 'rectangle':
        return False
    tr, sr = t.get('rotation', 0) or 0, s.get('rotation', 0) or 0
    tc = (t['x'] + t['width'] / 2, t['y'] + t['height'] / 2)
    sc = (s['x'] + s['width'] / 2, s['y'] + s['height'] / 2)
    if tr != sr:
        return False
    if abs(tc[0] - sc[0]) > 0.2 * s['width'] + 2:
        return False
    if abs(tc[1] - sc[1]) > 0.25 * s['height'] + 2:
        return False
    if '\n' in str(t.get('content', '')):
        return False
    return s['height'] <= 5.0 * float(t['font_size'])


def _pill_proportions(t, s):
    f = float(t['font_size'])
    natural = natural_width(t, int(round(f)))
    rh = s['height'] / f
    rw = max(0.0, (s['width'] - natural) / 2 / f)
    radius = float(s.get('corner_radius', 0) or 0)
    return {'rh': rh, 'rw': rw, 'round': radius >= 0.42 * s['height'],
            'radius_ratio': radius / max(s['height'], 1e-6)}


def _assign_roles(units, els, sw, sh):
    area = sw * sh
    for u in units:
        x, y, w, h = u.box
        near_l, near_r = x <= 0.12 * sw, x + w >= 0.88 * sw
        near_t, near_b = y <= 0.12 * sh, y + h >= 0.88 * sh
        cx = x + w / 2
        u.side = 'left' if cx < 0.33 * sw else 'right' if cx > 0.67 * sw else 'center'
        if (near_l != near_r) and (near_t != near_b) and w * h <= 0.08 * area:
            u.corner = ('t' if near_t else 'b') + ('l' if near_l else 'r')

    texts = [u for u in units if u.kind == 'text']
    headline = None
    if texts:
        def prominence(u):
            e = els[u.text_id]
            bold = _norm_weight(e.get('font_weight')) in _BOLD_WEIGHTS
            return (u.font * (1.12 if bold else 1.0), -u.box[1])
        headline = max(texts, key=prominence)
    top_font = max((u.font for u in units if u.kind in ('text', 'rigid')), default=0)
    # Fine print: clearly smaller than the other free-standing copy *and* the
    # headline, and either hugging the top/bottom edge or tiny outright.
    others = sorted(u.font for u in texts if u is not headline)
    body_ref = others[len(others) // 2] if others else 0
    for u in texts:
        if u is headline:
            u.role = 'headline'
            continue
        e = els[u.text_id]
        cy = u.box[1] + u.box[3] / 2
        smaller = u.font <= 0.7 * body_ref and u.font <= 0.4 * top_font
        edge = cy >= 0.8 * sh or cy <= 0.15 * sh or u.font <= 0.3 * top_font
        u.role = 'fine' if smaller and edge and len(str(e.get('content', ''))) > 12 else 'body'

    pills = [u for u in units if u.kind == 'pill']
    flat = [u for u in pills if not u.rotation]
    cta = None
    if flat:
        def size_of(u):
            return els[u.shape_id]['width'] * els[u.shape_id]['height']
        ranked = sorted(flat, key=size_of, reverse=True)
        if len(ranked) == 1:
            cta = ranked[0] if ranked[0].corner is None else None
        elif size_of(ranked[0]) >= 1.25 * size_of(ranked[1]):
            cta = ranked[0]
    for u in pills:
        u.role = 'tag' if u.rotation else 'cta' if u is cta else 'badge'

    images = [u for u in units if u.kind == 'image']
    hero = None
    if images:
        cand = sorted(images, key=lambda u: (not u.missing, u.box[2] * u.box[3]), reverse=True)
        if cand[0].box[2] * cand[0].box[3] >= 0.04 * area:
            hero = cand[0]
    logo = None
    small = [u for u in images if u is not hero and u.box[2] * u.box[3] <= 0.12 * area and not u.missing]
    if small:
        def logo_rank(u):
            edge = min(u.box[1], sh - u.box[1] - u.box[3]) / sh
            aspect = u.box[2] / max(u.box[3], 1e-6)
            return edge - (0.08 if aspect >= 1.6 else 0)
        best = min(small, key=logo_rank)
        if logo_rank(best) <= 0.25:
            logo = best
    for u in images:
        if u is hero:
            u.role = 'hero'
        elif u is logo:
            u.role = 'logo'
        elif u.box[2] * u.box[3] <= 0.06 * area and (u.corner or u.missing):
            u.role = 'stamp'
        else:
            u.role = 'image'

    for u in units:
        if u.kind == 'rigid':
            has_text = any(els[i]['type'] == 'text' for i in u.ids)
            only_shapes = all(els[i]['type'] == 'shape' for i in u.ids)
            if only_shapes:
                u.role = 'shape'
                x, y, w, h = u.box
                if (min(w, h) <= 0.02 * min(sw, sh) and max(w, h) >= 4 * max(min(w, h), 1)) or \
                        all(els[i].get('shape_type') == 'line' for i in u.ids):
                    u.role = 'divider'
                    u.lane = 'C'
            elif has_text and headline is not None and u.font >= 0.9 * headline.font:
                u.role = 'group'
                u.lane = 'C'
            else:
                u.role = 'group'

    for u in units:
        if u.role in ('headline', 'body'):
            u.lane = 'C'
        elif u.role == 'fine':
            u.lane = 'F'
        elif u.role == 'cta':
            u.lane = 'A'
        elif u.role == 'badge':
            u.lane = 'D'
        elif u.role == 'logo':
            u.lane = 'B'
        elif u.role in ('tag', 'stamp'):
            u.lane = 'X'
        elif u.role == 'divider':
            u.lane = 'C'
        elif u.lane != 'C':
            u.lane = 'V'
        u.floater = u.role in ('tag', 'stamp') or (u.role == 'badge' and u.corner is not None)


# ==========================================================================
# Blocks: units materialised at a size, in local coordinates
# ==========================================================================

class Block:
    __slots__ = ('w', 'h', 'els', 'unit', 'font', 'kind', 'scale')

    def __init__(self, w, h, els, unit=None, font=0.0, kind='', scale=1.0):
        self.w, self.h, self.els = w, h, els
        self.unit, self.font, self.kind, self.scale = unit, font, kind, scale


def _floor2(v):
    return math.floor(v * 100 + 1e-7) / 100


def _ceil2(v):
    return math.ceil(v * 100 - 1e-7) / 100


class Ctx:
    def __init__(self, source, target, an):
        self.an = an
        self.els = an.els
        self.cw, self.ch = float(target['width']), float(target['height'])
        self.sw, self.sh = an.sw, an.sh
        cw, ch = self.cw, self.ch
        self.short, self.long = min(cw, ch), max(cw, ch)
        self.m = max(min(6.0, 0.08 * self.short), 0.04 * self.short)
        self.floor_mult = 1.0
        self.organization_groups = []
        for roles in ({'badge'}, {'logo', 'cta', 'badge'}):
            related = [u for u in an.units if u.role in roles]
            spacing = roles == {'badge'}
            members = [[u] for u in related if spacing or u.role != 'badge']
            badges = [u for u in related if u.role == 'badge']
            if not spacing and badges:
                # Evaluate the badge cluster as a whole against brand/action;
                # individual badges must not mask a misaligned logo or CTA.
                members.append(badges)
            if len(members) >= 3:
                source_cost = _arrangement_cost(tuple(tuple(_union(u.box for u in group))
                                                      for group in members), spacing)
                # Only preserve organization for which the source is evidence.
                if source_cost <= 0.35:
                    self.organization_groups.append((tuple(tuple(i for u in group for i in u.ids)
                                                           for group in members), source_cost, spacing))
        self.mx = max(self.m, min(0.02 * cw, 2.5 * self.m))
        self.my = self.m
        self.gap = max(self.m * 0.5, min(0.022 * cw, 2.5 * self.m)) if cw >= ch else max(self.m * 0.5, 0.02 * ch)
        self.gmin = max(2.0, 0.25 * self.m)
        rx, ry = cw / self.sw, ch / self.sh
        self.s_txt = max(rx, ry)
        self.s_img = math.sqrt(rx * ry)
        self.aspect = cw / ch
        self.src_aspect = self.sw / self.sh
        L = self.long
        self.soft = {'headline': max(12.0, 0.024 * L), 'cta': max(9.0, 0.013 * L),
                     'body': max(9.0, 0.011 * L), 'badge': max(7.0, 0.008 * L),
                     'tag': max(7.0, 0.008 * L), 'fine': max(6.0, 0.0065 * L),
                     'label': max(8.0, 0.01 * L)}

    # -- sizing policy -------------------------------------------------------
    def tcap(self, role, src_font):
        """Source-scaled size cap, raised toward the role preference if needed.

        Any increase is limited to 1.8 times the source-scaled base size.
        """
        base = src_font * self.s_txt
        return max(base, min(self.soft.get(role, 8.0), 1.8 * base))

    def floor(self, role, src_font):
        floor = max(1, int(round(TEXT_FLOOR.get(role, 6) * self.floor_mult)))
        return min(floor, max(1, int(self.tcap(role, src_font))))

    def tsize(self, role, src_font, p):
        e = TEXT_ELASTICITY.get(role, 1.1)
        f = self.tcap(role, src_font) * (max(p, 0.0) ** e)
        return max(self.floor(role, src_font), int(math.floor(f + 1e-9)))

    def vscale(self, u, p, r):
        return self.s_img * r * (max(p, 0.02) ** 0.9)


def _text_unit_block(ctx, u, size, width, shrink=True):
    """Fit text starting at `size`; width=None uses its natural line width.

    Reduce font size if needed for width/line limits. By default, also try to
    tighten the resulting box width without changing its measured line count.
    """
    src = ctx.els[u.text_id]
    e = dict(src)
    k = size / max(float(src['font_size']), 1e-6)
    if src.get('letter_spacing'):
        e['letter_spacing'] = float(src['letter_spacing']) * k
    if width is None:
        width = natural_width(e, size) + 1.0
    width = _floor2(width)
    fit = _text_ok(e, width, size)
    while fit is None and size > 1:
        size -= 1
        if src.get('letter_spacing'):
            e['letter_spacing'] = float(src['letter_spacing']) * size / max(float(src['font_size']), 1e-6)
        fit = _text_ok(e, width, size)
        if size < max(1, ctx.floor(u.role, float(src['font_size'])) - 2):
            return None
    if fit is None:
        return None
    if shrink:
        width, fit = _shrink_wrap(e, width, size)
    e['font_size'] = size
    e['width'] = width
    e['height'] = _ceil2(fit.height)
    e['x'], e['y'] = 0.0, 0.0
    return Block(width, e['height'], [(u.text_id, e)], u, size, 'text')


def _pill_block(ctx, u, size, compact, max_w=None):
    t, s = ctx.els[u.text_id], ctx.els[u.shape_id]
    pr = u.pill
    rh = _clamp(pr['rh'], 1.6, 2.0) if compact else _clamp(pr['rh'], 1.7, 3.6)
    rw = _clamp(pr['rw'], 0.55, 0.9) if compact else _clamp(pr['rw'], 0.6, 1.8)
    e = dict(t)
    k = size / max(float(t['font_size']), 1e-6)
    if t.get('letter_spacing'):
        e['letter_spacing'] = float(t['letter_spacing']) * k
    pad_x = rw * size
    inner = _floor2(natural_width(e, size) + 1.0)
    if max_w is not None and inner + 2 * pad_x > max_w:
        inner = _floor2(max_w - 2 * pad_x)
        if inner <= size:
            return None
    fit = _text_ok(e, inner, size)
    if fit is None:
        return None
    th = _ceil2(fit.height)
    if fit.lines == 1:
        H = max(rh * size, th + 0.2 * size)
    else:
        H = th + 2 * max(0.35 * size, (rh - 1.4) / 2 * size)
    W = inner + 2 * pad_x
    if fit.lines == 1 and not u.rotation:
        ty = H / 2 + fit.cap / 2 - fit.asc
        ty = _clamp(ty, 0.0, H - th)
    else:
        ty = (H - th) / 2
    shape = dict(s)
    shape.update(width=W, height=H)
    if 'corner_radius' in s or pr['round']:
        shape['corner_radius'] = H / 2 if pr['round'] else pr['radius_ratio'] * H
    if s.get('stroke_width'):
        shape['stroke_width'] = float(s['stroke_width']) * k
    e.update(font_size=size, width=inner, height=th)
    rot = u.rotation
    bx, by, bw, bh = _rot_box(0, 0, W, H, rot)
    ox, oy = -bx, -by
    shape['x'], shape['y'] = ox, oy
    e['x'], e['y'] = ox + pad_x, oy + ty
    return Block(bw, bh, [(u.shape_id, shape), (u.text_id, e)], u, size, 'pill')


def _scaled_block(ctx, u, scale):
    """Image or rigid group scaled uniformly; contained text is re-fitted."""
    ux, uy, uw, uh = u.box
    if uw <= 0 or uh <= 0 or scale <= 0:
        return None
    items = []
    min_font = None
    for i in u.ids:
        src = ctx.els[i]
        e = dict(src)
        e['x'] = (src['x'] - ux) * scale
        e['y'] = (src['y'] - uy) * scale
        e['width'] = src['width'] * scale
        e['height'] = src['height'] * scale
        for key in ('corner_radius', 'stroke_width'):
            if src.get(key):
                e[key] = float(src[key]) * scale
        if src['type'] == 'text':
            size = max(1, int(math.floor(float(src['font_size']) * scale + 1e-9)))
            if src.get('letter_spacing'):
                e['letter_spacing'] = float(src['letter_spacing']) * scale
            src_fit = measure(src, src['width'], int(src['font_size']))
            target_lines = src_fit.lines
            limit = max(e['height'], src_fit.height * scale)
            fit = measure(e, _floor2(e['width']), size)
            while size > 1 and (fit.max_w > _floor2(e['width']) + 1e-6 or fit.lines > target_lines
                                or fit.height > limit + 1e-6):
                size -= 1
                fit = measure(e, _floor2(e['width']), size)
            e['font_size'] = size
            e['width'] = _floor2(e['width'])
            # Use measured text height rather than loose source-box height
            # when calculating block extents and candidate collisions.
            e['height'] = _ceil2(fit.height)
            min_font = size if min_font is None else min(min_font, size)
        items.append((i, e))
    bx, by, bw, bh = _union(_el_box(e) for _, e in items)
    for _, e in items:
        e['x'] -= bx
        e['y'] -= by
    return Block(bw, bh, items, u, float(min_font or 0), 'vis', scale)


def _flow(blocks, width, gap):
    """Wrap nonempty blocks into centred rows using `width` as the row limit.

    Individual oversized blocks are not shrunk here; callers must check fit.
    """
    rows, row, used = [], [], 0.0
    for b in blocks:
        need = b.w if not row else used + gap + b.w
        if row and need > width + 1e-6:
            rows.append(row)
            row, used = [b], b.w
        else:
            row.append(b)
            used = need
    if row:
        rows.append(row)
    total_w = max(sum(b.w for b in r) + gap * (len(r) - 1) for r in rows)
    items, y = [], 0.0
    vgap = gap * 0.6
    for r in rows:
        rh = max(b.h for b in r)
        rw = sum(b.w for b in r) + gap * (len(r) - 1)
        x = (total_w - rw) / 2
        for b in r:
            for i, e in b.els:
                e2 = dict(e)
                e2['x'] += x
                e2['y'] += y + (rh - b.h) / 2
                items.append((i, e2))
            x += b.w + gap
        y += rh + vgap
    font = max((b.font for b in blocks), default=0)
    return Block(total_w, y - vgap, items, None, font, 'flow')


def _shift(block, dx, dy, out):
    for i, e in block.els:
        e2 = dict(e)
        e2['x'] = e['x'] + dx
        e2['y'] = e['y'] + dy
        out[i] = e2


# ==========================================================================
# Pieces and stacks
# ==========================================================================

class Opts:
    def __init__(self, compact=False, r=1.0, r_logo=1.0, per_row=None, fine_strip=False, q=1.0,
                 grid_per=None, sizes=None):
        self.sizes = sizes if sizes is not None else {}  # explicit size per unit uid (zones)
        self.q = q  # pressure multiplier for action-lane pills and floaters
        self.grid_per = grid_per  # visuals per row in a grid piece (None: one row)
        self.compact = compact
        self.r = r
        self.r_logo = r_logo
        self.per_row = per_row
        self.fine_strip = fine_strip


def _vis_limit(ctx, u, opts, p):
    r = opts.r_logo if u.role == 'logo' else opts.r
    if u.role == 'stamp':
        r = min(r, 1.0)
    if u.kind == 'rigid' and any(ctx.els[i]['type'] == 'text' for i in u.ids):
        role = 'headline' if u.lane == 'C' else 'body'
        return ctx.tsize(role, u.font, p) / max(u.font, 1e-6)
    return ctx.vscale(u, p, r)


def _lane_p(u, p, opts):
    if u.lane in ('A', 'D', 'X'):
        return min(1.0, p * opts.q)
    return p


def _make_piece(ctx, piece, p, opts, width, hmax):
    """Materialise a column piece. `width` None means natural width."""
    kind, payload = piece
    if kind == 'text':
        u = payload
        size = opts.sizes.get(u.uid) or ctx.tsize(u.role, u.font, p)
        return _text_unit_block(ctx, u, size, width)
    if kind == 'pill':
        u = payload
        size = opts.sizes.get(u.uid) or ctx.tsize(u.role, u.font, _lane_p(u, p, opts))
        b = _pill_block(ctx, u, size, opts.compact, width)
        while b is None and size > ctx.floor(u.role, u.font):
            size -= 1
            b = _pill_block(ctx, u, size, opts.compact, width)
        return b
    if kind == 'flow':
        blocks = []
        for u in payload:
            size = opts.sizes.get(u.uid) or ctx.tsize(u.role, u.font, _lane_p(u, p, opts))
            b = _pill_block(ctx, u, size, opts.compact, width)
            if b is None:
                return None
            blocks.append(b)
        g = max(ctx.gmin, 0.5 * max(b.font for b in blocks))
        if width is None:
            k = opts.per_row or len(blocks)
            # Natural width: the widest of the rows produced by `k` per row.
            width = max(sum(b.w for b in blocks[i:i + k]) + g * (len(blocks[i:i + k]) - 1)
                        for i in range(0, len(blocks), k))
        return _flow(blocks, width, g)
    if kind == 'grid':
        return _grid_block(ctx, payload, min(_vis_limit(ctx, u, opts, p) for u in payload),
                           opts, width, hmax)
    if kind == 'vis':
        u = payload
        scale = _vis_limit(ctx, u, opts, p)
        if width is not None:
            scale = min(scale, width / max(u.box[2], 1e-6))
        if hmax is not None:
            scale = min(scale, hmax / max(u.box[3], 1e-6))
        if u.role == 'logo' and hmax is not None:
            scale = min(scale, 0.62 * hmax / max(u.box[3], 1e-6)) if hmax < 0.3 * ctx.long else scale
        return _scaled_block(ctx, u, scale)
    raise ValueError(kind)


def _similar_visuals(units):
    """Split a run of units into runs of >= 3 similar visuals (grid) or singles."""
    out, k = [], 0
    while k < len(units):
        u = units[k]
        if u.kind in ('image', 'rigid') and u.role not in ('divider', 'logo'):
            j = k
            area0 = max(u.box[2] * u.box[3], 1.0)
            while j + 1 < len(units):
                v = units[j + 1]
                a = max(v.box[2] * v.box[3], 1.0)
                if v.kind not in ('image', 'rigid') or v.role in ('divider', 'logo') or not (0.4 <= a / area0 <= 2.5):
                    break
                j += 1
            if j - k + 1 >= 3:
                out.append(('grid', units[k:j + 1]))
                k = j + 1
                continue
        out.append(('one', u))
        k += 1
    return out


def _grid_block(ctx, units, scale, opts, width, hmax, exact=False):
    """Scale visuals uniformly into centred rows using opts.grid_per or one row."""
    n = len(units)
    per = min(n, opts.grid_per or n)
    rows = [units[i:i + per] for i in range(0, n, per)]
    g = max(ctx.gmin, 0.5 * ctx.gap)
    if not exact:
        row_w = max(sum(u.box[2] for u in r) for r in rows)
        row_h = max(u.box[3] for u in units)
        if width is not None:
            scale = min(scale, (width - g * (per - 1)) / max(row_w, 1e-6))
        if hmax is not None:
            scale = min(scale, (hmax - g * (len(rows) - 1)) / max(row_h * len(rows), 1e-6))
    if scale <= 0:
        return None
    blocks = []
    for u in units:
        b = _scaled_block(ctx, u, scale)
        if b is None:
            return None
        blocks.append(b)
    total_w = max(sum(b.w for b in blocks[i:i + per]) + g * (len(blocks[i:i + per]) - 1)
                  for i in range(0, n, per))
    items, y = [], 0.0
    for i in range(0, n, per):
        row = blocks[i:i + per]
        rh = max(b.h for b in row)
        x = (total_w - (sum(b.w for b in row) + g * (len(row) - 1))) / 2
        for b in row:
            for j, e in b.els:
                e2 = dict(e)
                e2['x'] += x
                e2['y'] += y + (rh - b.h) / 2
                items.append((j, e2))
            x += b.w + g
        y += rh + g
    blk = Block(total_w, y - g, items, None, 0.0, 'grid', scale)
    blk.unit = units
    return blk


def _pieces_for(units):
    pieces, run = [], []
    seq = []
    for kind, payload in _similar_visuals(units):
        seq.append(payload if kind == 'one' else ('grid', payload))
    for u in seq:
        if isinstance(u, tuple):
            if run:
                pieces.append(('flow', run) if len(run) > 1 else ('pill', run[0]))
                run = []
            pieces.append(u)
            continue
        if u.kind == 'pill' and u.role == 'badge' and not u.rotation:
            run.append(u)
            continue
        if run:
            pieces.append(('flow', run) if len(run) > 1 else ('pill', run[0]))
            run = []
        if u.kind == 'text':
            pieces.append(('text', u))
        elif u.kind == 'pill':
            pieces.append(('pill', u))
        else:
            pieces.append(('vis', u))
    if run:
        pieces.append(('flow', run) if len(run) > 1 else ('pill', run[0]))
    return pieces


def _piece_gap(ctx, a, b, H):
    fa = a.font if a.kind in ('text', 'pill', 'flow') else 0
    fb = b.font if b.kind in ('text', 'pill', 'flow') else 0
    if not fa and not fb:
        return max(ctx.gmin, 0.06 * H)
    g = 0.28 * (fa + fb) if fa and fb else 0.45 * (fa or fb)
    if a.kind == 'text' and b.kind == 'text':
        g *= 0.85
    return max(ctx.gmin, g)


def _column(ctx, pieces, p, opts, width, H, align='center'):
    """Stack pieces vertically. Visuals shrink to make the column fit H."""
    blocks = [None] * len(pieces)
    vis_idx = []
    for k, pc in enumerate(pieces):
        if pc[0] in ('vis', 'grid'):
            vis_idx.append(k)
            continue
        b = _make_piece(ctx, pc, p, opts, width, H)
        if b is None:
            return None
        blocks[k] = b
    for k in vis_idx:
        b = _make_piece(ctx, pieces[k], p, opts, width, H)
        if b is None:
            return None
        blocks[k] = b

    def total():
        gaps = sum(_piece_gap(ctx, blocks[k], blocks[k + 1], H) for k in range(len(blocks) - 1))
        return sum(b.h for b in blocks) + gaps

    t = total()
    if t > H + 1e-6 and vis_idx:
        vis_h = sum(blocks[k].h for k in vis_idx)
        fixed = t - vis_h
        room = H - fixed
        if room <= 0.12 * H:
            return None
        factor = room / vis_h
        for k in vis_idx:
            if pieces[k][0] == 'grid':
                nb = _grid_block(ctx, pieces[k][1], blocks[k].scale * factor * 0.999, opts, None, None, True)
            else:
                nb = _scaled_block(ctx, pieces[k][1], blocks[k].scale * factor * 0.999)
            if nb is None:
                return None
            blocks[k] = nb
        t = total()
    if t > H + 1e-6:
        return None
    colw = width if width is not None else max(b.w for b in blocks)
    if any(b.w > colw + 1e-6 for b in blocks):
        return None
    items, y = [], 0.0
    for k, b in enumerate(blocks):
        if align == 'left':
            x = 0.0
        elif align == 'right':
            x = colw - b.w
        else:
            x = (colw - b.w) / 2
        for i, e in b.els:
            e2 = dict(e)
            e2['x'] += x
            e2['y'] += y
            items.append((i, e2))
        y += b.h
        if k < len(blocks) - 1:
            y += _piece_gap(ctx, b, blocks[k + 1], H)
    return Block(colw, t, items, None, max((b.font for b in blocks), default=0), 'column')


# ==========================================================================
# Layout candidates
# ==========================================================================

def _search(build, iters=14):
    """Binary-search a feasible pressure, assuming feasibility is monotone.

    Return the best layout sampled, including p=0, or None if that also fails.
    """
    top = build(1.0)
    if top is not None:
        return top
    low = build(0.0)
    if low is None:
        return None
    lo, hi, best = 0.0, 1.0, low
    for _ in range(iters):
        mid = (lo + hi) / 2
        res = build(mid)
        if res is not None:
            lo, best = mid, res
        else:
            hi = mid
    return best


# ---------------- horizontal: lanes in columns ----------------------------

def _lanes(an):
    lanes = {k: [] for k in 'BVCADFX'}
    for u in an.units:
        lanes['X' if u.floater else u.lane].append(u)
    return lanes


def _column_templates(lanes):
    """Column orders built from lanes: B brand, V visual, C copy, A action
    (CTA), D badges, F fine print, X floaters. Badges may join the action
    column, sit under the copy, or take their own column."""
    base = _base_templates(lanes)
    if not lanes['D']:
        return base
    out, seen = [], set()
    for cols in base:
        variants = []
        has_a = any('A' in c for c in cols)
        if has_a:
            variants.append([c + ['D'] if 'A' in c else c for c in cols])
        variants.append([c + ['D'] if 'C' in c else c for c in cols])
        k = next((i for i, c in enumerate(cols) if 'A' in c), len(cols) - 1)
        variants.append(cols[:k] + [['D']] + cols[k:])
        for v in variants:
            sig = tuple(tuple(c) for c in v)
            if sig not in seen:
                seen.add(sig)
                out.append(v)
    return out


def _base_templates(lanes):
    present = {k for k, v in lanes.items() if v} - {'D'}
    pat = [
        'B|V|C|A', 'V|B|C|A', 'BV|C|A', 'V|BC|A', 'V|BCA', 'BCA|V', 'BV|CA',
        'V|C|BA', 'B|C|VA', 'V|CA', 'B|CA|V', 'BC|A|V', 'B|VC|A', 'V|C|A|B',
        'BCAV', 'C|A',
    ]
    seen, out = set(), []
    for t in pat:
        cols = []
        for col in t.split('|'):
            lanes_in = [c for c in col if c in present]
            if lanes_in:
                cols.append(lanes_in)
        used = {c for col in cols for c in col}
        missing = (present - {'F', 'X'}) - used
        if missing:
            continue
        sig = tuple(tuple(c) for c in cols)
        if sig in seen or not cols:
            continue
        seen.add(sig)
        out.append(cols)
    return out


def _build_columns(ctx, cols, p, opts, floaters_inline):
    lanes = ctx.lanes
    W = ctx.cw - 2 * ctx.mx
    top, bottom = ctx.my, ctx.ch - ctx.my
    out = {}
    fine = lanes['F']
    col_units = []
    for col in cols:
        us = []
        for lane in col:
            us += lanes[lane]
        col_units.append(us)
    inline = [u for u in lanes['X']] if floaters_inline else []
    if inline:
        # Floaters folded into the action column (or the last column).
        target = next((k for k, col in enumerate(cols) if 'A' in col), len(cols) - 1)
        col_units[target] = col_units[target] + inline
    strip = None
    if fine:
        if opts.fine_strip:
            strip = _column(ctx, [('text', u) for u in fine], p, opts, W, bottom - top, 'center')
            if strip is None:
                return None
            bottom -= strip.h + ctx.gmin * 1.5
        else:
            target = next((k for k, col in enumerate(cols) if 'C' in col), len(cols) - 1)
            col_units[target] = col_units[target] + fine
    H = bottom - top
    if H <= 4:
        return None
    flex = [k for k, col in enumerate(cols) if 'C' in col]
    if not flex:
        flex = [max(range(len(cols)), key=lambda k: sum(1 for u in col_units[k] if u.kind == 'text'))] \
            if any(u.kind == 'text' for us in col_units for u in us) else []
    aligns = []
    for us in col_units:
        al = [u.align for u in us if u.kind == 'text']
        aligns.append(max(set(al), key=al.count) if al else 'center')
    gap = ctx.gap
    built = [None] * len(cols)
    for k in range(len(cols)):
        if k in flex:
            continue
        c = _column(ctx, _pieces_for(col_units[k]), p, opts, None, H, aligns[k])
        if c is None:
            return None
        built[k] = c
    fixed_w = sum(c.w for c in built if c is not None)
    room = W - fixed_w - gap * (len(cols) - 1)
    if flex:
        each = room / len(flex)
        if each < 3 * ctx.soft['body']:
            return None
        cap = 0.62 * ctx.cw if len(cols) > 1 else W
        each = min(each, cap)
        for k in flex:
            c = _column(ctx, _pieces_for(col_units[k]), p, opts, each, H, aligns[k])
            if c is None:
                return None
            built[k] = c
    used = sum(c.w for c in built) + gap * (len(cols) - 1)
    if used > W + 1e-6:
        return None
    extra = W - used
    # Spare width becomes breathing room between columns (and a little at the
    # outer edges), so the composition stays balanced instead of left-heavy.
    slots = len(cols) + 1
    edge = extra / slots * 0.5
    between = (extra - 2 * edge) / max(1, len(cols) - 1) if len(cols) > 1 else 0
    if len(cols) == 1:
        edge = extra / 2
    x = ctx.mx + edge
    for k, c in enumerate(built):
        y = top + (H - c.h) / 2
        _shift(c, x, y, out)
        x += c.w + gap + between
    if strip is not None:
        _shift(strip, ctx.mx, ctx.ch - ctx.my - strip.h, out)
    return out


# ---------------- extreme banners: three-zone solver ----------------------
#
# Eligible strips (aspect >= 4, one visual unit and standalone copy) add zone
# candidates alongside generic columns. Hero, copy and brand/action regions
# compete for width. Short strips try the hero before or after copy, several
# hero sizes/gutters, parallel copy columns and compact badge/header packing.
# Hero growth can then use nearby free space. Brand/action regions can stack
# or split; tags/stamps are either slotted or placed in remaining free space.
# Candidate validation and scoring decide which sampled arrangement wins.

BANNER_HEADLINE_TARGET = 14   # px: preferred minimum headline size on strips
BANNER_CTA_TARGET = 11        # px: preferred minimum CTA label size on strips
BANNER_COPY_FIRST_BONUS = 0.6  # composition preference, below headline/body shortfall penalties


def _zone_eligible(ctx):
    lanes = ctx.lanes
    return (ctx.aspect >= 4.0 and len(lanes['V']) == 1
            and lanes['V'][0].kind in ('image', 'rigid')
            and any(u.kind == 'text' for u in lanes['C']))


def _vstack(blocks, gaps, align='center'):
    """Stack blocks vertically with explicit gaps; returns one Block."""
    width = max(b.w for b in blocks)
    items, y = [], 0.0
    for k, b in enumerate(blocks):
        x = 0.0 if align == 'left' else width - b.w if align == 'right' else (width - b.w) / 2
        for i, e in b.els:
            e2 = dict(e)
            e2['x'] += x
            e2['y'] += y
            items.append((i, e2))
        y += b.h + (gaps[k] if k < len(gaps) else 0.0)
    return Block(width, y, items, None, max(b.font for b in blocks), 'stack')


def _zone_copy(ctx, units, width, H, align, cache, split=None):
    """Fit zone copy as a stack or split columns, then greedily grow text.

    1. Search a fitting common pressure using role elasticity and size floors.
    2. Leftover room is shared out one pixel at a time, round-robin: headline,
       body, fine print, badges -- each within a cap relative to the headline.
       This encourages hierarchy without guaranteeing a globally optimal fit.
    """
    key = (tuple(u.uid for u in units), width, H, align, split)
    if key in cache:
        return cache[key]
    texts = [u for u in units if u.kind in ('text', 'pill')]
    head = next((u for u in texts if u.role == 'headline'), None)
    pieces = _pieces_for(units)
    sizes = {}
    opts = Opts(compact=True, sizes=sizes)

    def build():
        if split is None:
            return _column(ctx, pieces, 1.0, opts, width, H, align)
        # Give headline and supporting copy their own vertical space. Fine
        # print spans both columns below them, retaining every original unit.
        if head is None:
            return None
        supporting = [u for u in units if u is not head and u.role != 'fine']
        if not supporting:
            return None
        fine = [u for u in units if u.role == 'fine']
        footer = _column(ctx, _pieces_for(fine), 1.0, opts, width, H, align) if fine else None
        if fine and footer is None:
            return None
        gap = max(ctx.gmin, 0.06 * H)
        room = H - footer.h - gap if footer else H
        if room <= 0 or width <= gap:
            return None
        left_w = (width - gap) * split
        left = _column(ctx, _pieces_for([head]), 1.0, opts, left_w, room, align)
        right = _column(ctx, _pieces_for(supporting), 1.0, opts,
                        width - gap - left_w, room, align)
        if left is None or right is None:
            return None
        row = _hrow([left, right], gap)
        return _vstack([row, footer], [gap], align) if footer else row

    def at(p):
        for u in texts:
            sizes[u.uid] = ctx.tsize(u.role, u.font, p)
        return build()

    best = at(1.0)
    if best is None:
        best = at(0.0)
        if best is None:
            cache[key] = None
            return None
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            if at(mid) is not None:
                lo = mid
            else:
                hi = mid
        best = at(lo)
    caps = {'headline': 1.0, 'body': 0.62, 'fine': 0.48, 'badge': 0.55}
    if ctx.aspect >= 4.0 and ctx.ch <= 120:
        # Compress the typographic hierarchy on short banners: supporting
        # copy can use its available space independently of headline growth.
        # Relative caps limit further body growth; measurements still gate fit.
        caps['body'] = 0.85
    order = sorted(texts, key=lambda u: list(caps).index(u.role) if u.role in caps else 9)
    grow = [u for u in order if u.role in caps]
    while grow:
        still = []
        for u in grow:
            limit = int(ctx.tcap(u.role, u.font))
            if u.role == 'headline':
                limit = min(limit, int(0.34 * ctx.ch))
            elif head is not None:
                limit = min(limit, max(sizes[u.uid], int(caps[u.role] * sizes[head.uid])))
            if sizes[u.uid] >= limit:
                continue
            sizes[u.uid] += 1
            b = build()
            if b is None:
                sizes[u.uid] -= 1
                continue
            best = b
            still.append(u)
        grow = still
    cache[key] = (best, dict(sizes))
    return cache[key]


def _hrow(blocks, sep, top_align=False):
    """Place blocks side by side, centred vertically or optionally top-aligned."""
    h = max(b.h for b in blocks)
    items, x = [], 0.0
    for b in blocks:
        for i, e in b.els:
            e2 = dict(e)
            e2['x'] += x
            e2['y'] += 0.0 if top_align else (h - b.h) / 2
            items.append((i, e2))
        x += b.w + sep
    return Block(x - sep, h, items, None, max(b.font for b in blocks), 'row')


def _zone_right(ctx, logos, ctas, badges, fine, mode, per_row, logo_h, cta_size, badge_size, H,
                tags=(), stamps=(), slot='head'):
    """Build the brand/action zone as a stack, split columns or compact stack.

    Tags and stamps get designed slots instead of leftover space:
    slot 'head' puts tags beside the logo and stamps in the badge group;
    slot 'col' puts tags at the head of the badge group and stamps under the
    CTA. Compact mode puts stamps in the top-aligned brand/tag header and uses
    tighter gaps with extra clearance for missing-image placeholders. Badges
    can wrap into rows; fine print is appended last when supplied here.
    """
    top = []
    head = []
    compact = mode == 'compact'
    header_missing_height = 0.0
    for u in logos:
        sc = min(logo_h / max(u.box[3], 1e-6), 0.3 * ctx.cw / max(u.box[2], 1e-6),
                 H / max(u.box[3], 1e-6))
        b = _scaled_block(ctx, u, sc)
        if b is None:
            return None
        head.append(b)
    tag_blocks = []
    for u in tags:
        size = max(ctx.floor(u.role, u.font), badge_size or 0,
                   0 if compact else (cta_size or 0) - 2)
        b = _pill_block(ctx, u, size, True) if u.kind == 'pill' else None
        if b is None:
            return None
        tag_blocks.append(b)
    if slot == 'head':
        head += tag_blocks
    if compact:
        # Put small corner assets beside the brand, leaving the lower rows
        # for action and wrapped badges. Top alignment keeps a diagnostic
        # image clear of the CTA even when the rotated tag is taller.
        for u in stamps:
            b = _scaled_block(ctx, u, 0.18 * ctx.ch / max(u.box[3], 1e-6))
            if b is None:
                return None
            head.append(b)
            if u.missing:
                header_missing_height = max(header_missing_height, b.h)
    if head:
        # Brand and its sale tag share the top row, clear of the CTA below.
        sep = max(ctx.gmin * 2, 0.6 * (cta_size or 8))
        if header_missing_height:
            sep = max(sep, min(12.0, MISSING_ASSET_GAP_FRACTION * ctx.short))
        top.append(_hrow(head, sep, top_align=compact) if len(head) > 1 else head[0])
    for u in ctas:
        b = (_pill_block(ctx, u, cta_size, True) if u.kind == 'pill'
             else _text_unit_block(ctx, u, cta_size, None))
        if b is None:
            return None
        top.append(b)
    stamp_h = 0.22 * ctx.ch
    missing_gaps = []
    if slot == 'col':
        for u in stamps:
            b = _scaled_block(ctx, u, stamp_h * 0.8 / max(u.box[3], 1e-6))
            if b is None:
                return None
            if u.missing and top:
                missing_gaps.append(len(top) - 1)
            top.append(b)
    gaps = [ctx.gmin if compact else max(ctx.gmin, 0.4 * (cta_size or 8))] * max(0, len(top) - 1)
    if header_missing_height and gaps:
        gaps[0] = max(gaps[0], header_missing_height +
                      min(12.0, MISSING_ASSET_GAP_FRACTION * ctx.short) - top[0].h)
    # Unresolved assets render as opaque diagnostic boxes, not transparent
    # artwork. Give them their own visual breathing room; if the stack no
    # longer fits, candidate selection can choose another logo size/layout.
    # The rule adjusts gaps, not text measurements or asset proportions.
    for index in missing_gaps:
        gaps[index] = max(gaps[index], min(12.0, MISSING_ASSET_GAP_FRACTION * ctx.short), 0.8 * (cta_size or 8))
    bblocks = list(tag_blocks) if slot == 'col' else []
    for u in badges:
        b = _pill_block(ctx, u, badge_size, True)
        if b is None:
            return None
        bblocks.append(b)
    row_h = max((b.h for b in bblocks), default=0.18 * ctx.ch)
    if slot == 'head' and not compact:
        for u in stamps:
            b = _scaled_block(ctx, u, min(1.2 * row_h, stamp_h) / max(u.box[3], 1e-6))
            if b is None:
                return None
            bblocks.append(b)
    bg = max(ctx.gmin, 0.5 * (badge_size or 0))
    if not top and not bblocks and not fine:
        return Block(0.0, 0.0, [], None, 0.0, 'empty')
    if mode == 'split' and bblocks and top:
        left = _vstack(top, gaps)
        right = _vstack(bblocks, [bg * 0.6] * (len(bblocks) - 1))
        sep = max(ctx.gmin * 2, 0.012 * ctx.cw)
        h = max(left.h, right.h)
        items = []
        for blk, x in ((left, 0.0), (right, left.w + sep)):
            for i, e in blk.els:
                e2 = dict(e)
                e2['x'] += x
                e2['y'] += (h - blk.h) / 2
                items.append((i, e2))
        blocks = [Block(left.w + sep + right.w, h, items, None, max(left.font, right.font), 'split')]
        bgaps = []
    else:
        blocks, bgaps = list(top), list(gaps)
        if bblocks:
            k = per_row or len(bblocks)
            width = max(sum(b.w for b in bblocks[i:i + k]) + bg * (len(bblocks[i:i + k]) - 1)
                        for i in range(0, len(bblocks), k))
            if blocks:
                # Separate the badge block from the preceding brand/action stack.
                bgaps.append(ctx.gmin if compact else max(ctx.gmin * 1.5, 0.7 * (cta_size or badge_size or 8)))
            blocks.append(_flow(bblocks, width, bg))
    if fine:
        width = max((b.w for b in blocks), default=0.3 * ctx.cw)
        for u in fine:
            fb = _text_unit_block(ctx, u, ctx.floor('fine', u.font), width)
            if fb is None:
                return None
            if blocks:
                bgaps.append(max(ctx.gmin, 0.5 * fb.font))
            blocks.append(fb)
    blk = _vstack(blocks, bgaps)
    return blk if blk.h <= H + 1e-6 else None


def _zone_size_options(ctx, tight):
    lanes = ctx.lanes
    ctas, badges = lanes['A'], lanes['D']
    if ctas:
        lo = max(ctx.floor(u.role, u.font) for u in ctas)
        hi = max(lo, int(min(ctx.tcap('cta', u.font) for u in ctas)))
        if tight:
            want = (lo, BANNER_CTA_TARGET - 1, BANNER_CTA_TARGET, BANNER_CTA_TARGET + 2)
        else:
            want = (hi, int(hi * 0.8), int(hi * 0.65))
        cta_sizes = sorted({_clamp(v, lo, hi) for v in want})
    else:
        cta_sizes = [None]
    if badges:
        lo = max(ctx.floor('badge', u.font) for u in badges)
        hi = max(lo, int(min(ctx.tcap('badge', u.font) for u in badges)))
        want = (lo, lo + 1) if tight else (hi, int(hi * 0.75))
        badge_sizes = sorted({_clamp(v, lo, hi) for v in want})
    else:
        badge_sizes = [None]
    return cta_sizes, badge_sizes


def _zone_configs(ctx, tight):
    """Yield normal zone configurations; compact mode builds its own set."""
    lanes = ctx.lanes
    fine, badges, logos = lanes['F'], lanes['D'], lanes['B']
    n_b = len(badges)
    per_rows = sorted({n_b, max(1, (n_b + 1) // 2), 1}) if n_b > 1 else [None]
    # Sample prominent logo heights before width/height fitting caps apply.
    # The separate compact branch also tries smaller header logos.
    logo_fracs = (0.48, 0.4, 0.32) if tight else (0.34, 0.26, 0.2)
    cta_sizes, badge_sizes = _zone_size_options(ctx, tight)
    float_modes = ('free', 'head', 'col') if lanes['X'] else ('free',)
    layouts = []  # (badge placement, right-zone mode, badges per row)
    for bp in (('right', 'copy') if badges else ('none',)):
        if bp == 'right':
            layouts += [(bp, 'stack', k) for k in per_rows]
            if n_b > 1:
                layouts.append((bp, 'split', None))
        else:
            layouts.append((bp, 'stack', None))
    for fp, (bp, mode, k), lf, cs, bsz, fm in itertools.product(
            ('copy', 'strip', 'right') if fine else ('none',), layouts,
            logo_fracs if logos else (None,), cta_sizes, badge_sizes, float_modes):
        if bp != 'right' and bsz != badge_sizes[0]:
            continue  # badge size only matters when badges sit in the right zone
        yield fp, bp, mode, k, lf, cs, bsz, fm


def _zones_candidates(ctx, compact=False):
    lanes = ctx.lanes
    hero = lanes['V'][0]
    cw, ch = ctx.cw, ctx.ch
    tight = ch <= 120
    g0 = max(0.03 * cw, 2.2 * ctx.m)                 # generous gutters between zones
    mh = ctx.mx
    my = max(4.5, 0.05 * ch) if tight else ctx.my
    copy_units = list(lanes['C'])
    fine, badges, logos, ctas = lanes['F'], lanes['D'], lanes['B'], lanes['A']
    al = [u.align for u in copy_units if u.kind == 'text']
    align = max(set(al), key=al.count) if al else 'center'
    tags = [u for u in lanes['X'] if u.kind == 'pill']
    stamps = [u for u in lanes['X'] if u.kind != 'pill']
    configs = list(_zone_configs(ctx, tight))
    if compact:
        cta_sizes, badge_sizes = _zone_size_options(ctx, tight)
        rows = sorted({max(1, (len(badges) + 1) // 2), max(1, len(badges))})
        configs = [(fp, 'right', 'compact', k, lf, cs, badge_sizes[0], 'head')
                   for fp, k, lf, cs in itertools.product(
                       ('copy', 'strip', 'right') if fine else ('none',),
                       rows, (0.24, 0.28), cta_sizes)]
    copy_cache, strip_cache = {}, {}
    pre = []
    # Keep the original showcase layouts, but also let measured copy compete
    # for space with more compact visuals and gutters on short banners.
    # Scoring still decides whether the extra text width is worth the tradeoff.
    for g, hf, z1 in itertools.product((g0, g0 * 0.85, max(ctx.gmin, g0 * 0.4)) if tight else (g0,),
                                       (0.88, 0.84, 0.8, 0.65, 0.5) if tight else (0.88, 0.76),
                                       (None, 0.22)):
        # Initial left-hero placement; a copy-first variant is also tried below.
        scale = min(hf * ch / max(hero.box[3], 1e-6), 0.3 * cw / max(hero.box[2], 1e-6))
        hb = _scaled_block(ctx, hero, scale)
        if hb is None:
            continue
        w1 = hb.w if z1 is None else max(hb.w, z1 * cw - mh - g)
        if z1 is not None and w1 <= hb.w + 1:
            continue
        x2 = mh + w1 + g
        # Small width changes can remove a whole wrapped line, freeing height
        # for the independently sized body copy without reducing the headline.
        copy_splits = (None,) + tuple(0.45 + 0.025 * i for i in range(9)) if tight else (None,)
        for config, copy_split in itertools.product(configs, copy_splits):
            fp, bp, mode, k, lf, cs, bsz, fm = config
            Hc = Hr = ch - 2 * my
            strip = None
            if fp == 'strip':  # fine print under the centre and right zones
                skey = round(x2)
                if skey not in strip_cache:
                    sb = [_text_unit_block(ctx, u, ctx.floor('fine', u.font), cw - mh - x2) for u in fine]
                    strip_cache[skey] = None if any(b is None for b in sb) else \
                        _vstack(sb, [ctx.gmin] * len(sb))
                strip = strip_cache[skey]
                if strip is None:
                    continue
                Hc = Hr = Hc - strip.h - max(ctx.gmin, 0.4 * strip.font)
            slotted = fm != 'free'
            right = _zone_right(ctx, logos, ctas, badges if bp == 'right' else [],
                                fine if fp == 'right' else [], mode, k, (lf or 0) * ch, cs, bsz, Hr,
                                tags if slotted else (), stamps if slotted else (),
                                fm if slotted else 'head')
            if right is None:
                continue
            x3 = cw - mh - right.w
            w2 = (x3 - g - x2) if right.els else (cw - mh - x2)
            if w2 < 0.25 * cw:
                continue
            cu = copy_units + (badges if bp == 'copy' else []) + (fine if fp == 'copy' else [])
            res = _zone_copy(ctx, cu, w2, Hc, align, copy_cache, split=copy_split)
            if res is None:
                continue
            out = {}
            _shift(hb, mh + (w1 - hb.w) / 2, (ch - hb.h) / 2, out)
            _shift(res[0], x2 + (w2 - res[0].w) / 2, my + (Hc - res[0].h) / 2, out)
            if right.els:
                _shift(right, x3, my + (Hr - right.h) / 2, out)
            if strip is not None:
                _shift(strip, x2, ch - my - strip.h, out)
            kind = 'zones' if fp in ('copy', 'none') else f'zones:fine_{fp}'
            label = (f'zones hero={hf} gutter={g:.0f} z1={z1} fine={fp} badges={bp}/{mode}/k={k} '
                     f'logo={lf} cta={cs} badge={bsz} floaters={fm} copy_split={copy_split}')
            pre.append((_score(ctx, out, kind, organization=False), label, out, slotted, kind, (x3 - 0.25 * g, 0.0, cw, ch)))
            if tight and strip is None:
                # Also read copy first, then the product beside brand/action.
                # Reuse the measured copy blocks so this arrangement never
                # buys product placement by shrinking the text. A spanning
                # fine-print strip needs a different packing and is excluded.
                moved = {i: dict(e) for i, e in out.items()}
                for i, _ in res[0].els:
                    moved[i]['x'] -= w1 + g
                for i, _ in hb.els:
                    moved[i]['x'] += x3 - g - w1 - mh
                right_label = label + ' hero_side=right'
                pre.append((_score(ctx, moved, kind, organization=False), right_label, moved, slotted,
                            kind, (x3 - 0.25 * g, 0.0, cw, ch)))
    # Floaters and full validation only for the most promising layouts.
    pre.sort(key=lambda t: -t[0])
    clearance = max(ctx.gmin, 0.35 * g0)
    results = []
    side_counts = {'left': 0, 'right': 0}
    for _, label, out, slotted, kind, region in pre:
        side = 'right' if 'hero_side=right' in label else 'left'
        if side_counts[side] >= 16:
            continue
        # Slotted tags/stamps already sit in the right zone; otherwise they
        # float, but only inside the brand/action zone, clear of the gutters.
        done = out if slotted else _place_floaters(ctx, dict(out), 1.0, Opts(compact=True),
                                                   clearance, region)
        if done is not None:
            if tight:
                done = _grow_hero_in_free_space(ctx, done)
            results.append((kind, done, Opts(compact=True), label))
            side_counts[side] += 1
            if all(count >= 16 for count in side_counts.values()):
                break
    if tight and not compact:
        # Readable text alone does not prove the first structure is best.
        # Compare compact/wrapped alternatives under the same final scoring.
        results += _zones_candidates(ctx, compact=True)
    return results


# ---------------- anchored: keep source positions -------------------------

def _build_anchored(ctx):
    """Start from scaled source positions, preserving edge anchors where possible.

    Clamp blocks and relax small overlaps; the caller still validates the result.
    """
    k = min(ctx.cw / ctx.sw, ctx.ch / ctx.sh)
    out = {}
    for u in ctx.an.units:
        b = _scaled_block(ctx, u, k)
        if b is None:
            return None
        # Anchor the *scaled source box*; the block keeps its top-left (text
        # is drawn top-aligned, so a trimmed text box must not re-centre).
        x, y, w, h = u.box
        sw_, sh_ = w * k, h * k
        cx = (x + w / 2) / ctx.sw * ctx.cw - sw_ / 2
        cy = (y + h / 2) / ctx.sh * ctx.ch - sh_ / 2
        if x <= 0.12 * ctx.sw:
            cx = x * k
        elif x + w >= 0.88 * ctx.sw:
            cx = ctx.cw - (ctx.sw - x - w) * k - sw_
        if y <= 0.12 * ctx.sh:
            cy = y * k
        elif y + h >= 0.88 * ctx.sh:
            cy = ctx.ch - (ctx.sh - y - h) * k - sh_
        cx = _clamp(cx, 0.0, max(0.0, ctx.cw - b.w))
        cy = _clamp(cy, 0.0, max(0.0, ctx.ch - b.h))
        _shift(b, cx, cy, out)
    return _relax(ctx, out)


def _relax(ctx, out):
    """Try up to four passes of vertical nudges for small unit-box overlaps.

    Move the lower unit down or the upper one up when it fits. Leave remaining
    overlaps, including large ones, for candidate validation to reject.
    """
    units = ctx.an.units
    gap = 2.0
    for _ in range(4):
        boxes = {u.uid: _union(_el_box(out[i]) for i in u.ids) for u in units}
        order = sorted(units, key=lambda u: boxes[u.uid][1])
        moved = False
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                ua, ub = order[a], order[b]
                A, B = boxes[ua.uid], boxes[ub.uid]
                if _inter(A, B) <= 0.25:
                    continue
                dy = A[1] + A[3] - B[1] + gap
                if dy > max(0.35 * min(A[3], B[3]), 0.03 * ctx.ch):
                    continue
                if B[1] + B[3] + dy <= ctx.ch:
                    target, delta = ub, dy
                elif A[1] - dy >= 0:
                    target, delta = ua, -dy
                else:
                    continue
                for i in target.ids:
                    e = dict(out[i])
                    e['y'] += delta
                    out[i] = e
                boxes[target.uid] = _union(_el_box(out[i]) for i in target.ids)
                moved = True
        if not moved:
            break
    return out


# ---------------- vertical: stacks of rows --------------------------------

def _stack_rows_source(an):
    rows = []
    for row in an.rows:
        rows.append([u for u in row])
    return rows


def _stack_rows_lanes(an):
    lanes = _lanes(an)
    rows = []
    top_f = [u for u in lanes['X'] if (u.corner or 'b')[0] == 't']
    bot_f = [u for u in lanes['X'] if u not in top_f]
    first = lanes['B'][:1]
    rows.append(top_f[:1] + first + top_f[1:2])
    for u in lanes['B'][1:]:
        rows.append([u])
    for u in lanes['V']:
        rows.append([u])
    for u in lanes['C']:
        rows.append([u])
    if lanes['D']:
        rows.append(list(lanes['D']))
    for u in lanes['A']:
        rows.append([u])
    for u in top_f[2:]:
        rows.append([u])
    if bot_f:
        rows.append(bot_f)
    for u in lanes['F']:
        rows.append([u])
    return [r for r in rows if r]


def _row_block(ctx, row, p, opts, W, H):
    """One stack row: centred content, corner units pinned to their side."""
    pinned_l = [u for u in row if u.corner and u.side == 'left' and len(row) > 1]
    pinned_r = [u for u in row if u.corner and u.side == 'right' and len(row) > 1]
    center = [u for u in row if u not in pinned_l and u not in pinned_r]
    if not center:
        center, pinned_l, pinned_r = row, [], []

    def natural(u, width):
        if u.kind == 'text':
            return _make_piece(ctx, ('text', u), p, opts, width, H)
        if u.kind == 'pill':
            return _make_piece(ctx, ('pill', u), p, opts, width, H)
        return _make_piece(ctx, ('vis', u), p, opts, width, H)

    side_blocks = {}
    for u in pinned_l + pinned_r:
        b = natural(u, W * 0.3)
        if b is None:
            return None
        side_blocks[u.uid] = b
    side = 0.0
    if side_blocks:
        side = max(b.w for b in side_blocks.values()) + ctx.gap
    wc = W - 2 * side
    if wc < 0.4 * W:
        return None
    if len(center) == 1:
        u = center[0]
        b = natural(u, wc)
        if b is None:
            return None
        cblock = b
    else:
        bs = []
        for u in center:
            limit = wc if u.kind != 'text' else wc / len(center)
            b = natural(u, limit if u.kind in ('text', 'pill') else wc / len(center))
            if b is None:
                return None
            bs.append(b)
        g = max(ctx.gmin, 0.6 * max((b.font for b in bs), default=0), 0.5 * ctx.gap)
        cblock = _flow(bs, wc, g)
    rh = max([cblock.h] + [b.h for b in side_blocks.values()])
    items = []
    cx = (W - cblock.w) / 2
    for i, e in cblock.els:
        e2 = dict(e)
        e2['x'] += cx
        e2['y'] += (rh - cblock.h) / 2
        items.append((i, e2))
    for u in pinned_l + pinned_r:
        b = side_blocks[u.uid]
        bx = 0.0 if u in pinned_l else W - b.w
        by = 0.0 if (u.corner or 't')[0] == 't' else rh - b.h
        for i, e in b.els:
            e2 = dict(e)
            e2['x'] += bx
            e2['y'] += by
            items.append((i, e2))
    fonts = [cblock.font] + [b.font for b in side_blocks.values()]
    return Block(W, rh, items, None, max(fonts), 'row')


def _build_stack(ctx, rows, p, opts, rhythm):
    W = ctx.cw - 2 * ctx.mx
    H = ctx.ch - 2 * ctx.my
    blocks = []
    final_rows = []
    for row in rows:
        b = _row_block(ctx, row, p, opts, W, H)
        if b is None and len(row) > 1:
            # Corner items that cannot share the width get a row of their own,
            # above the content for top corners and below it otherwise.
            pinned = [u for u in row if u.corner]
            rest = [u for u in row if not u.corner]
            if not pinned or not rest:
                return None
            top = [u for u in pinned if u.corner[0] == 't']
            bottom = [u for u in pinned if u.corner[0] != 't']
            parts = ([top] if top else []) + [rest] + ([bottom] if bottom else [])
            for part in parts:
                pb = _row_block(ctx, part, p, opts, W, H)
                if pb is None:
                    return None
                blocks.append(pb)
                final_rows.append(part)
            continue
        if b is None:
            return None
        blocks.append(b)
        final_rows.append(row)
    rows = final_rows
    gaps = []
    for k in range(len(blocks) - 1):
        a, b = blocks[k], blocks[k + 1]
        fa, fb = a.font, b.font
        g = 0.3 * (fa + fb) if fa and fb else 0.5 * (fa or fb) if (fa or fb) else 0.04 * H
        gaps.append(max(ctx.gmin, g, 0.012 * ctx.ch))
    content = sum(b.h for b in blocks) + sum(gaps)
    if content > H + 1e-6:
        return None
    free = H - content
    # Distribute spare height following the source's vertical rhythm.
    if rhythm:
        spans = [(min(u.box[1] for u in row), max(u.box[1] + u.box[3] for u in row)) for row in rows]
        weights = [max(spans[0][0], 0.0)]
        weights += [max(spans[k + 1][0] - spans[k][1], 0.0) for k in range(len(spans) - 1)]
        weights.append(max(ctx.sh - spans[-1][1], 0.0))
        # Damped: keeps the source's rhythm without exaggerating empty bands.
        weights = [(w + 0.02 * ctx.sh) ** 0.75 for w in weights]
    else:
        weights = [1.4] + [1.0] * (len(blocks) - 1) + [1.4]
    tw = sum(weights)
    extra = [free * w / tw for w in weights]
    if not rhythm:
        # Lane order has no source rhythm: let gaps grow to at most 2.5x their
        # base size, then centre the block instead of letting it drift apart.
        inner_cap = [min(extra[k + 1], gaps[k] * 1.5) for k in range(len(gaps))]
        spare = free - sum(inner_cap)
        extra = [spare / 2] + inner_cap + [spare / 2]
    out = {}
    y = ctx.my + extra[0]
    for k, b in enumerate(blocks):
        _shift(b, ctx.mx, y, out)
        y += b.h
        if k < len(gaps):
            y += gaps[k] + extra[k + 1]
    return out


# ==========================================================================
# Floaters, decorations, panels, backgrounds
# ==========================================================================

def _grow_hero_in_free_space(ctx, out):
    """Grow resolved hero images locally, holding every other element fixed.

    Search obstacle edges as well as the original centre, allowing upward or
    sideways movement into free space. Use rotated bounds and preserve the
    extra clearance required by missing-image diagnostic boxes.
    """
    out = dict(out)
    for u in ctx.an.units:
        if u.role != 'hero' or u.kind != 'image' or u.missing or len(u.ids) != 1:
            continue
        index = u.ids[0]
        if index not in out:
            continue
        original = out[index]
        box = _el_box(original)
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        obstacles = []
        for other in ctx.an.units:
            if other is u:
                continue
            gap = max(ctx.gmin, ctx.m)
            if other.missing:
                gap = max(gap, min(12.0, MISSING_ASSET_GAP_FRACTION * ctx.short))
            obstacles.extend((_el_box(out[i]), gap) for i in other.ids if i in out)
        edge = max(2.0, ctx.my)
        reach = 0.5 * ctx.short
        scale = original['width'] / ctx.els[index]['width']

        def fit(factor):
            block = _scaled_block(ctx, u, scale * factor)
            w, h = block.w, block.h
            xmin = max(edge, cx - reach - w / 2)
            xmax = min(ctx.cw - edge - w, cx + reach - w / 2)
            ymin = max(edge, cy - reach - h / 2)
            ymax = min(ctx.ch - edge - h, cy + reach - h / 2)
            if xmin > xmax or ymin > ymax:
                return None
            xs = {xmin, xmax, _clamp(cx - w / 2, xmin, xmax)}
            ys = {ymin, ymax, _clamp(cy - h / 2, ymin, ymax)}
            for ob, gap in obstacles:
                xs.update((ob[0] - gap - w, ob[0] + ob[2] + gap))
                ys.update((ob[1] - gap - h, ob[1] + ob[3] + gap))
            positions = sorted(((x, y) for x in xs if xmin <= x <= xmax
                                for y in ys if ymin <= y <= ymax),
                               key=lambda pos: ((pos[0] + w / 2 - cx) ** 2 +
                                                (pos[1] + h / 2 - cy) ** 2, pos[1], pos[0]))
            for x, y in positions:
                if not any(_hits((x, y, w, h), ob, gap - 1e-6) for ob, gap in obstacles):
                    return block, x, y
            return None

        lo, hi = 1.0, min(ctx.cw / box[2], ctx.ch / box[3])
        best = None
        for _ in range(12):
            mid = (lo + hi) / 2
            found = fit(mid)
            if found is None:
                hi = mid
            else:
                best, lo = found, mid
        if best is not None and min(best[0].w - box[2], best[0].h - box[3]) >= 0.5:
            block, x, y = best
            _shift(block, x, y, out)
    return out


def _unit_boxes(ctx, out, units):
    res = []
    for u in units:
        if all(i in out for i in u.ids):
            res.append(_union(_el_box(out[i]) for i in u.ids))
    return res


def _place_floaters(ctx, out, p, opts, clearance=None, region=None):
    floaters = sorted(ctx.lanes['X'], key=lambda u: -IMPORTANCE.get(u.role, 0.5))
    if not floaters:
        return out
    placed_units = [u for u in ctx.an.units if all(i in out for i in u.ids)]
    obstacles = _unit_boxes(ctx, out, placed_units)
    g = clearance if clearance is not None else max(ctx.gmin, 0.5 * ctx.gap)
    cw, ch = ctx.cw, ctx.ch
    base_gap = g
    for u in floaters:
        g = max(base_gap, min(12.0, MISSING_ASSET_GAP_FRACTION * ctx.short)) if u.missing else base_gap
        ax = (u.box[0] + u.box[2] / 2) / ctx.sw * cw
        ay = (u.box[1] + u.box[3] / 2) / ctx.sh * ch
        done = False
        for mult in (1.0, 0.85, 0.7, 0.58, 0.48, 0.4):
            if u.kind == 'pill':
                size = max(ctx.floor(u.role, u.font), int(ctx.tsize(u.role, u.font, _lane_p(u, p, opts)) * mult))
                b = _pill_block(ctx, u, size, True)
            else:
                scale = ctx.vscale(u, p, min(opts.r, 1.0)) * mult
                scale = min(scale, 0.45 * ch / max(u.box[3], 1e-6), 0.3 * cw / max(u.box[2], 1e-6))
                b = _scaled_block(ctx, u, scale)
            if b is None or b.w > cw - 2 or b.h > ch - 2:
                continue
            xs = {ctx.mx, cw - ctx.mx - b.w, (cw - b.w) / 2, ax - b.w / 2}
            ys = {ctx.my, ch - ctx.my - b.h, (ch - b.h) / 2, ay - b.h / 2}
            for ob in obstacles:
                xs.update((ob[0] - g - b.w, ob[0] + ob[2] + g, ob[0], ob[0] + ob[2] - b.w))
                ys.update((ob[1] - g - b.h, ob[1] + ob[3] + g, ob[1], ob[1] + ob[3] - b.h))
            best = None
            edge = min(ctx.mx, ctx.my) * 0.5
            rx0, ry0, rx1, ry1 = region or (0.0, 0.0, cw, ch)
            xs = [x for x in xs if max(edge, rx0) - 1e-6 <= x and x + b.w <= min(cw - edge, rx1) + 1e-6]
            ys = [y for y in ys if max(edge, ry0) - 1e-6 <= y and y + b.h <= min(ch - edge, ry1) + 1e-6]
            # Nearest-to-anchor first (lazy best-first over the x/y grid): the
            # first collision-free spot wins.
            if xs and ys:
                xs.sort(key=lambda x: abs(x + b.w / 2 - ax))
                ys.sort(key=lambda y: abs(y + b.h / 2 - ay))
                dx = [((x + b.w / 2 - ax) / cw) ** 2 for x in xs]
                dy = [((y + b.h / 2 - ay) / ch) ** 2 for y in ys]
                heap = [(dx[0] + dy[j], 0, j) for j in range(len(ys))]
                heapq.heapify(heap)
                while heap:
                    d, a, j = heapq.heappop(heap)
                    box = (xs[a], ys[j], b.w, b.h)
                    if not any(_hits(box, ob, g * 0.999) for ob in obstacles):
                        best = (d, xs[a], ys[j])
                        break
                    if a + 1 < len(xs):
                        heapq.heappush(heap, (dx[a + 1] + dy[j], a + 1, j))
            if best is not None:
                _, x, y = best
                _shift(b, x, y, out)
                obstacles.append((x, y, b.w, b.h))
                done = True
                break
        if not done:
            return None
    return out


def _place_decorations(ctx, out, keep_layering=False):
    an = ctx.an
    els = ctx.els
    cw, ch, sw, sh = ctx.cw, ctx.ch, ctx.sw, ctx.sh
    # Text coverage is weighted by role importance: hiding a headline behind a
    # decoration is never "the same" as the badge label it covered in the source.
    weight = {i: IMPORTANCE.get(an.unit_of[i].role, 1.0) if i in an.unit_of else 0.5
              for i in range(len(els)) if els[i]['type'] == 'text'}
    texts = [(out[i], weight[i]) for i in out if els[i]['type'] == 'text']
    for i in an.decorations:
        others = []
        src = els[i]
        sb = an.boxes[i]
        area = max(sb[2] * sb[3], 1e-6)
        anchor = None
        best = 0.0
        for u in an.units:
            if u.kind != 'image' or not all(j in out for j in u.ids):
                continue
            j = u.ids[0]
            if els[j].get('z_index', 0) <= src.get('z_index', 0):
                continue
            ratio = _inter(sb, an.boxes[j]) / min(area, max(an.boxes[j][2] * an.boxes[j][3], 1e-6))
            if ratio > best:
                best, anchor = ratio, j
        e = dict(src)
        for u in an.units:
            if anchor is not None and best > 0.2 and anchor in u.ids:
                continue
            if u.kind != 'text' and all(j in out for j in u.ids):
                others.append(_union(_el_box(out[j]) for j in u.ids))
        if anchor is not None and best > 0.2:
            a, b = els[anchor], out[anchor]
            k = b['width'] / max(a['width'], 1e-6)
            e['width'], e['height'] = src['width'] * k, src['height'] * k
            e['x'] = b['x'] + (src['x'] - a['x']) * k
            e['y'] = b['y'] + (src['y'] - a['y']) * k
        else:
            k = ctx.s_img
            e['width'], e['height'] = src['width'] * k, src['height'] * k
            cx = (sb[0] + sb[2] / 2) / sw * cw
            cy = (sb[1] + sb[3] / 2) / sh * ch
            e['x'], e['y'] = cx - e['width'] / 2, cy - e['height'] / 2
        if src.get('stroke_width'):
            e['stroke_width'] = float(src['stroke_width']) * k
        if src.get('corner_radius'):
            e['corner_radius'] = float(src['corner_radius']) * k
        # Stay on the canvas; keep proportions.
        bx, by, bw, bh = _el_box(e)
        fit = min(1.0, (cw - 1) / max(bw, 1e-6), (ch - 1) / max(bh, 1e-6))
        if fit < 1.0:
            cx, cy = e['x'] + e['width'] / 2, e['y'] + e['height'] / 2
            e['width'] *= fit
            e['height'] *= fit
            e['x'], e['y'] = cx - e['width'] / 2, cy - e['height'] / 2
        bx, by, bw, bh = _el_box(e)
        ox = e['x'] - bx
        oy = e['y'] - by
        nx = _clamp(bx, 0.0, cw - bw)
        ny = _clamp(by, 0.0, ch - bh)
        candidates = [(nx, ny)] + [(xx, yy) for xx in (0.0, cw - bw) for yy in (0.0, ch - bh)]
        candidates += [((cw - bw) / 2, 0.0), ((cw - bw) / 2, ch - bh), (0.0, (ch - bh) / 2), (cw - bw, (ch - bh) / 2)]

        # Covering what the source decoration already covered is intended
        # layering; only additional coverage of copy or imagery costs.
        src_texts = [(an.boxes[j], weight[j]) for j in range(len(els)) if els[j]['type'] == 'text']
        src_others = [u.box for u in an.units if u.kind != 'text']
        base_t = sum(w * _inter(sb, t) / max(1.0, t[2] * t[3]) for t, w in src_texts)
        base_o = sum(_inter(sb, o) / max(1.0, o[2] * o[3]) for o in src_others)

        def cost(pos):
            box = (pos[0], pos[1], bw, bh)
            occ = sum(w * _inter(box, _el_box(t)) / max(1.0, t['width'] * t['height']) for t, w in texts)
            occ2 = sum(_inter(box, o) / max(1.0, o[2] * o[3]) for o in others)
            d = ((pos[0] - nx) / cw) ** 2 + ((pos[1] - ny) / ch) ** 2
            return 20 * max(0.0, occ - base_t) + 4 * max(0.0, occ2 - base_o) + d
        px, py = (nx, ny) if keep_layering else min(candidates, key=cost)
        e['x'], e['y'] = px + ox, py + oy
        out[i] = e
    return out


def _place_panels(ctx, out):
    an = ctx.an
    for i, kids in an.panels.items():
        src = ctx.els[i]
        kid_boxes = [_el_box(out[j]) for j in kids if j in out]
        e = dict(src)
        if kid_boxes:
            pad = ctx.m * 0.75
            x, y, w, h = _union(kid_boxes)
            x0, y0 = max(0.0, x - pad), max(0.0, y - pad)
            x1, y1 = min(ctx.cw, x + w + pad), min(ctx.ch, y + h + pad)
            sb = an.boxes[i]
            if sb[0] <= 1:
                x0 = 0.0
            if sb[1] <= 1:
                y0 = 0.0
            if sb[0] + sb[2] >= ctx.sw - 1:
                x1 = ctx.cw
            if sb[1] + sb[3] >= ctx.sh - 1:
                y1 = ctx.ch
            e.update(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        else:
            k = ctx.s_img
            e.update(x=src['x'] * ctx.cw / ctx.sw, y=src['y'] * ctx.ch / ctx.sh,
                     width=src['width'] * k, height=src['height'] * k)
        if src.get('corner_radius'):
            e['corner_radius'] = float(src['corner_radius']) * ctx.s_img
        out[i] = e
    return out


def _place_inert(ctx, out):
    """Retain empty-text and zero-size source elements with clamped geometry."""
    for i in ctx.an.inert:
        src = ctx.els[i]
        e = dict(src)
        k = ctx.s_img
        e['width'] = max(0.0, src['width'] * k)
        e['height'] = max(0.0, src['height'] * k)
        if e['type'] == 'text':
            e['font_size'] = max(1, int(float(src.get('font_size', 1)) * k))
            e['width'] = max(e['width'], 1.0)
            e['height'] = max(e['height'], _ceil2(measure(e, e['width'], e['font_size']).height))
        e['width'] = min(e['width'], ctx.cw)
        e['height'] = min(e['height'], ctx.ch)
        e['x'] = _clamp(src['x'] / ctx.sw * ctx.cw, 0.0, ctx.cw - e['width'])
        e['y'] = _clamp(src['y'] / ctx.sh * ctx.ch, 0.0, ctx.ch - e['height'])
        out[i] = e
    return out


def _place_backgrounds(ctx, out):
    for i in ctx.an.backgrounds:
        e = dict(ctx.els[i])
        e.update(x=0.0, y=0.0, width=ctx.cw, height=ctx.ch)
        if e.get('rotation'):
            e['rotation'] = 0  # a full-bleed backdrop must cover the canvas
        if e['type'] == 'image':
            e['fit'] = 'cover'  # crop to the new canvas; never stretch pixels
        out[i] = e
    return out


# ==========================================================================
# Validation and scoring
# ==========================================================================

def _problems(ctx, out, strict=True):
    """Check fit using 0.5px bounds/text and 0.25px-squared overlap tolerances.

    Independent-unit collisions use rotated enclosing boxes; intentional
    layering inside a unit is excluded. Background image ratios are exempt.
    """
    els = ctx.els
    cw, ch = ctx.cw, ctx.ch
    probs = []
    tol = 0.5
    for i, src in enumerate(els):
        e = out.get(i)
        if e is None:
            probs.append(('missing', i))
            continue
        x, y, w, h = _el_box(e)
        if x < -tol or y < -tol or x + w > cw + tol or y + h > ch + tol:
            probs.append(('bounds', i))
        if e['type'] == 'text':
            fit = measure(e, e['width'], e['font_size'])
            if fit.max_w > e['width'] + 0.5 or fit.height > e['height'] + 0.5:
                probs.append(('text', i))
        if e['type'] == 'image' and i not in ctx.an.backgrounds and src['height'] > 0 and e['height'] > 0:
            if abs(e['width'] / e['height'] - src['width'] / src['height']) > 0.01 * src['width'] / src['height']:
                probs.append(('aspect', i))
    if strict:
        units = ctx.an.units
        boxes = []
        for u in units:
            boxes.append([_el_box(out[i]) for i in u.ids if i in out])
        for a in range(len(units)):
            for b in range(a + 1, len(units)):
                for ba in boxes[a]:
                    if any(_inter(ba, bb) > 0.25 for bb in boxes[b]):
                        probs.append(('overlap', units[a].uid, units[b].uid))
                        break
    return probs


def _collision_area(ctx, out):
    units = ctx.an.units
    boxes = [[_el_box(out[i]) for i in u.ids if i in out] for u in units]
    total = 0.0
    for a in range(len(units)):
        for b in range(a + 1, len(units)):
            total += sum(_inter(x, y) for x in boxes[a] for y in boxes[b])
    missing = sum(1 for i in range(len(ctx.els)) if i not in out)
    return total + 1e9 * missing


@lru_cache(maxsize=8192)
def _arrangement_cost(boxes, spacing=True):
    """Disorder of a small related set, allowing either rows or columns.

    Measures alignment and uneven gaps, not absolute location, reading order,
    or a preferred number of rows. Variable text widths are allowed.
    """
    if len(boxes) < 3:
        return 0.0

    def oriented(transpose):
        bb = [(y, x, h, w) if transpose else (x, y, w, h) for x, y, w, h in boxes]
        bands = []
        for b in sorted(bb, key=lambda b: (b[1] + b[3] / 2, b[0])):
            for band in bands:
                if all(abs(b[1] + b[3] / 2 - a[1] - a[3] / 2) <= 0.3 * min(b[3], a[3])
                       for a in band):
                    band.append(b)
                    break
            else:
                bands.append([b])

        def alignment(items, axis, size, scale):
            spread = min(max(b[axis] + anchor * b[size] for b in items) -
                         min(b[axis] + anchor * b[size] for b in items)
                         for anchor in (0, 0.5, 1))
            return min(1.0, max(0.0, spread / max(scale, 1e-6) - 0.15))

        costs = []
        for band in bands:
            scale = min(min(b[2], b[3]) for b in band)
            costs.append(alignment(band, 1, 3, scale))
            ordered = sorted(band)
            gaps = [b[0] - a[0] - a[2] for a, b in zip(ordered, ordered[1:])]
            if spacing and len(gaps) >= 2:
                costs.append(min(1.0, (max(gaps) - min(gaps)) / max(scale, sum(abs(g) for g in gaps) / len(gaps), 1e-6)))
        if len(bands) > 1:
            bounds = [tuple(_union(band)) for band in bands]
            scale = min(min(b[2], b[3]) for b in bounds)
            costs.append(alignment(bounds, 0, 2, scale))
            gaps = [b[1] - a[1] - a[3] for a, b in zip(bounds, bounds[1:])]
            if spacing and len(gaps) >= 2:
                costs.append(min(1.0, (max(gaps) - min(gaps)) / max(scale, sum(abs(g) for g in gaps) / len(gaps), 1e-6)))
        return max(costs, default=0.0)

    return min(oriented(False), oriented(True))


def _organization_penalty(ctx, out):
    """Penalize lost organization only for source-supported related sets."""
    costs = []
    for group, source_cost, spacing in ctx.organization_groups:
        if not all(i in out for ids in group for i in ids):
            continue
        boxes = [tuple(_union(_el_box(out[i]) for i in ids)) for ids in group]
        # Translation-normalize cache keys; relative positions are what matter.
        ox, oy = min(b[0] for b in boxes), min(b[1] for b in boxes)
        normalized = tuple((round(x - ox, 2), round(y - oy, 2), round(w, 2), round(h, 2))
                           for x, y, w, h in boxes)
        costs.append(max(0.0, _arrangement_cost(normalized, spacing) - source_cost))
    return ORGANIZATION_WEIGHT * max(costs, default=0.0)


def _score(ctx, out, layout_kind, organization=True):
    an = ctx.an
    els = ctx.els
    cw, ch = ctx.cw, ctx.ch
    s = 0.0
    hier = []
    tight_banner = ctx.aspect >= 4.0 and ch <= 120
    for u in an.units:
        if not all(i in out for i in u.ids):
            continue  # a floater not placed yet (zone pre-scoring)
        imp = IMPORTANCE.get(u.role, 1.0)
        if u.kind in ('text', 'pill'):
            e = out[u.text_id]
            f = float(e['font_size'])
            src_f = float(els[u.text_id]['font_size'])
            cap = max(1.0, ctx.tcap(u.role, src_f))
            s += imp * math.log(max(f, 1.0) / cap)
            soft = ctx.soft.get(u.role, 8.0)
            if f < soft:
                # On short strips, readable copy takes priority over visual
                # size and keeping the original number of text lines.
                readability_weight = 16.0 if tight_banner else 8.0
                s -= imp * readability_weight * ((soft - f) / soft) ** 2
            if u.kind == 'text':
                lines = measure(e, e['width'], e['font_size']).lines
                limit = {'headline': 3, 'body': 4, 'fine': 3}.get(u.role, 4)
                if lines > limit:
                    s -= (0.15 if tight_banner else 0.35) * imp * (lines - limit)
            hier.append((src_f, f, imp))
        else:
            boxes = [_el_box(out[i]) for i in u.ids]
            bx = _union(boxes)
            vis = u.visible
            vh = bx[3] * (vis[3] - vis[1])
            ref = math.sqrt(max(u.box[2] * u.box[3], 1.0)) * ctx.s_img
            size = math.sqrt(max(bx[2] * bx[3], 1.0))
            s += imp * 0.8 * min(0.25, math.log(size / ref))
            if u.role == 'logo':
                want = 0.28 * ctx.short if ctx.short < 300 else 0.07 * ctx.short
                if vh < want:
                    s -= imp * 2.0 * ((want - vh) / want) ** 2
            if u.role == 'hero':
                frac = (bx[2] * bx[3]) / (cw * ch)
                s -= imp * 1.5 * max(0.0, 0.08 - frac) / 0.08
    # Penalize hierarchy inversions; this is a preference, not a hard constraint.
    for a in hier:
        for b in hier:
            if a[0] >= 1.3 * b[0] and a[1] < 1.1 * b[1]:
                s -= 0.8 * (1.1 * b[1] - a[1]) / max(b[1], 1.0)
    if ctx.aspect >= 4.0:
        s += _banner_terms(ctx, out)
    # Coverage and balance of the foreground content.
    boxes = [_el_box(out[i]) for u in an.units for i in u.ids if i in out]
    if boxes:
        x, y, w, h = _union(boxes)
        cov_w = w / (cw - 2 * ctx.mx)
        cov_h = h / (ch - 2 * ctx.my)
        s -= 1.2 * max(0.0, 0.85 - cov_w) + 1.2 * max(0.0, 0.7 - cov_h)
        area = sum(b[2] * b[3] for b in boxes) or 1.0
        mx = sum((b[0] + b[2] / 2) * b[2] * b[3] for b in boxes) / area
        my = sum((b[1] + b[3] / 2) * b[2] * b[3] for b in boxes) / area
        s -= 1.0 * abs(mx - cw / 2) / cw + 0.6 * abs(my - ch / 2) / ch
    s += {'stack_source': 0.25, 'stack_lanes': 0.0, 'columns': 0.0, 'fallback': -5.0}.get(layout_kind, 0.0)
    if layout_kind == 'anchored':
        # Favor source-based placement when the aspect ratio changes little.
        drift = abs(math.log(ctx.aspect / ctx.src_aspect))
        s += 1.5 * max(0.0, 1.0 - drift / 0.3)
    if 'inline' in layout_kind:
        s -= 0.15   # floaters folded into a column drift from the source composition
    if 'logo_late' in layout_kind:
        s -= 0.35   # brand reads first: logo left of / above the copy
    if layout_kind.startswith('zones'):
        s += 2.0    # on strips, the designed three-zone composition is preferred
        # Fine print belongs under the subhead; elsewhere it is a compromise.
        s -= {'zones:fine_strip': 0.3, 'zones:fine_right': 0.8}.get(layout_kind, 0.0)
    return s - (_organization_penalty(ctx, out) if organization else 0.0)


def _banner_terms(ctx, out):
    """Strip priorities: readable copy, a prominent CTA and visible imagery.

    Short banners tolerate extra wrapped lines and a smaller hero when that
    permits larger text; fit and collision checks remain mandatory.
    """
    s = 0.0
    tight = ctx.ch <= 120
    if tight:
        # In near-equal short-banner layouts, prefer copy followed by the
        # product and brand/action group. Infer order from geometry and roles,
        # never sample ids. This remains a soft preference: fit is mandatory
        # and readability penalties can outweigh it on other designs.
        boxes = [(u, _union(_el_box(out[i]) for i in u.ids))
                 for u in ctx.an.units if all(i in out for i in u.ids)]
        copy = [box for u, box in boxes if u.role in ('headline', 'body')]
        actions = [box for u, box in boxes if u.role in ('logo', 'cta')]
        heroes = [box for u, box in boxes if u.role == 'hero']
        if copy and actions and any(
                all(b[0] + b[2] <= hero[0] + 0.5 for b in copy) and
                all(b[0] >= hero[0] + hero[2] - 0.5 for b in actions)
                for hero in heroes):
            s += BANNER_COPY_FIRST_BONUS
    for u in ctx.an.units:
        if not all(i in out for i in u.ids):
            continue
        if u.role == 'headline' and u.kind == 'text':
            e = out[u.text_id]
            f = float(e['font_size'])
            target = min(BANNER_HEADLINE_TARGET, ctx.tcap('headline', u.font))
            if f < target:
                s -= 0.9 * (target - f)
            lines = measure(e, e['width'], e['font_size']).lines
            if lines > 3:
                s -= (0.25 if tight else 1.0) * (lines - 3)
        elif u.role == 'body' and u.kind == 'text' and tight:
            # Below its readable target, prioritize supporting copy enough
            # that a larger hero cannot routinely keep it at the size floor.
            f = float(out[u.text_id]['font_size'])
            target = min(ctx.soft['body'], ctx.tcap('body', u.font))
            s -= 1.8 * max(0.0, target - f)
        elif u.role == 'cta' and u.kind == 'pill':
            f = float(out[u.text_id]['font_size'])
            target = min(BANNER_CTA_TARGET, ctx.tcap('cta', u.font))
            if f < target:
                s -= 0.4 * (target - f)
        elif u.role == 'hero' and tight:
            frac = _union(_el_box(out[i]) for i in u.ids)[3] / ctx.ch
            if frac < 0.8:
                s -= 2.0 * (0.8 - frac) / 0.8
        elif u.role == 'logo' and tight:
            # Prefer visible logo height near 30% of the strip; smaller is allowed.
            box = _union(_el_box(out[i]) for i in u.ids)
            visible = box[3] * (u.visible[3] - u.visible[1])
            s -= 0.15 * max(0.0, 0.3 * ctx.ch - visible)
    # Penalize floaters close beside the CTA; this does not forbid placement.
    ctas = [_union(_el_box(out[i]) for i in u.ids) for u in ctx.an.units
            if u.role == 'cta' and all(i in out for i in u.ids)]
    for u in ctx.lanes['X']:
        if not all(i in out for i in u.ids):
            continue
        b = _union(_el_box(out[i]) for i in u.ids)
        for c in ctas:
            side_by_side = min(b[1] + b[3], c[1] + c[3]) - max(b[1], c[1]) > 0.3 * min(b[3], c[3])
            gap = max(b[0] - (c[0] + c[2]), c[0] - (b[0] + b[2]))
            if side_by_side and gap < 1.2 * c[3]:
                s -= 3.0 if u.missing else 0.6
    # Missing images show an opaque, high-contrast placeholder. Prefer a
    # clear gap from ALL content (including brand marks and fine print), not
    # just a collision-free slot next to the CTA. Keep this a soft preference
    # so dense formats do not sacrifice readable copy to a diagnostic box.
    for u in ctx.an.units:
        if not u.missing or not all(i in out for i in u.ids):
            continue
        box = _union(_el_box(out[i]) for i in u.ids)
        distances = []
        for other in ctx.an.units:
            if other is u or not all(i in out for i in other.ids):
                continue
            ob = _union(_el_box(out[i]) for i in other.ids)
            dx = max(0, box[0] - ob[0] - ob[2], ob[0] - box[0] - box[2])
            dy = max(0, box[1] - ob[1] - ob[3], ob[1] - box[1] - box[3])
            distances.append(math.hypot(dx, dy))
        if distances:
            s -= 0.5 * max(0, min(12.0, MISSING_ASSET_GAP_FRACTION * ctx.short) - min(distances))
    return s


# ==========================================================================
# Fallback and final repair
# ==========================================================================

def _fallback(ctx):
    """Propose scaled source units in a centred frame; repair/validation follow."""
    k = min(ctx.cw / ctx.sw, ctx.ch / ctx.sh)
    ox = (ctx.cw - ctx.sw * k) / 2
    oy = (ctx.ch - ctx.sh * k) / 2
    out = {}
    for u in ctx.an.units:
        b = _scaled_block(ctx, u, k)
        if b is None:
            continue
        _shift(b, ox + u.box[0] * k, oy + u.box[1] * k, out)
    return out


def _grid_fallback(ctx):
    """Propose a reading-order grid with one scaled unit per cell.

    Text refitting can change block extents, so repair/validation are required.
    """
    units = ctx.an.units
    n = len(units)
    if not n:
        return {}
    W, H = ctx.cw - 2 * ctx.mx, ctx.ch - 2 * ctx.my
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cw_, ch_ = W / cols, H / rows
        fill = sum(min(cw_ / max(u.box[2], 1e-6), ch_ / max(u.box[3], 1e-6)) ** 2 * u.box[2] * u.box[3]
                   for u in units)
        if best is None or fill > best[0]:
            best = (fill, cols, rows)
    _, cols, rows = best
    cw_, ch_ = W / cols, H / rows
    pad = min(2.0, 0.05 * min(cw_, ch_))
    out = {}
    for k, u in enumerate(units):
        r, c = divmod(k, cols)
        scale = min((cw_ - 2 * pad) / max(u.box[2], 1e-6), (ch_ - 2 * pad) / max(u.box[3], 1e-6))
        b = _scaled_block(ctx, u, max(scale, 1e-3))
        if b is None:
            continue
        x = ctx.mx + c * cw_ + (cw_ - b.w) / 2
        y = ctx.my + r * ch_ + (ch_ - b.h) / 2
        _shift(b, x, y, out)
    return out


def _repair(ctx, out):
    """Adjust text fit, size, bounds and rounding; callers must validate afterward.

    This pass can introduce unit collisions and cannot guarantee a valid layout.
    """
    cw, ch = ctx.cw, ctx.ch
    for i, e in list(out.items()):
        e = dict(e)
        for key in ('x', 'y', 'width', 'height'):
            e[key] = float(e.get(key, 0) or 0)
        e['width'] = max(e['width'], 0.0)
        e['height'] = max(e['height'], 0.0)
        if e['type'] == 'text':
            e['font_size'] = max(1, int(e.get('font_size', 1)))
            e['width'] = _floor2(min(e['width'], cw))
            fit = measure(e, e['width'], e['font_size'])
            while (fit.max_w > e['width'] + 1e-6 or fit.height > ch) and e['font_size'] > 1:
                e['font_size'] -= 1
                fit = measure(e, e['width'], e['font_size'])
            if fit.max_w > e['width'] + 1e-6:  # even 1px overflows: widen, keep centre
                grow = min(cw, math.ceil(fit.max_w + 1)) - e['width']
                e['x'] -= grow / 2
                e['width'] = _floor2(e['width'] + grow)
                fit = measure(e, e['width'], e['font_size'])
            e['height'] = max(e['height'], _ceil2(fit.height))
        x, y, w, h = _el_box(e)
        if w > cw or h > ch:
            k = min(cw / max(w, 1e-6), ch / max(h, 1e-6)) * 0.999
            cx, cy = e['x'] + e['width'] / 2, e['y'] + e['height'] / 2
            e['width'] *= k
            e['height'] *= k
            e['x'], e['y'] = cx - e['width'] / 2, cy - e['height'] / 2
            if e['type'] == 'text':
                e['width'] = _floor2(e['width'])
                while e['font_size'] > 1:
                    fit = measure(e, e['width'], e['font_size'])
                    if fit.max_w <= e['width'] and fit.height <= e['height']:
                        break
                    e['font_size'] -= 1
            x, y, w, h = _el_box(e)
        dx = -x if x < 0 else (cw - (x + w)) if x + w > cw else 0.0
        dy = -y if y < 0 else (ch - (y + h)) if y + h > ch else 0.0
        e['x'] += dx
        e['y'] += dy
        # Round to 1/100 px without leaving the canvas.
        e['x'] = round(e['x'], 2)
        e['y'] = round(e['y'], 2)
        e['width'] = _floor2(e['width']) if e['type'] == 'text' else round(e['width'], 2)
        e['height'] = _ceil2(e['height']) if e['type'] == 'text' else round(e['height'], 2)
        x, y, w, h = _el_box(e)
        if x < 0:
            e['x'] = round(e['x'] - x + 0.005, 2)
        if y < 0:
            e['y'] = round(e['y'] - y + 0.005, 2)
        x, y, w, h = _el_box(e)
        if x + w > cw:
            e['x'] = round(e['x'] - (x + w - cw) - 0.005, 2)
        if y + h > ch:
            e['y'] = round(e['y'] - (y + h - ch) - 0.005, 2)
        for key in ('corner_radius', 'stroke_width', 'letter_spacing'):
            if isinstance(e.get(key), float):
                e[key] = round(e[key], 3)
        out[i] = e
    return out


# ==========================================================================
# Driver
# ==========================================================================

class LayoutValidationError(ValueError):
    """The bounded search could not produce a verified layout."""

    def __init__(self, problems):
        self.problems = tuple(problems)
        super().__init__(f'No valid layout found after bounded repair: {self.problems[:8]}')


def _final_problems(ctx, out):
    """Validate geometry and preserved fields on the actual returned elements.

    Check index coverage, preserved fields, finite geometry, font sizes and
    line limits as well as _problems' fit checks and tolerances. Background
    fit/rotation changes are allowed by the existing placement policy.
    """
    problems = []
    if set(out) != set(range(len(ctx.els))):
        problems.append(('element_count',))
    for i, src in enumerate(ctx.els):
        e = out.get(i)
        if e is None:
            problems.append(('missing', i))
            continue
        allowed = {'x', 'y', 'width', 'height', 'font_size',
                   'corner_radius', 'stroke_width', 'letter_spacing'}
        if i in ctx.an.backgrounds:
            allowed |= {'rotation', 'fit'}
        for key in sorted(set(src) | set(e)):
            if key not in allowed and (key not in src or key not in e or src[key] != e[key]):
                problems.append(('preservation', i, key))
        for key in ('x', 'y', 'width', 'height'):
            value = e.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(('geometry', i, key))
            elif key in ('width', 'height') and value < 0:
                problems.append(('geometry', i, key))
        if e.get('type') == 'text':
            f = e.get('font_size')
            if not isinstance(f, (int, float)) or not math.isfinite(f) or f < 1:
                problems.append(('font_size', i))
    if problems:
        return problems
    problems = _problems(ctx, out)
    for i, e in out.items():
        if e['type'] == 'text' and e.get('max_lines'):
            if measure(e, e['width'], e['font_size']).lines > int(e['max_lines']):
                problems.append(('max_lines', i))
    return problems


def _review_and_repair(ctx, out, rounds=3):
    """Repair fit, then resolve collisions by translating whole layout units.

    By default, try three greedy rounds with up to eight collisions per round
    and eight edge moves per collision. Accepted moves strictly improve the
    problem-count/collision-area pair; this does not guarantee convergence.
    Inputs are never mutated; content, sizes and group-relative positions
    remain fixed during collision repair.
    """
    best = _repair(ctx, dict(out))
    problems = _final_problems(ctx, best)
    units = {u.uid: u for u in ctx.an.units}
    for _ in range(rounds):
        if not problems:
            break
        rank = (len(problems), _collision_area(ctx, best))
        choice = None
        for _, a, b in [p for p in problems if p[0] == 'overlap'][:8]:
            for uid, other in ((a, b), (b, a)):
                unit, obstacle = units[uid], units[other]
                x, y, w, h = _union(_el_box(best[i]) for i in unit.ids)
                ox, oy, ow, oh = _union(_el_box(best[i]) for i in obstacle.ids)
                for dx, dy in ((ox - x - w - 0.5, 0), (ox + ow - x + 0.5, 0),
                               (0, oy - y - h - 0.5), (0, oy + oh - y + 0.5)):
                    if x + dx < 0 or y + dy < 0 or x + dx + w > ctx.cw or y + dy + h > ctx.ch:
                        continue
                    trial = dict(best)
                    for i in unit.ids:
                        trial[i] = dict(best[i], x=round(best[i]['x'] + dx, 2),
                                        y=round(best[i]['y'] + dy, 2))
                    issues = _final_problems(ctx, trial)
                    trial_rank = (len(issues), _collision_area(ctx, trial))
                    if trial_rank < rank:
                        choice, rank = (trial, issues), trial_rank
        if choice is None:
            break
        best, problems = choice
    return best, problems


def _select_reviewed(ctx, candidates):
    """Finalize valid candidates, then apply readability and organization policy.

    If none passes, try bounded collision recovery on six rejected candidates.
    """
    valid, rejected = [], []
    for kind, full, label in candidates:
        problems = _problems(ctx, full)
        if problems:
            rejected.append((len(problems), _collision_area(ctx, full), kind, full))
        else:
            sc = _score(ctx, full, kind, organization=False)
            valid.append((sc, full))
            if _DEBUG is not None:
                _DEBUG.append((sc, f'{label} floors={ctx.floor_mult}', full))
    finished_valid = []
    for sc, full in sorted(valid, key=lambda item: -item[0]):
        finished, problems = _review_and_repair(ctx, full, rounds=0)
        if not problems:
            finished_valid.append((sc, finished))
        else:
            rejected.append((len(problems), _collision_area(ctx, finished), '', finished))
    if finished_valid:
        return _pick_organized(ctx, finished_valid)
    repaired = []
    for _, _, kind, full in sorted(rejected, key=lambda item: item[:2])[:6]:
        finished, problems = _review_and_repair(ctx, full)
        if not problems:
            repaired.append((_score(ctx, finished, kind, organization=False), finished))
    return _pick_organized(ctx, repaired) if repaired else None


def _pick_organized(ctx, candidates):
    """Apply organization penalties among candidates preserving text preferences.

    Use the supplied base scores to choose the readability reference.
    """
    reference = max(candidates, key=lambda item: item[0])[1]
    eligible = [(sc - _organization_penalty(ctx, full), full) for sc, full in candidates
                if _no_worse_readability(ctx, reference, full)]
    return max(eligible, key=lambda item: item[0])[1]


def _no_worse_readability(ctx, reference, candidate):
    """Organization cannot increase any text unit's preferred-size shortfall.

    The reference is the best valid candidate under the existing score, so
    this requires no sample-specific sizes or positions. Sizes above their
    preference remain free to trade off against organization.
    """
    for u in ctx.an.units:
        if u.kind not in ('text', 'pill'):
            continue
        target = min(ctx.soft.get(u.role, 8), ctx.tcap(u.role, u.font))
        a = max(0, target - reference[u.text_id]['font_size'])
        b = max(0, target - candidate[u.text_id]['font_size'])
        if b > a + 1e-6:
            return False
    return True


def _candidates(ctx):
    an = ctx.an
    units = an.units
    if not units:
        return []
    lanes = ctx.lanes
    results = []
    ch = ctx.ch
    wide = ctx.aspect >= 1.25
    tall_ok = ctx.aspect <= 2.2

    # ---- anchored (identical / nearby ratios)
    drift = abs(math.log(ctx.aspect / ctx.src_aspect))
    if drift < 0.3:
        lay = _build_anchored(ctx)
        if lay is not None:
            results.append(('anchored', lay, Opts(), 'anchored'))

    # ---- vertical stacks
    if tall_ok:
        row_sets = [('stack_source', _stack_rows_source(an), True)]
        lane_rows = _stack_rows_lanes(an)
        if [[u.uid for u in r] for r in lane_rows] != [[u.uid for u in r] for r in row_sets[0][1]]:
            row_sets.append(('stack_lanes', lane_rows, False))
        for name, rows, rhythm in row_sets:
            for compact in (False, True):
                for r in (1.0, 0.8, 0.6):
                    opts = Opts(compact=compact, r=r, r_logo=r)
                    lay = _search(lambda p: _build_stack(ctx, rows, p, opts, rhythm))
                    if lay is not None:
                        results.append((name, lay, opts, f'{name} compact={compact} r={r}'))
    # ---- extreme banners: three-zone composition
    if _zone_eligible(ctx):
        results += _zones_candidates(ctx)

    # ---- horizontal lane columns
    if wide:
        compact_set = (True,) if ch < 320 else (False, True)
        fine_opts = (False, True) if lanes['F'] else (False,)
        badge_n = len(lanes['D'])
        per_rows = sorted({None, 1, max(1, (badge_n + 1) // 2)}, key=lambda v: v or 99) if badge_n > 1 else [None]
        if ch < 320:   # banner strips: copy competes with everything for width
            grid = ((1.0, 1.0, 1.0), (4.0, 2.4, 1.0), (4.0, 4.0, 1.0), (4.0, 2.4, 1.5), (4.0, 4.0, 1.5),
                    (2.2, 3.0, 1.5), (4.0, 2.4, 2.2))
        else:
            grid = ((1.0, 1.0, 1.0), (2.2, 1.6, 1.0), (4.0, 2.4, 1.0), (2.2, 1.6, 1.5))
        inlines = (False, True) if lanes['X'] else (False,)

        def run(cols, opts, inline, iters):
            found = {}

            def build(p):
                out = _build_columns(ctx, cols, p, opts, inline)
                if out is not None:
                    found[id(out)] = p
                return out
            lay = _search(build, iters=iters)
            if lay is None:
                return None
            if not inline:
                lay = _place_floaters(ctx, dict(lay), found.get(id(lay), 0.0), opts)
                if lay is None:
                    return None
            label = '|'.join(''.join(c) for c in cols)
            kind = 'columns_inline' if inline else 'columns'
            if 'B' in label and 'C' in label and label.index('B') > label.index('C'):
                kind += '_logo_late'
            return (kind, lay, opts, f'cols {label} strip={opts.fine_strip} compact={opts.compact} '
                    f'r={opts.r}/{opts.r_logo} q={opts.q} k={opts.per_row} inline={inline}')

        # Stage 1: every template with a mid-range default, coarse search.
        stage1 = []
        for cols in _column_templates(lanes):
            for fine_strip in fine_opts:
                for inline in inlines:
                    opts = Opts(compact=compact_set[0], r=grid[1][0], r_logo=grid[1][1], q=grid[1][2],
                                per_row=per_rows[-1], fine_strip=fine_strip)
                    res = run(cols, opts, inline, 8)
                    if res is not None:
                        full = _complete(ctx, res[1])
                        if not _problems(ctx, full):
                            stage1.append((_score(ctx, full, res[0]), cols, fine_strip, inline))
                            results.append(res)
        stage1.sort(key=lambda t: -t[0])
        # Stage 2: refine imagery/action sizing and badge wrapping for the best few.
        grid_n = max((len(p[1]) for p in _pieces_for(lanes['V']) if p[0] == 'grid'), default=0)
        grid_opts = [None] + ([(grid_n + 1) // 2, (grid_n + 2) // 3] if grid_n >= 3 else [])
        for _, cols, fine_strip, inline in stage1[:6]:
            for compact in compact_set:
                for r, r_logo, q in grid:
                    for per_row in per_rows:
                        for gp in grid_opts:
                            opts = Opts(compact=compact, r=r, r_logo=r_logo, per_row=per_row,
                                        fine_strip=fine_strip, q=q, grid_per=gp)
                            res = run(cols, opts, inline, 11)
                            if res is not None:
                                results.append(res)
    return results


def _complete(ctx, out, kind=''):
    out = dict(out)
    out = _place_panels(ctx, out)
    out = _place_backgrounds(ctx, out)
    out = _place_inert(ctx, out)
    # Anchored candidates retain source-based decorative anchors, clamped to
    # the canvas, rather than searching other decorative positions.
    out = _place_decorations(ctx, out, keep_layering=(kind == 'anchored'))
    return out


# Development hook (tools/debug_candidates.py): set to a list to collect
# (base score, label, pre-repair layout) for candidates passing _problems.
# These are not final organization-adjusted rankings. Unused in production.
_DEBUG = None


def resize(source, target_canvas):
    """Return a validated reflow, or raise LayoutValidationError if search fails.

    Preserve source element IDs/count/order. Nonpositive canvas dimensions
    raise ValueError before layout search.
    """
    cw, ch = int(target_canvas['width']), int(target_canvas['height'])
    if cw <= 0 or ch <= 0:
        raise ValueError('Canvas dimensions must be positive')
    els = source['elements']
    an = analyse(source)
    ctx = Ctx(source, {'width': cw, 'height': ch}, an)
    ctx.lanes = _lanes(an)

    best = None
    # Readability floors relax in stages only when nothing fits (tiny canvases
    # with a lot of copy); the first stage that yields a valid layout wins.
    for floor_mult in (1.0, 0.8, 0.62, 0.48, 0.36, 0.25):
        ctx.floor_mult = floor_mult
        candidates = [(kind, _complete(ctx, out, kind), label)
                      for kind, out, opts, label in _candidates(ctx)]
        best = _select_reviewed(ctx, candidates)
        if best is not None:
            break
    if best is None:
        # Try alternate structures, applying the same final validation gate.
        options = [_complete(ctx, _fallback(ctx)), _complete(ctx, _grid_fallback(ctx))]
        best = _select_reviewed(ctx, [('fallback', o, 'fallback') for o in options])
        if best is None:
            reviewed = [_review_and_repair(ctx, o) for o in options]
            _, problems = min(reviewed, key=lambda item: (len(item[1]), _collision_area(ctx, item[0])))
            raise LayoutValidationError(problems)
    problems = _final_problems(ctx, best)
    if problems:
        raise LayoutValidationError(problems)

    elements = []
    for i, src in enumerate(els):
        elements.append(best[i])
    return {'canvas': {'width': cw, 'height': ch}, 'elements': elements}
