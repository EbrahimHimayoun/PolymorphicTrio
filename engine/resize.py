"""Offline, geometry-driven candidate layout engine.

Simple label/shape groups can reflow; uncertain groups remain rigid. Candidate columns preserve
source reading order, with a dynamic-programming partition rather than a fixed
number of elements per column. No decisions depend on IDs or asset names.
"""
from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path


def _bounds(e):
    a = math.radians(e.get('rotation', 0))
    w, h = e['width'], e['height']
    rw = abs(math.cos(a)) * w + abs(math.sin(a)) * h
    rh = abs(math.sin(a)) * w + abs(math.cos(a)) * h
    return e['x'] + (w-rw)/2, e['y'] + (h-rh)/2, rw, rh


def _union(elements):
    boxes = [_bounds(e) for e in elements]
    x, y = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return x, y, max(b[0]+b[2] for b in boxes)-x, max(b[1]+b[3] for b in boxes)-y


@lru_cache(maxsize=1)
def _font_manager():
    try:
        from renderer.converter.brand_fonts import load_brand_font_manager
        return load_brand_font_manager(str(Path(__file__).resolve().parents[1]/'brand_kit/fonts'))
    except ImportError:
        return None


@lru_cache(maxsize=2048)
def _advances(family, weight, italic, content):
    manager = _font_manager()
    if manager is not None:
        from renderer.converter.wrap import char_widths
        style = ('bolditalic' if weight == 'bold' else 'italic') if italic else ('bold' if weight == 'bold' else 'regular')
        return tuple(char_widths(manager, family, weight, style, 100, content))
    # Pillow is already a project dependency. This path also allows using the
    # engine alone when the optional official renderer is not installed.
    from PIL import ImageFont
    directory = Path(__file__).resolve().parents[1]/'brand_kit/fonts'
    suffix = ('BoldItalic' if weight == 'bold' else 'Italic') if italic else ('Bold' if weight == 'bold' else 'Regular')
    paths = list(directory.glob(f"{family.replace(' ', '')}-{suffix}.*"))
    font = ImageFont.truetype(str(paths[0]), 100) if paths else ImageFont.load_default(size=100)
    return tuple(font.getlength(c) for c in content)


def _text_size(e, width, size):
    advances = list(_advances(e['font_family'], e.get('font_weight', 'regular'), e.get('font_style') == 'italic', e['content']))
    for run in e.get('style_runs') or []:
        variant = _advances(e['font_family'], run.get('font_weight', e.get('font_weight', 'regular')), run.get('font_style', e.get('font_style')) == 'italic', e['content'])
        # The current converter applies the end index inclusively. Measure both
        # the schema range and that extra glyph conservatively, without edits.
        for i in range(max(0, run['start']), min(len(advances), run['end']+1)):
            advances[i] = max(advances[i], variant[i])
    widths = [v*size/100 + e.get('letter_spacing', 0) for v in advances]
    # Reserve a little width for glyph bearings and styled runs.
    wrapped = _wrap(e['content'], widths, max(.01, width*.96))
    lines = wrapped.split('\n')
    longest = max((sum(widths[start:start+len(line)]) for start, line in _line_offsets(lines)), default=0)
    return longest, len(lines)*size*e.get('line_height', 1.2), len(lines)


def _wrap(content, widths, width):
    # Match the official converter's space-only wrap without importing skia.
    chars, used, space = list(content), 0, None
    for i, char in enumerate(chars):
        if char == '\n':
            used, space = 0, None
            continue
        if used+widths[i] > width and space is not None:
            chars[space] = '\n'
            used = sum(widths[space+1:i+1])
            space = None
        else:
            used += widths[i]
        if char == ' ':
            space = i
    return ''.join(chars)


def _line_offsets(lines):
    start = 0
    for line in lines:
        yield start, line
        start += len(line)+1


def _scale(e, factor):
    for key in ('width', 'height', 'corner_radius', 'stroke_width', 'letter_spacing'):
        if key in e:
            e[key] *= factor
    if e['type'] == 'text':
        e['font_size'] = max(1, e['font_size']*factor)


def _fit_unit(elements, box):
    """Translate/scale the entire unit once, preserving internal overlaps."""
    x, y, w, h = _union(elements)
    bx, by, bw, bh = box
    factor = min(1, bw/max(w, .001), bh/max(h, .001))
    nx = min(max(x+(w-w*factor)/2, bx), bx+bw-w*factor)
    ny = min(max(y+(h-h*factor)/2, by), by+bh-h*factor)
    for e in elements:
        ex, ey = e['x'], e['y']
        _scale(e, factor)
        e['x'] = nx+(ex-x)*factor
        e['y'] = ny+(ey-y)*factor


def _units(source):
    elements = source['elements']
    sw, sh = source['canvas']['width'], source['canvas']['height']
    backgrounds, decorations, foreground = [], [], []
    for i, e in enumerate(elements):
        covers = e['x'] <= 0 and e['y'] <= 0 and e['width'] >= sw*.98 and e['height'] >= sh*.98
        if covers and e['type'] != 'text' and not e.get('group_id') and e.get('z_index', 0) <= min((f.get('z_index', 0) for f in elements if f['type']=='text'), default=0):
            backgrounds.append(i)
        elif e['type']=='shape' and not e.get('group_id') and e.get('opacity', 1) < .4:
            decorations.append(i)
        else:
            foreground.append(i)
    parent = {i:i for i in foreground}
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for a, i in enumerate(foreground):
        e = elements[i]
        for j in foreground[a+1:]:
            f = elements[j]
            grouped = e.get('group_id') is not None and e.get('group_id') == f.get('group_id')
            # Infer strong text containment in a shape or image, not arbitrary overlap:
            # two intersecting text boxes in the source may already be a defect.
            pair = 'text' in (e['type'], f['type']) and e['type'] != f['type']
            shape, text = (f, e) if e['type']=='text' else (e, f)
            contained = pair and not e.get('group_id') and not f.get('group_id') and shape['x'] <= text['x'] and shape['y'] <= text['y'] and shape['x']+shape['width'] >= text['x']+text['width'] and shape['y']+shape['height'] >= text['y']+text['height']
            if grouped or contained:
                parent[root(j)] = root(i)
    grouped = {}
    for i in foreground:
        grouped.setdefault(root(i), []).append(i)
    return sorted(grouped.values(), key=lambda ids: (_union([elements[i] for i in ids])[1], _union([elements[i] for i in ids])[0])), backgrounds, decorations


def _materialize(original, ids, width, scale):
    items = [deepcopy(original[i]) for i in ids]
    if len(items)==1 and items[0]['type']=='text' and not items[0].get('rotation', 0):
        e = items[0]
        _scale(e, scale)
        e['width'] = width
        longest, height, _ = _text_size(e, width, e['font_size'])
        if longest > width:
            e['font_size'] = max(1, e['font_size']*width/longest*.96)
            _, height, _ = _text_size(e, width, e['font_size'])
        e['height'] = height*1.08
        e['x'] = e['y'] = 0
    else:
        x, y, w, h = _union(items)
        factor = min(scale, width/max(w, .001))
        for e in items:
            e['x'], e['y'] = (e['x']-x)*factor, (e['y']-y)*factor
            _scale(e, factor)
    return items


def _partition(heights, columns, gap):
    """Contiguous reading-order partition minimizing the tallest column."""
    n = len(heights)
    prefix = [0]
    for h in heights:
        prefix.append(prefix[-1]+h+gap)
    dp = {(0, 0):(0, [])}
    for c in range(1, columns+1):
        for end in range(c, n+1):
            choices = [(max(dp[c-1, start][0], prefix[end]-prefix[start]-gap), dp[c-1, start][1]+[(start, end)]) for start in range(c-1, end) if (c-1, start) in dp]
            dp[c, end] = min(choices, key=lambda v:v[0])
    return dp[columns, n]


def _score(original, output, units, reference, prefs=None):
    prefs = prefs or {}
    score = 0
    for ids in units:
        for i in ids:
            e, src = output[i], original[i]
            if e['type']=='text':
                w, h, lines = _text_size(e, e['width'], e['font_size'])
                score += 40*max(0, h/max(e['height'], .001)-1) + 40*max(0, w/max(e['width'], .001)-1)
                score += 10*max(0, lines-(e.get('max_lines') or lines))
                ratio = e['font_size']/max(1, src['font_size']*reference)
                score += abs(math.log(max(.001, ratio))) * 2
                minimum, importance, _ = prefs.get(i,(9,1,'normal'))
                score += 140*importance*(max(0,minimum-e['font_size'])/minimum)**2
            else:
                ratio = math.sqrt(e['width']*e['height']/max(.001, src['width']*src['height']))/reference
                score += abs(math.log(max(.001, ratio)))
    text_ids = [i for ids in units for i in ids if original[i]['type']=='text']
    for i in text_ids:
        for j in text_ids:
            # Enforce only clearly signalled source hierarchy, not an assumed
            # semantic order. Equal-size source text is free to remain equal.
            if original[i]['font_size'] >= original[j]['font_size']*1.45:
                ratio = output[i]['font_size']/max(1,output[j]['font_size'])
                score += 40*max(0,1.18-ratio)**2
    # Different units should not collide. Layering inside a unit is intentional.
    for a, ids in enumerate(units):
        for other in units[a+1:]:
            for i in ids:
                for j in other:
                    x,y,w,h = _bounds(output[i]); xx,yy,ww,hh = _bounds(output[j])
                    overlap = max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
                    score += 80*overlap/max(1,min(w*h,ww*hh))
    return score


# Engineering preferences in output pixels, not challenge constraints. Applications
# can tune this mapping without changing resize(source, target_canvas).
PREFERRED_TEXT_SIZES = {'prominent': 14.0, 'normal': 11.0,
                        'secondary': 9.0, 'footer': 7.0}


def _text_preferences(source, units):
    texts = [e for e in source['elements'] if e['type'] == 'text']
    if not texts:
        return {}
    sizes = sorted(e['font_size'] for e in texts)
    median, largest = sizes[len(sizes)//2], sizes[-1]
    sh = source['canvas']['height']
    preferences = {}
    for ids in units:
        for i in ids:
            e = source['elements'][i]
            if e['type'] != 'text':
                continue
            role = 'normal'
            # Require several geometric/typographic signals before treating copy
            # as a footer. Words, IDs, notes and asset names are never inspected.
            if (len(ids) == 1 and e['y'] >= .90*sh
                    and e['height'] <= .045*sh
                    and e['font_size'] <= .65*largest
                    and e['font_size'] <= .75*median):
                role = 'footer'
            elif e['font_size'] >= 1.45*median and e['font_size'] >= .9*largest:
                role = 'prominent'
            elif len(ids) > 1 and e['font_size'] < .8*median:
                role = 'secondary'
            preferences[i] = (PREFERRED_TEXT_SIZES[role],
                              2.5 if role == 'prominent' else 1.0, role)
    return preferences


def readability_report(source, output):
    """Diagnostics kept outside the schema; shortfalls are reported, not hidden."""
    units, _, _ = _units(source)
    prefs = _text_preferences(source, units)
    return [{'id': source['elements'][i]['id'], 'role': role,
             'font_size': output['elements'][i]['font_size'], 'preferred': minimum,
             'shortfall': max(0, minimum-output['elements'][i]['font_size'])}
            for i, (minimum, _, role) in prefs.items()]


def _simple_group(original, ids):
    if len(ids) != 2:
        return None
    texts = [i for i in ids if original[i]['type'] == 'text']
    shapes = [i for i in ids if original[i]['type'] == 'shape']
    if len(texts) != 1 or len(shapes) != 1:
        return None
    t, s = original[texts[0]], original[shapes[0]]
    # Rotated pairs are flexible only when a centered label and rectangle
    # share exactly one pivot/angle. Otherwise their relationship is uncertain.
    if s.get('shape_type') != 'rectangle':
        return None
    if t.get('rotation',0) or s.get('rotation',0):
        same_pivot = (abs(t['x']+t['width']/2-s['x']-s['width']/2) < .01
                      and abs(t['y']+t['height']/2-s['y']-s['height']/2) < .01)
        if (t.get('rotation',0) != s.get('rotation',0)
                or t.get('text_align') != 'center' or not same_pivot):
            return None
    if (t['x'] < s['x']-.01 or t['y'] < s['y']-.01
            or t['x']+t['width'] > s['x']+s['width']+.01
            or t['y']+t['height'] > s['y']+s['height']+.01):
        return None
    return texts[0], shapes[0]


def _desired_size(e, preference, reference, pressure):
    minimum, _, role = preference
    ideal = max(minimum, e['font_size']*reference)
    # Under pressure, the largest source typography loses proportionally less.
    # Footer/secondary copy can give up space sooner, but still incurs a penalty.
    exponent = {'prominent': .72, 'normal': .95, 'secondary': 1.10, 'footer': 1.15}[role]
    return max(1, ideal*pressure**exponent)


def _adaptive_block(original, ids, width, reference, pressure, prefs, compact=True):
    simple = _simple_group(original, ids)
    text_id = ids[0] if len(ids) == 1 and original[ids[0]]['type'] == 'text' and not original[ids[0]].get('rotation', 0) else None
    if simple:
        text_id = simple[0]
    if text_id is None:
        text_ids = [i for i in ids if original[i]['type'] == 'text']
        if text_ids:
            factor = max(_desired_size(original[i], prefs[i], reference, pressure)/original[i]['font_size'] for i in text_ids)
        else:
            factor = reference*pressure**1.1
        return _materialize(original, ids, width, factor)
    t = deepcopy(original[text_id])
    size = _desired_size(t, prefs[text_id], reference, pressure)
    spacing_scale = size/max(1,t['font_size'])
    if 'letter_spacing' in t:
        t['letter_spacing'] *= spacing_scale
    px = py = 0
    if simple:
        src_shape = original[simple[1]]
        # Keep genuine padding, while compacting large empty label boxes.
        px = max(size*.55, min(size*1.5, (src_shape['width']-original[text_id]['width'])/2*spacing_scale))
        py = max(size*.22, min(size*.6, (src_shape['height']-original[text_id]['height'])/2*spacing_scale))
    available = max(.1,width-2*px)
    if compact and simple:
        natural, _, _ = _text_size(t, 1e9, size)
        available = min(available, natural/ .94+size*.25)
    for _ in range(8):
        longest, height, lines = _text_size(t, available, size)
        if longest <= available and (not t.get('max_lines') or lines <= t['max_lines']):
            break
        size *= .9
    t.update(x=px,y=py,width=available,height=height*1.04,font_size=size)
    if not simple:
        return [t]
    shape = deepcopy(original[simple[1]])
    shape.update(x=0,y=0,width=available+2*px,height=t['height']+2*py)
    if 'corner_radius' in shape:
        shape['corner_radius'] *= shape['height']/max(.01,original[simple[1]]['height'])
    if 'stroke_width' in shape:
        shape['stroke_width'] *= spacing_scale
    block = [t if i == text_id else shape for i in ids]
    # A rotated compact button's enclosing width exceeds its local width.
    # Scale the pair once after reflow; never distort or detach its label.
    _,_,rotated_width,_ = _union(block)
    if rotated_width > width:
        factor = width/rotated_width
        for e in block:
            e['x'] *= factor
            e['y'] *= factor
            _scale(e,factor)
    return block


def _pack_column(original, column, width, reference, pressure, prefs, gap):
    """Flow compact simple groups into rows; other units get their own row."""
    rows, row, used = [], [], 0
    for ids in column:
        block = _adaptive_block(original,ids,width,reference,pressure,prefs)
        _,_,w,h = _union(block)
        compact = _simple_group(original,ids) is not None
        if row and (not compact or used+gap+w > width):
            rows.append(row)
            row,used = [],0
        row.append((ids,block,w,h))
        used += w+(gap if len(row)>1 else 0)
        if not compact:
            rows.append(row)
            row,used = [],0
    if row:
        rows.append(row)
    height = sum(max(item[3] for item in row) for row in rows)+gap*max(0,len(rows)-1)
    return rows,height


def _place_rows(output, rows, x, y, width, gap):
    for row in rows:
        rh = max(item[3] for item in row)
        used = sum(item[2] for item in row)+gap*(len(row)-1)
        xx = x+(width-used)/2
        for ids,block,w,h in row:
            x0,y0,_,_ = _union(block)
            for i,e in zip(ids,block):
                e = deepcopy(e)
                e['x'] += xx-x0
                e['y'] += y+(rh-h)/2-y0
                output[i] = e
            xx += w+gap
        y += rh+gap


def _missing_asset(element):
    """Use the same local assets directory as the official pipeline."""
    if element['type'] != 'image':
        return False
    asset = element.get('asset_id')
    return not asset or not (Path(__file__).resolve().parents[1]/'brand_kit/assets'/asset).is_file()


def _grow_missing_images(source, output, units, reference, cw, ch):
    """Give unresolved standalone images useful space without changing pixels.

    Search nearby collision-free positions at source-proportional optical sizes.
    Complex/grouped images retain their associations and are not moved alone.
    """
    current = deepcopy(output)
    gap = min(cw,ch)*.015
    for ids in units:
        if len(ids) != 1 or not _missing_asset(source['elements'][ids[0]]):
            continue
        i = ids[0]
        src,e = source['elements'][i],current[i]
        if not src['width'] or not src['height']:
            continue
        existing = e['width']/src['width']
        obstacles = [_bounds(current[j]) for other in units if other != ids for j in other]
        best,best_cost = current,None
        # Prefer the largest useful size, with smaller options when constrained.
        for factor in sorted({max(existing,reference*1.25),max(existing,reference),existing},reverse=True):
            resized = deepcopy(src)
            _scale(resized,factor)
            resized['x']=resized['y']=0
            ox,oy,w,h = _bounds(resized)
            if w > cw or h > ch:
                continue
            ex,ey,ew,eh = _bounds(e)
            xs = {max(0,min(cw-w,ex+(ew-w)/2)),0,cw-w}
            ys = {max(0,min(ch-h,ey+(eh-h)/2)),0,ch-h}
            for x,y,ww,hh in obstacles:
                xs.update((x-w-gap,x+ww+gap))
                ys.update((y-h-gap,y+hh+gap))
            for x in xs:
                for y in ys:
                    if x < 0 or y < 0 or x+w > cw or y+h > ch:
                        continue
                    if any(x < bx+bw+gap and x+w+gap > bx and y < by+bh+gap and y+h+gap > by for bx,by,bw,bh in obstacles):
                        continue
                    distance=((x+w/2-ex-ew/2)/cw)**2+((y+h/2-ey-eh/2)/ch)**2
                    cost=(-factor,distance)
                    if best_cost is not None and cost >= best_cost:
                        continue
                    trial=deepcopy(current)
                    trial[i]=dict(resized,x=x-ox,y=y-oy)
                    if _safe_change(current,trial,units,cw,ch):
                        best,best_cost=trial,cost
            if best_cost is not None:
                break
        current=best
    return current


def _adaptive_candidates(source, units, reference, cw, ch, prefs):
    if not units:
        return
    original = source['elements']
    margin = min(cw,ch)*.025
    # Confident footers gain a full-width strip instead of competing with
    # main text for the narrow remainder of a sidebar.
    footers = [ids for ids in units if len(ids)==1 and prefs.get(ids[0],(0,0,''))[2]=='footer']
    body = [ids for ids in units if ids not in footers]
    if not body:
        body,footers = footers,[]
    plans = [[body]]
    # Contiguous partitions retain reading order; widths then follow content demand.
    for count in range(2,min(5,len(body))+1):
        hs = [_union([original[i] for i in ids])[3] for ids in body]
        _,partitions = _partition(hs,count,0)
        plans.append([body[a:b] for a,b in partitions])
    # Also consider spatially independent visual, text and compact-control lanes.
    # This works without knowing whether any image is a logo or product.
    visual,copy,controls = [],[],[]
    for ids in body:
        if _simple_group(original,ids):
            controls.append(ids)
        elif len(ids)==1 and original[ids[0]]['type']=='text' and not original[ids[0]].get('rotation',0):
            copy.append(ids)
        else:
            visual.append(ids)
    if cw/ch > 1.4 and copy and (visual or controls):
        plans.append([lane for lane in (visual,copy,controls) if lane])
        if any(len(ids)==1 and _missing_asset(original[ids[0]]) for ids in visual):
            # An unresolved image is still real content. Offer it space with
            # similarly proportioned visual units instead of letting it trail
            # the control stack. Never rename, replace or suppress its asset.
            wide,other = [],[]
            for ids in visual:
                _,_,w,h = _union([original[i] for i in ids])
                (wide if w/max(.001,h) >= 1.8 else other).append(ids)
            if wide and other:
                plans.append([lane for lane in (wide,other,copy,controls) if lane])
    for plan in plans:
        count = len(plan)
        if count > 1 and cw/ch < .65:
            continue
        demands = []
        for column in plan:
            area = 0
            for ids in column:
                for i in ids:
                    e = original[i]
                    if e['type']=='text':
                        size = prefs[i][0]
                        natural,_,_ = _text_size(e,1e9,size)
                        area += natural*size*e.get('line_height',1.2)*prefs[i][1]
                    elif e['type']=='image':
                        area += min(ch*.70,e['height']*reference)**2*e['width']/max(.01,e['height'])
            demands.append(max(area, ch*ch*.08))
        weights = [d**.65 for d in demands]
        weight_sets = [[1]*count,weights]
        # A text-heavy column may need most of the width. Explore multiple
        # allocations, allowing the fit/scoring stage to decide by measurements.
        if count > 1:
            weight_sets += [[w*(1.5 if i==j else 1) for i,w in enumerate(weights)] for j in range(count)]
        for weights in weight_sets:
            for gap_fraction in (.018,.007):
                gap = min(cw,ch)*gap_fraction
                widths = [(cw-2*margin-gap*(count-1))*w/sum(weights) for w in weights]
                lo,hi,best = 0,1.2,None
                for _ in range(17):
                    pressure = (lo+hi)/2
                    packed = [_pack_column(original,column,width,reference,pressure,prefs,gap) for column,width in zip(plan,widths)]
                    footer_rows,footer_h = _pack_column(original,footers,cw-2*margin,reference,pressure,prefs,gap) if footers else ([],0)
                    available = ch-2*margin-(footer_h+gap if footers else 0)
                    if all(h <= available for _,h in packed):
                        lo,best = pressure,(packed,footer_rows,footer_h,available)
                    else:
                        hi = pressure
                if best is None:
                    continue
                packed,footer_rows,footer_h,available = best
                output = deepcopy(original)
                x = margin
                for width,(rows,height) in zip(widths,packed):
                    _place_rows(output,rows,x,margin+(available-height)/2,width,gap)
                    x += width+gap
                if footers:
                    _place_rows(output,footer_rows,margin,ch-margin-footer_h,cw-2*margin,gap)
                yield output


def _overlap(a,b):
    x,y,w,h = _bounds(a); xx,yy,ww,hh = _bounds(b)
    return max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))


def _safe_change(before, after, units, cw, ch):
    for ids in units:
        for i in ids:
            e = after[i]
            x,y,w,h = _bounds(e)
            if x < -1e-6 or y < -1e-6 or x+w > cw+1e-6 or y+h > ch+1e-6:
                return False
            if e['type']=='text':
                tw,th,lines = _text_size(e,e['width'],e['font_size'])
                if tw > e['width']+1e-6 or th > e['height']+1e-6 or (e.get('max_lines') and lines > e['max_lines']):
                    return False
    for a,ids in enumerate(units):
        for other in units[a+1:]:
            for i in ids:
                for j in other:
                    if _overlap(after[i],after[j]) > _overlap(before[i],before[j])+1e-6:
                        return False
    return True


def _grow_text(source, output, units, reference, cw, ch, prefs):
    """Try real free rectangles; accept only fully measured, collision-free growth.

    A rotated or complex group can grow rigidly, never as detached label text.
    A simple button grows its shape, padding and label as a single transaction.
    """
    current = deepcopy(output)
    gap = min(cw,ch)*.007
    order = sorted(units,key=lambda ids:max((prefs[i][1] for i in ids if i in prefs),default=0),reverse=True)
    for ids in order:
        text_ids = [i for i in ids if i in prefs]
        if not text_ids:
            continue
        old = [current[i] for i in ids]
        x,y,w,h = _union(old)
        obstacles = [_bounds(current[i]) for other in units if other != ids for i in other]
        # Alternate horizontal/vertical expansion, since each exposes a different
        # free rectangle in staggered layouts. Keep original text alignment.
        rectangles = [(x,y,w,h)]
        for horizontal_first in (True,False):
            l,t,r,b = x,y,x+w,y+h
            for horizontal in (horizontal_first,not horizontal_first):
                if horizontal:
                    l,r = 0,cw
                    for ox,oy,ow,oh in obstacles:
                        if oy < b+gap and oy+oh > t-gap:
                            if ox+ow <= x+1e-6: l=max(l,ox+ow+gap)
                            elif ox >= x+w-1e-6: r=min(r,ox-gap)
                            else: l,r=max(l,x),min(r,x+w)
                else:
                    t,b = 0,ch
                    for ox,oy,ow,oh in obstacles:
                        if ox < r+gap and ox+ow > l-gap:
                            if oy+oh <= y+1e-6: t=max(t,oy+oh+gap)
                            elif oy >= y+h-1e-6: b=min(b,oy-gap)
                            else: t,b=max(t,y),min(b,y+h)
            if r>l and b>t:
                rectangles.append((l,t,r-l,b-t))
        best = current
        best_score = _score(source['elements'],current,units,reference,prefs)
        simple = _simple_group(source['elements'],ids)
        standalone = len(ids)==1 and not old[0].get('rotation',0)
        for bx,by,bw,bh in rectangles:
            if standalone or simple:
                ti = text_ids[0]
                # Cap growth by source hierarchy / intended optical scale, not
                # an arbitrary headline assumption. Never enlarge to fill all
                # available space when the source was intentionally modest.
                cap = max(prefs[ti][0],source['elements'][ti]['font_size']*reference)
                # Growth must not reverse a clear source hierarchy just because
                # a small label happens to have more nearby free space.
                for other in prefs:
                    if source['elements'][other]['font_size'] >= source['elements'][ti]['font_size']*1.45:
                        cap = min(cap,max(current[ti]['font_size'],current[other]['font_size']/1.18))
                low,high = current[ti]['font_size'],max(current[ti]['font_size'],cap)
                block_best = None
                for _ in range(18):
                    size=(low+high)/2
                    custom = deepcopy(source['elements'])
                    # Materializer at pressure 1 gives exactly the requested size.
                    custom[ti]['font_size'] = size
                    block = _adaptive_block(custom,ids,bw,1,1,{**prefs,ti:(size,prefs[ti][1],prefs[ti][2])},compact=False)
                    _,_,ww,hh = _union(block)
                    if hh <= bh and ww <= bw and all(e['type']!='text' or _text_size(e,e['width'],e['font_size'])[0] <= e['width'] for e in block):
                        low,block_best = size,block
                    else:
                        high = size
                if block_best is None or low <= current[ti]['font_size']+.01:
                    continue
                block = block_best
            else:
                factor = min(bw/max(w,.001),bh/max(h,.001),1.5)
                if factor <= 1.001:
                    continue
                block = deepcopy(old)
                for e in block:
                    e['x'],e['y']=(e['x']-x)*factor,(e['y']-y)*factor
                    _scale(e,factor)
            xx,yy,ww,hh = _union(block)
            # Preserve left/right/center alignment about the unit's original
            # anchor, clamped to the free rectangle only when necessary.
            align = current[text_ids[0]].get('text_align','left')
            anchor = x if align=='left' else x+w-ww if align=='right' else x+(w-ww)/2
            nx = min(max(anchor,bx),bx+bw-ww)
            ny = min(max(y+(h-hh)/2,by),by+bh-hh)
            trial = deepcopy(current)
            for i,e in zip(ids,block):
                e['x'] += nx-xx; e['y'] += ny-yy
                trial[i]=e
            score = _score(source['elements'],trial,units,reference,prefs)
            if score < best_score and _safe_change(current,trial,units,cw,ch):
                best,best_score = trial,score
        current=best
    return current


def _place_decorations(source, output, units, decorations, cw, ch):
    original = source['elements']
    sw,sh=source['canvas']['width'],source['canvas']['height']
    for i in decorations:
        src,e = original[i],output[i]
        source_area = max(.001,src['width']*src['height'])
        associations=[]
        for ids in units:
            # Text alone is not a decorative anchor. Significant overlap with
            # imagery, lower stacking and an unambiguous winner are required.
            for j in ids:
                f=original[j]
                if f['type']=='image' and src.get('z_index',0)<f.get('z_index',0):
                    ratio=_overlap(src,f)/min(source_area,max(.001,f['width']*f['height']))
                    associations.append((ratio,j))
        associations.sort(reverse=True)
        anchor = associations[0][1] if associations and associations[0][0]>.22 and (len(associations)==1 or associations[0][0] > associations[1][0]*1.6) else None
        if anchor is not None:
            a,b=original[anchor],output[anchor]
            factor=b['width']/max(.001,a['width'])
            _scale(e,factor)
            e['x']=b['x']+(src['x']-a['x'])*factor
            e['y']=b['y']+(src['y']-a['y'])*factor
        else:
            factor=min(cw/sw,ch/sh)
            _scale(e,factor)
            e['x']=(src['x']+src['width']/2)/sw*cw-e['width']/2
            e['y']=(src['y']+src['height']/2)/sh*ch-e['height']/2
        _fit_unit([e],(0,0,cw,ch))
        # Retain opacity/style and choose a nearby clear location when a moved
        # decoration would cover text. Canvas anchors prefer their source corner.
        texts=[output[j] for ids in units for j in ids if output[j]['type']=='text']
        x,y,w,h=_bounds(e)
        positions=[(e['x'],e['y'])]
        positions += [(xx+(w-e['width'])/2,yy+(h-e['height'])/2)
                      for xx in (0,max(0,cw-w)) for yy in (0,max(0,ch-h))]
        def cost(position):
            trial=dict(e,x=position[0],y=position[1])
            occlusion=sum(_overlap(trial,t)/max(1,t['width']*t['height']) for t in texts)
            distance=((position[0]-e['x'])/cw)**2+((position[1]-e['y'])/ch)**2
            return 20*occlusion+distance
        px,py=min(positions,key=cost)
        e['x'],e['y']=px,py

def resize(source, target_canvas):
    """Return a new design, preserving IDs, assets, styles and stacking order."""
    cw, ch = target_canvas['width'], target_canvas['height']
    if cw <= 0 or ch <= 0:
        raise ValueError('Canvas dimensions must be positive')
    original = source['elements']
    units, backgrounds, decorations = _units(source)
    sw, sh = source['canvas']['width'], source['canvas']['height']
    reference = math.sqrt(cw*ch/(sw*sh))
    prefs = _text_preferences(source,units)
    margin = min(cw, ch)*.04
    gap = min(cw, ch)*.025
    candidates = []
    # Preserve spatial anchors for nearby shapes, while scaling units uniformly.
    preserved = deepcopy(original)
    for ids in units:
        x,y,w,h = _union([original[i] for i in ids])
        items = [preserved[i] for i in ids]
        factor = min(cw/sw, ch/sh)
        for e in items:
            ex,ey = e['x'],e['y']
            _scale(e, factor)
            e['x'] = (x+w/2)*cw/sw-w*factor/2+(ex-x)*factor
            e['y'] = (y+h/2)*ch/sh-h*factor/2+(ey-y)*factor
        _fit_unit(items, (0,0,cw,ch))
    aspect_change = abs(math.log((cw/ch)/(sw/sh)))
    candidates.append((_score(original,preserved,units,reference,prefs)+aspect_change*len(units)*3, preserved))
    # Search widths and scales, not specific format names or target dimensions.
    for columns in range(1, min(len(units), max(1, min(8, math.ceil(cw/ch*1.5))))+1):
        width = (cw-2*margin-gap*(columns-1))/columns
        if width <= 0:
            continue
        lo,hi = 0, reference*1.25
        best = None
        for _ in range(22):
            scale = (lo+hi)/2
            blocks = [_materialize(original, ids, width, scale) for ids in units]
            heights = [_union(block)[3] for block in blocks]
            used, partitions = _partition(heights,columns,gap)
            if used <= ch-2*margin:
                lo = scale
                best = blocks, heights, partitions
            else:
                hi = scale
        if best is None:
            continue
        blocks, heights, partitions = best
        output = deepcopy(original)
        for col,(start,end) in enumerate(partitions):
            used = sum(heights[start:end])+gap*(end-start-1)
            y = margin+(ch-2*margin-used)/2
            for index in range(start,end):
                block = blocks[index]
                x0,y0,w,h = _union(block)
                for i,e in zip(units[index],block):
                    e['x'] += margin+col*(width+gap)+(width-w)/2-x0
                    e['y'] += y-y0
                    output[i] = e
                y += h+gap
        score = _score(original,output,units,reference,prefs)
        # Favor source anchors for nearby ratios, unless they cause collisions.
        score += max(0, .4-aspect_change)*len(units)*2
        candidates.append((score,output))
    # Keep the accepted nearby-ratio baseline when it already meets text fit,
    # readability and collision checks. Spare space need not become huge type.
    baseline = min(candidates,key=lambda candidate:candidate[0])[1]
    preserve_baseline = (aspect_change < .35
        and all(baseline[i]['font_size'] >= pref[0] for i,pref in prefs.items())
        and _safe_change(baseline,baseline,units,cw,ch)
        and all(_overlap(baseline[i],baseline[j]) < 1e-6
                for a,ids in enumerate(units) for other in units[a+1:]
                for i in ids for j in other))
    if not preserve_baseline:
        for adaptive in _adaptive_candidates(source,units,reference,cw,ch,prefs):
            if _safe_change(adaptive,adaptive,units,cw,ch):
                candidates.append((_score(original,adaptive,units,reference,prefs),adaptive))
    output = baseline if preserve_baseline else min(candidates,key=lambda candidate:candidate[0])[1]
    if not preserve_baseline:
        output = _grow_text(source,output,units,reference,cw,ch,prefs)
    output = _grow_missing_images(source,output,units,reference,cw,ch)
    for i in backgrounds:
        e = output[i]
        e.update(x=0,y=0,width=cw,height=ch)
        if e['type']=='image':
            e['fit']='cover'  # full-bleed crop; never distort asset pixels
        _fit_unit([e], (0,0,cw,ch))
    _place_decorations(source,output,units,decorations,cw,ch)
    return {'canvas': {'width':cw,'height':ch}, 'elements':output}
