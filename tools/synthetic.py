"""Generate adversarial synthetic task files for robustness sweeps (dev aid).

    python -m tools.synthetic /path/to/dir
    python -m tools.validate --inputs /path/to/dir

Covers structures the three samples do not: landscape sources, text-only
designs, image galleries, rotated free text, ellipse badges with text, panels,
divider lines, style runs, letter spacing, max_lines, unknown font families,
empty strings, zero-size elements, image+caption cards and very busy designs.
"""
import json
import random
import sys
from pathlib import Path

W = 'rgba(255,255,255,1)'
ORANGE = 'rgba(232,106,58,1)'
DARK = 'rgba(43,27,46,1)'


def text(id, x, y, w, h, content, size, family='Solstice Sans', **kw):
    e = dict(id=id, type='text', x=x, y=y, width=w, height=h, content=content,
             font_family=family, font_size=size, color=kw.pop('color', W), z_index=kw.pop('z', 5))
    e.update(kw)
    return e


def image(id, x, y, w, h, asset, z=3, **kw):
    e = dict(id=id, type='image', x=x, y=y, width=w, height=h, asset_id=asset, z_index=z, fit='contain')
    e.update(kw)
    return e


def shape(id, x, y, w, h, fill=ORANGE, kind='rectangle', z=2, **kw):
    e = dict(id=id, type='shape', x=x, y=y, width=w, height=h, shape_type=kind, fill_color=fill, z_index=z)
    e.update(kw)
    return e


def bg(w, h):
    return image('bg', 0, 0, w, h, 'solstice_bg_gradient.png', z=0, fit='cover')


def landscape():
    return {'canvas': {'width': 1920, 'height': 1080}, 'elements': [
        bg(1920, 1080),
        image('product', 120, 140, 800, 800, 'solstice_bottle.png'),
        image('logo', 1040, 120, 360, 120, 'solstice_logo_lockup.png'),
        text('headline', 1040, 300, 780, 220, 'Chase the Light', 96, 'Solstice Serif', font_weight='bold',
             text_align='left', style_runs=[{'start': 6, 'end': 15, 'color': 'rgba(240,178,66,1)'}]),
        text('subhead', 1040, 540, 760, 120, 'Botanical sparkling infusions, crafted for golden hour.', 36,
             text_align='left'),
        shape('cta_bg', 1040, 720, 460, 100, corner_radius=50, group_id='cta'),
        text('cta_label', 1040, 748, 460, 50, 'Shop the Summer Line', 34, font_weight='bold',
             text_align='center', group_id='cta'),
    ]}


def text_only():
    return {'canvas': {'width': 1080, 'height': 1080}, 'elements': [
        shape('bg', 0, 0, 1080, 1080, fill=DARK, z=0),
        text('eyebrow', 100, 120, 880, 40, 'SUMMER 2026 COLLECTION', 26, font_weight='bold',
             text_align='center', letter_spacing=6, color=ORANGE),
        text('headline', 100, 220, 880, 360, 'Every sunset deserves a toast worth remembering', 84,
             'Solstice Serif', font_weight='bold', text_align='center', max_lines=4),
        shape('divider', 440, 640, 200, 6, fill=ORANGE, kind='line', stroke_color=ORANGE, stroke_width=6),
        text('body', 160, 700, 760, 140, 'Five botanical blends. Zero compromises. Available in stores and online '
             'while the light lasts.', 30, text_align='center', font_style='italic'),
        text('footer', 100, 1000, 880, 30, 'solstice.example', 20, text_align='center'),
    ]}


def gallery():
    els = [bg(1080, 1350), text('title', 80, 60, 920, 120, 'Meet the Golden Hour Lineup', 64,
                                'Solstice Serif', font_weight='bold', text_align='center')]
    for k in range(6):
        x, y = 80 + (k % 3) * 320, 240 + (k // 3) * 470
        els.append(shape(f'card{k}', x, y, 280, 420, fill='rgba(255,255,255,0.08)', corner_radius=24,
                         group_id=f'c{k}', z=2))
        els.append(image(f'img{k}', x + 40, y + 30, 200, 260, 'solstice_bottle.png', group_id=f'c{k}'))
        els.append(text(f'cap{k}', x + 20, y + 320, 240, 70, ['Blood Orange', 'Wild Honey', 'Rosemary',
                                                            'Yuzu Spritz', 'Hibiscus', 'Night Pear'][k],
                        28, text_align='center', group_id=f'c{k}'))
    els.append(shape('cta_bg', 340, 1230, 400, 80, corner_radius=40, group_id='cta'))
    els.append(text('cta_label', 340, 1250, 400, 40, 'Shop all six', 28, font_weight='bold',
                    text_align='center', group_id='cta'))
    return {'canvas': {'width': 1080, 'height': 1350}, 'elements': els}


def odd_bits():
    return {'canvas': {'width': 1080, 'height': 1080}, 'elements': [
        bg(1080, 1080),
        shape('panel', 60, 560, 960, 460, fill='rgba(43,27,46,1)', corner_radius=30, z=1),
        image('logo', 60, 60, 300, 100, 'solstice_logo_lockup.png'),
        text('stamp_text', 760, 80, 260, 60, 'NEW!', 44, 'Solstice Serif', font_weight='bold', rotation=15,
             text_align='center', color=ORANGE),
        shape('seal', 820, 260, 180, 180, fill=ORANGE, kind='ellipse', group_id='seal', z=3),
        text('seal_text', 840, 320, 140, 60, '30% OFF', 30, font_weight='bold', text_align='center',
             group_id='seal'),
        image('product', 340, 180, 360, 360, 'solstice_bottle.png'),
        text('headline', 120, 600, 840, 120, 'Summer Sale', 80, 'Solstice Serif', font_weight='bold',
             text_align='center', z=4),
        text('body', 160, 740, 760, 110, 'Mix & match any three flavours. Offer ends Sunday.', 30,
             text_align='center', z=4, line_height=1.4),
        text('empty', 500, 870, 80, 20, '', 18, z=4),
        shape('zero', 540, 900, 0, 0, fill=ORANGE, z=4),
        shape('btn_bg', 360, 900, 360, 80, fill=ORANGE, corner_radius=12, group_id='btn', z=4),
        text('btn', 360, 920, 360, 40, 'Claim offer', 30, 'Mystery Grotesk', font_weight='semibold',
             text_align='center', group_id='btn', z=5),
        shape('ring', 900, 900, 260, 260, fill='rgba(255,255,255,0.1)', kind='ellipse', z=1),
    ]}


def busy(seed=7):
    rnd = random.Random(seed)
    els = [bg(1080, 1080)]
    words = ('golden hour botanical sparkle citrus honey rosemary orange bloom zest light summer '
             'twilight fizz garden harvest').split()
    for k in range(14):
        x, y = rnd.randint(0, 800), rnd.randint(0, 1000)
        n = rnd.randint(2, 14)
        els.append(text(f't{k}', x, y, rnd.randint(160, 500), 60, ' '.join(rnd.choice(words) for _ in range(n)),
                        rnd.choice([14, 18, 22, 28, 36, 48]), rnd.choice(['Solstice Sans', 'Solstice Serif']),
                        font_weight=rnd.choice(['regular', 'bold']), text_align=rnd.choice(['left', 'center'])))
    for k in range(8):
        s = rnd.randint(80, 260)
        els.append(image(f'i{k}', rnd.randint(0, 900), rnd.randint(0, 900), s, s,
                         rnd.choice(['solstice_bottle.png', 'missing_%d.png' % k])))
    for k in range(6):
        x, y = rnd.randint(0, 900), rnd.randint(0, 1000)
        els.append(shape(f'p{k}_bg', x, y, 180, 44, corner_radius=22, group_id=f'p{k}'))
        els.append(text(f'p{k}_t', x, y + 10, 180, 24, rnd.choice(['NEW', 'SALE', 'LIMITED', 'TRY ME', 'ORGANIC']),
                        16, font_weight='bold', text_align='center', group_id=f'p{k}'))
    for k in range(10):
        s = rnd.randint(30, 200)
        els.append(shape(f'dot{k}', rnd.randint(-50, 1000), rnd.randint(-50, 1000), s, s,
                         fill='rgba(255,255,255,0.2)', kind='ellipse', z=1))
    return {'canvas': {'width': 1080, 'height': 1080}, 'elements': els}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    designs = {'landscape': landscape(), 'text_only': text_only(), 'gallery': gallery(),
               'odd_bits': odd_bits(), 'busy': busy()}
    targets = {'landscape': (1080, 1920), 'text_only': (728, 90), 'gallery': (1600, 400),
               'odd_bits': (1280, 720), 'busy': (728, 90)}
    for name, src in designs.items():
        w, h = targets[name]
        (out / f'synthetic_{name}.json').write_text(json.dumps(
            {'task_id': name, 'source': src, 'target_canvas': {'width': w, 'height': h}}, indent=1))
        print(out / f'synthetic_{name}.json')


if __name__ == '__main__':
    main()
