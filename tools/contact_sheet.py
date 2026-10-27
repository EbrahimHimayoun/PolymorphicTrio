"""Tile rendered PNGs into one labelled contact sheet per sample (dev aid)."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def sheet(files, out, cell=360, cols=4):
    rows = (len(files) + cols - 1) // cols
    W, H = cols * (cell + 10) + 10, rows * (cell + 30) + 10
    img = Image.new('RGB', (W, H), (245, 242, 236))
    d = ImageDraw.Draw(img)
    for k, f in enumerate(files):
        im = Image.open(f).convert('RGB')
        s = min(cell / im.width, cell / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        x = 10 + (k % cols) * (cell + 10)
        y = 10 + (k // cols) * (cell + 30)
        img.paste(im, (x, y + 20))
        d.text((x, y + 4), f.stem.split('__')[-1], fill=(40, 40, 40))
    img.save(out)


if __name__ == '__main__':
    src = Path(sys.argv[1])
    for prefix in sorted({p.name.split('__')[0] for p in src.glob('*__*.png')}):
        files = sorted(src.glob(f'{prefix}__*.png'))
        sheet(files, src / f'SHEET_{prefix}.png')
        print(src / f'SHEET_{prefix}.png')
