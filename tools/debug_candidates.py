"""Print the top-scoring candidate layouts for one task/target (dev aid)."""
import json
import sys

from engine import resize as R


def main():
    task = json.load(open(sys.argv[1]))
    w, h = (int(v) for v in sys.argv[2].split('x')) if len(sys.argv) > 2 else (
        task['target_canvas']['width'], task['target_canvas']['height'])
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    R._DEBUG = []
    R.resize(task['source'], {'width': w, 'height': h})
    an = R.analyse(task['source'])
    rows = sorted(R._DEBUG, key=lambda t: -t[0])
    print(f'{len(rows)} valid candidates')
    for sc, label, out in rows[:top]:
        sizes = []
        for u in an.units:
            if u.kind in ('text', 'pill'):
                sizes.append(f"{u.role[:4]}{out[u.text_id]['font_size']}")
            else:
                b = R._union(R._el_box(out[i]) for i in u.ids)
                sizes.append(f"{u.role[:4]}{b[3]:.0f}h")
        print(f'{sc:7.3f}  {label:60s} {" ".join(sizes)}')
    R._DEBUG = None


if __name__ == '__main__':
    main()
