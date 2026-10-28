"""Focused checks for the current role-aware engine (no legacy helper imports).

.venv312/bin/python -m unittest discover -s tests -p 'test_missing_asset_layout.py' -v
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from engine.resize import resize

ROOT = Path(__file__).resolve().parents[1]


def bounds(element):
    """Independent corner rotation, not the engine's bounds implementation."""
    cx = element['x'] + element['width']/2
    cy = element['y'] + element['height']/2
    a = math.radians(element.get('rotation', 0))
    corners = [(cx+x*math.cos(a)-y*math.sin(a), cy+x*math.sin(a)+y*math.cos(a))
               for x in (-element['width']/2, element['width']/2)
               for y in (-element['height']/2, element['height']/2)]
    return min(x for x,y in corners), min(y for x,y in corners), max(x for x,y in corners), max(y for x,y in corners)


def distance(first, second):
    l,t,r,b = bounds(first)
    ll,tt,rr,bb = bounds(second)
    return math.hypot(max(0,l-rr,ll-r), max(0,t-bb,tt-b))


class MissingAssetLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = [json.loads(p.read_text()) for p in sorted((ROOT/'input').glob('*.json'))]
        cls.results = [resize(t['source'],t['target_canvas']) for t in cls.tasks]

    def test_sample_bounds_properties_and_foreground_collisions(self):
        for task,output in zip(self.tasks,self.results):
            source=task['source']
            self.assertEqual(len(source['elements']),len(output['elements']))
            cw,ch=output['canvas']['width'],output['canvas']['height']
            foreground=[]
            for src,out in zip(source['elements'],output['elements']):
                for key in src.keys()-{'x','y','width','height','font_size','corner_radius','stroke_width','letter_spacing','fit'}:
                    self.assertEqual(src[key],out[key],key)
                self.assertEqual(src.get('opacity',1),out.get('opacity',1))
                l,t,r,b=bounds(out)
                self.assertGreaterEqual(l,-.02)
                self.assertGreaterEqual(t,-.02)
                self.assertLessEqual(r,cw+.02)
                self.assertLessEqual(b,ch+.02)
                bg=src['type']!='text' and src['x']<=0 and src['y']<=0 and src['width']>=source['canvas']['width'] and src['height']>=source['canvas']['height']
                deco=src['type']=='shape' and src.get('opacity',1)<.45 and not src.get('group_id')
                if not bg and not deco:
                    foreground.append(out)
                if src['type']=='image' and not bg:
                    self.assertAlmostEqual(src['width']/src['height'],out['width']/out['height'],delta=.003)
            for i,e in enumerate(foreground):
                for other in foreground[i+1:]:
                    if e.get('group_id') and e.get('group_id')==other.get('group_id'):
                        continue
                    l,t,r,b=bounds(e);ll,tt,rr,bb=bounds(other)
                    area=max(0,min(r,rr)-max(l,ll))*max(0,min(b,bb)-max(t,tt))
                    self.assertLessEqual(area,.02,(e['id'],other['id']))

    def test_missing_image_is_visible_proportional_and_separated(self):
        found=0
        for task,output in zip(self.tasks,self.results):
            for src,out in zip(task['source']['elements'],output['elements']):
                if src['type']!='image' or (ROOT/'brand_kit/assets'/src['asset_id']).is_file():
                    continue
                found+=1
                self.assertEqual(out['id'],src['id'])
                self.assertEqual(out['asset_id'],src['asset_id'])
                self.assertEqual(out.get('opacity',1),src.get('opacity',1))
                self.assertGreaterEqual(out['width'],20)
                self.assertGreaterEqual(out['height'],12)
                for other in output['elements']:
                    if other is out or other.get('z_index',0)<2:
                        continue
                    self.assertGreaterEqual(distance(out,other),10.7,other['id'])
        self.assertGreater(found,0)

    def test_renamed_missing_asset_and_ids_do_not_change_geometry(self):
        task=next(t for t in self.tasks if any(e['type']=='image' and not (ROOT/'brand_kit/assets'/e['asset_id']).is_file() for e in t['source']['elements']))
        source=deepcopy(task['source'])
        for i,e in enumerate(source['elements']):
            e['id']=f'generic-{i}'
            if e.get('group_id'):e['group_id']='renamed-'+e['group_id']
            if e['type']=='image' and not (ROOT/'brand_kit/assets'/e['asset_id']).is_file():
                e['asset_id']='another-deliberately-absent-asset.png'
                e['opacity']=.63
        original=resize(task['source'],task['target_canvas'])
        renamed=resize(source,task['target_canvas'])
        for src,a,b in zip(source['elements'],original['elements'],renamed['elements']):
            for key in ('x','y','width','height','font_size'):
                self.assertEqual(a.get(key),b.get(key))
            self.assertEqual(src.get('opacity',1),b.get('opacity',1))


if __name__=='__main__':
    unittest.main()
