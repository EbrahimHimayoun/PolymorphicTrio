"""Independent invariants plus allocation/growth regression fixtures.

Run: .venv312/bin/python -m unittest discover -s tests -v
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from engine.resize import (resize, readability_report, PREFERRED_TEXT_SIZES,
                           _grow_text, _text_preferences, _units, _place_decorations,
                           _grow_missing_images)
from renderer.converter.brand_fonts import load_brand_font_manager
from renderer.converter.wrap import char_widths, wrap_text

ROOT = Path(__file__).resolve().parents[1]
FONT_MANAGER = load_brand_font_manager(str(ROOT/'brand_kit/fonts'))


def extent(e):
    """Independent check: rotate all four actual corners about the center."""
    cx,cy = e['x']+e['width']/2,e['y']+e['height']/2
    angle=math.radians(e.get('rotation',0))
    points=[]
    for dx in (-e['width']/2,e['width']/2):
        for dy in (-e['height']/2,e['height']/2):
            points.append((cx+dx*math.cos(angle)-dy*math.sin(angle),
                           cy+dx*math.sin(angle)+dy*math.cos(angle)))
    return (min(x for x,y in points),min(y for x,y in points),
            max(x for x,y in points),max(y for x,y in points))


def intersection(a,b):
    l,t,r,bottom=extent(a); ll,tt,rr,bb=extent(b)
    return max(0,min(r,rr)-max(l,ll))*max(0,min(bottom,bb)-max(t,tt))


def renderer_measure(e):
    """Use official glyph advances/wrapper, independently of engine text fit."""
    weight=e.get('font_weight','regular')
    italic=e.get('font_style')=='italic'
    style=('bolditalic' if weight=='bold' else 'italic') if italic else ('bold' if weight=='bold' else 'regular')
    widths=char_widths(FONT_MANAGER,e['font_family'],weight,style,e['font_size'],e['content'])
    wrapped=wrap_text(e['content'],widths,e['width'])
    # Include style runs and letter spacing when checking actual line extents.
    for run in e.get('style_runs',[]):
        rw=run.get('font_weight',weight)
        ri=run.get('font_style',e.get('font_style'))=='italic'
        rs=('bolditalic' if rw=='bold' else 'italic') if ri else ('bold' if rw=='bold' else 'regular')
        variant=char_widths(FONT_MANAGER,e['font_family'],rw,rs,e['font_size'],e['content'])
        for i in range(max(0,run['start']),min(len(widths),run['end']+1)):
            widths[i]=variant[i]
    lengths=[]
    start=0
    for line in wrapped.split('\n'):
        lengths.append(sum(widths[start:start+len(line)])+max(0,len(line)-1)*e.get('letter_spacing',0))
        start+=len(line)+1
    return max(lengths,default=0),len(lengths)*e['font_size']*e.get('line_height',1.2),len(lengths)


def text(identifier, x, y, width, height, size, content='Text content', **extra):
    return dict(id=identifier,type='text',x=x,y=y,width=width,height=height,
                font_size=size,font_family='Solstice Sans',color='white',content=content,**extra)


def fixtures():
    for path in sorted((ROOT/'input').glob('*.json')):
        yield path,json.loads(path.read_text())


class ResizeTests(unittest.TestCase):
    def assert_valid(self,source,output):
        self.assertEqual(len(source['elements']),len(output['elements']))
        self.assertEqual([e['id'] for e in source['elements']],[e['id'] for e in output['elements']])
        cw,ch=output['canvas']['width'],output['canvas']['height']
        for src,out in zip(source['elements'],output['elements']):
            for key in ('id','type','content','font_family','asset_id','color','fill_color',
                        'style_runs','group_id','z_index','opacity','rotation','font_weight',
                        'font_style','text_align','max_lines','stroke_color'):
                self.assertEqual(src.get(key),out.get(key),key)
            self.assertTrue(set(src).issubset(out))
            self.assertGreater(out['width'],0)
            self.assertGreater(out['height'],0)
            l,t,r,b=extent(out)
            self.assertGreaterEqual(l,-1e-6)
            self.assertGreaterEqual(t,-1e-6)
            self.assertLessEqual(r,cw+1e-6)
            self.assertLessEqual(b,ch+1e-6)
            if src['type']=='image' and out.get('fit')!='cover':
                self.assertAlmostEqual(src['width']/src['height'],out['width']/out['height'])
            if out['type']=='text':
                w,h,lines=renderer_measure(out)
                self.assertLessEqual(w,out['width']+1e-5,out['id'])
                self.assertLessEqual(h,out['height']+1e-5,out['id'])
                if out.get('max_lines'):
                    self.assertLessEqual(lines,out['max_lines'])

        foreground=[]
        sw,sh=source['canvas']['width'],source['canvas']['height']
        for i,e in enumerate(source['elements']):
            background=e['type']!='text' and e['x']<=0 and e['y']<=0 and e['width']>=sw*.98 and e['height']>=sh*.98
            decorative=e['type']=='shape' and not e.get('group_id') and e.get('opacity',1)<.4
            if not background and not decorative:
                foreground.append(i)
        for a,i in enumerate(foreground):
            for j in foreground[a+1:]:
                src,other=source['elements'][i],source['elements'][j]
                if src.get('group_id') and src.get('group_id')==other.get('group_id'):
                    continue  # intentional label/background or complex layering
                nontext,copy=(other,src) if src['type']=='text' else (src,other)
                contained=(copy['type']=='text' and nontext['type']!='text'
                    and nontext['x']<=copy['x'] and nontext['y']<=copy['y']
                    and nontext['x']+nontext['width']>=copy['x']+copy['width']
                    and nontext['y']+nontext['height']>=copy['y']+copy['height'])
                if not contained:
                    self.assertLessEqual(intersection(output['elements'][i],output['elements'][j]),1e-5,(src['id'],other['id']))

    def assert_no_new_collisions(self,before,after,units):
        for n,ids in enumerate(units):
            for other in units[n+1:]:
                for i in ids:
                    for j in other:
                        self.assertLessEqual(intersection(after[i],after[j]),
                                             intersection(before[i],before[j])+1e-5,(i,j))

    def test_samples_and_unseen_formats(self):
        for path,task in fixtures():
            source=task['source']
            saved=deepcopy(source)
            for target in (task['target_canvas'],{'width':400,'height':1600},
                           {'width':900,'height':900},{'width':1920,'height':1080},
                           {'width':1200,'height':180}):
                with self.subTest(sample=path.name,target=target):
                    output=resize(source,target)
                    self.assertEqual(source,saved)
                    self.assert_valid(source,output)
                    renamed=deepcopy(source)
                    groups={}
                    for i,e in enumerate(renamed['elements']):
                        e['id']=f'unrelated_{i}'
                        if e.get('group_id'):
                            e['group_id']=groups.setdefault(e['group_id'],f'g{len(groups)}')
                    other=resize(renamed,target)
                    for a,b in zip(output['elements'],other['elements']):
                        for key in ('x','y','width','height','font_size'):
                            self.assertEqual(a.get(key),b.get(key))

    def test_level1_baseline_is_preserved(self):
        task=next(task for path,task in fixtures() if 'level1' in path.name)
        expected=json.loads((ROOT/'tests/fixtures/level1_baseline.json').read_text())
        self.assertEqual(resize(task['source'],task['target_canvas']),expected)

    def test_growth_never_adds_collisions(self):
        calls=[]
        def inspect(source,output,units,reference,cw,ch,prefs):
            before=deepcopy(output)
            result=_grow_text(source,output,units,reference,cw,ch,prefs)
            self.assert_no_new_collisions(before,result,units)
            indices=[i for ids in units for i in ids]
            foreground={'canvas':source['canvas'],'elements':[source['elements'][i] for i in indices]}
            self.assert_valid(foreground,{'canvas':{'width':cw,'height':ch},'elements':[result[i] for i in indices]})
            calls.append(1)
            return result
        with patch('engine.resize._grow_text',side_effect=inspect):
            for path,task in fixtures():
                resize(task['source'],task['target_canvas'])
        self.assertGreaterEqual(len(calls),2)

    def test_free_space_grows_text_and_button_together(self):
        source={'canvas':{'width':500,'height':200},'elements':[
            dict(id='shape',type='shape',shape_type='rectangle',fill_color='red',
                 x=20,y=30,width=200,height=50,group_id='pair',corner_radius=20),
            text('label',30,40,180,30,20,'Readable button',group_id='pair',text_align='center'),
            text('other',320,120,150,30,16,'Other text')]}
        units,_,_=_units(source)
        initial=deepcopy(source['elements'])
        for e in initial[:2]:
            for k in ('x','y','width','height'):
                e[k]*=.5
            if e['type']=='text':e['font_size']*=.5
        result=_grow_text(source,initial,units,1,500,200,_text_preferences(source,units))
        self.assertGreater(result[1]['font_size'],initial[1]['font_size'])
        self.assertGreater(result[0]['width'],initial[0]['width'])
        self.assertGreater(result[0]['height'],initial[0]['height'])
        self.assert_no_new_collisions(initial,result,units)
        self.assert_valid(source,{'canvas':source['canvas'],'elements':result})
        shape,label=result[:2]
        self.assertGreaterEqual(label['x'],shape['x'])
        self.assertGreaterEqual(label['y'],shape['y'])
        self.assertLessEqual(label['x']+label['width'],shape['x']+shape['width']+1e-6)
        self.assertLessEqual(label['y']+label['height'],shape['y']+shape['height']+1e-6)

    def test_feasible_targets_and_clear_hierarchy(self):
        source={'canvas':{'width':600,'height':400},'elements':[
            text('a',20,20,550,60,36,'A clear title'),
            text('b',20,120,550,70,18,'A short supporting description.'),
            text('c',20,250,550,30,16,'One more line.')]}
        output=resize(source,{'width':900,'height':240})
        self.assert_valid(source,output)
        report=readability_report(source,output)
        self.assertTrue(all(r['shortfall'] < .01 for r in report),report)
        sizes=[e['font_size'] for e in output['elements']]
        self.assertGreaterEqual(sizes[0],sizes[1]*1.18)
        self.assertGreaterEqual(sizes[0],sizes[2]*1.18)
        # Uniform typography is not automatically classified as a headline.
        equal=deepcopy(source)
        for e in equal['elements']:e['font_size']=18
        self.assertTrue(all(r['role']=='normal' for r in readability_report(equal,resize(equal,{'width':900,'height':240}))))

    def test_sample_hierarchy_after_growth(self):
        for path,task in fixtures():
            output=resize(task['source'],task['target_canvas'])['elements']
            source=task['source']['elements']
            for i,a in enumerate(source):
                if a['type']!='text':continue
                for j,b in enumerate(source):
                    if b['type']=='text' and a['font_size']>=b['font_size']*1.45:
                        self.assertGreaterEqual(output[i]['font_size']+1e-4,output[j]['font_size']*1.18,(path.name,a['id'],b['id']))

    def test_preferences_are_configurable(self):
        source={'canvas':{'width':400,'height':200},'elements':[text('a',20,40,360,50,10,'Room to grow')]}
        with patch.dict(PREFERRED_TEXT_SIZES,normal=18):
            output=resize(source,{'width':600,'height':200})
            self.assertGreaterEqual(output['elements'][0]['font_size'],17.99)

    def test_constrained_shortfalls_are_honest(self):
        task=next(task for path,task in fixtures() if 'level3' in path.name)
        output=resize(task['source'],task['target_canvas'])
        report=readability_report(task['source'],output)
        self.assertTrue(all(abs(r['shortfall']-max(0,r['preferred']-r['font_size']))<1e-8 for r in report))
        self.assert_valid(task['source'],output)
        largest=max(task['source']['elements'],key=lambda e:e.get('font_size',0))
        resized=next(e for e in output['elements'] if e['id']==largest['id'])
        self.assertGreater(resized['font_size'],10)

    def test_missing_asset_gets_real_collision_free_space(self):
        source={'canvas':{'width':400,'height':200},'elements':[
            text('copy',0,0,250,100,18,'Existing content stays unchanged.'),
            dict(id='unresolved',type='image',asset_id='absent-allocation-fixture.png',
                 x=290,y=145,width=100,height=50,z_index=2)]}
        initial=deepcopy(source['elements'])
        initial[1].update(x=300,y=150,width=10,height=5)
        result=_grow_missing_images(source,initial,[[0],[1]],1,400,200)
        self.assertEqual(result[0],initial[0])
        self.assertEqual(result[1]['id'],source['elements'][1]['id'])
        self.assertEqual(result[1]['asset_id'],source['elements'][1]['asset_id'])
        self.assertGreaterEqual(result[1]['width'],100)
        self.assertGreaterEqual(result[1]['height'],50)
        self.assert_no_new_collisions(initial,result,[[0],[1]])
        self.assert_valid(source,{'canvas':source['canvas'],'elements':result})

    def test_rotated_complex_group_remains_rigid(self):
        source={'canvas':{'width':300,'height':300},'elements':[
            dict(id='s',type='shape',shape_type='rectangle',fill_color='red',
                 x=20,y=20,width=160,height=60,rotation=30,group_id='g'),
            text('t',30,30,140,40,18,'Some words',rotation=30,group_id='g')]}
        for target in ({'width':120,'height':500},{'width':500,'height':120}):
            output=resize(source,target)
            self.assert_valid(source,output)
            out=output['elements']
            factor=out[0]['width']/160
            self.assertAlmostEqual(out[1]['width']/140,factor)
            self.assertAlmostEqual(out[1]['x']-out[0]['x'],10*factor)
            self.assertAlmostEqual(out[1]['y']-out[0]['y'],10*factor)

    def test_decoration_association_requires_evidence(self):
        image=dict(id='photo',type='image',asset_id='unchanged.png',x=100,y=100,
                   width=100,height=100,z_index=3)
        deco=dict(id='d',type='shape',shape_type='ellipse',fill_color='white',
                  opacity=.2,x=70,y=80,width=100,height=100,z_index=1)
        corner=dict(deco,id='corner',x=0,y=0,width=30,height=30)
        source={'canvas':{'width':400,'height':400},'elements':[image,deco,corner]}
        output=deepcopy(source['elements'])
        output[0].update(x=500,y=50)
        _place_decorations(source,output,[[0]],[1,2],800,200)
        # Strong overlap follows the relocated image at its relative offset.
        self.assertAlmostEqual(output[1]['x'],470)
        self.assertAlmostEqual(output[1]['y'],30)
        # Mere nearest-neighbor proximity does not attach a corner decoration.
        self.assertAlmostEqual(output[2]['x'],22.5)
        self.assertAlmostEqual(output[2]['y'],0)

    def test_empty_and_many_elements(self):
        source={'canvas':{'width':500,'height':500},'elements':[]}
        self.assertEqual(resize(source,source['canvas'])['elements'],[])
        source['elements']=[dict(id=str(i),type='shape',shape_type='ellipse',
            fill_color='red',x=(i%6)*80,y=(i//6)*50,width=30,height=30) for i in range(36)]
        self.assert_valid(source,resize(source,{'width':100,'height':600}))


if __name__=='__main__':
    unittest.main()
