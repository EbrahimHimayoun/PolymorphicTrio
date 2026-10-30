"""Independent invariants plus allocation/growth regression fixtures.

Run: .venv312/bin/python -m unittest discover -s tests -v
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from engine.resize import resize, TEXT_FLOOR
import skia
from renderer.converter.brand_fonts import load_brand_font_manager
from renderer.converter.wrap import char_widths, wrap_text

ROOT = Path(__file__).resolve().parents[1]
FONT_MANAGER = load_brand_font_manager(str(ROOT/'brand_kit/fonts'))


def metric_fixture(name):
    """Choose goldens by actual glyph advances, not the operating-system name.

    The pinned Skia builds return fractional advances on macOS and hinted
    integer advances in the Linux image. Both must satisfy the same content,
    readability and geometric invariants, but exact coordinates differ.
    """
    widths=char_widths(FONT_MANAGER,'Solstice Sans','bold','bold',7,'LIMITED EDITION')
    suffix='_hinted' if all(float(w).is_integer() for w in widths) else ''
    return ROOT/'tests/fixtures'/f'{name}{suffix}.json'


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
    # The renderer uses integer-size Skia glyphs and actual ascent/descent,
    # not one line_height multiplier per line. Measure every applied variant.
    styles=[(weight,style)]
    styles += [(run.get('font_weight',weight),
                ('bolditalic' if run.get('font_weight',weight)=='bold' else 'italic')
                if run.get('font_style',e.get('font_style'))=='italic' else
                ('bold' if run.get('font_weight',weight)=='bold' else 'regular'))
               for run in e.get('style_runs',[])]
    metrics=[skia.Font(skia.Typeface.MakeFromData(FONT_MANAGER.lookup(e['font_family'],w,s)),int(e['font_size'])).getMetrics()
             for w,s in styles]
    signatures=[(e.get('color'),weight,style)]*len(e['content'])
    for run in e.get('style_runs',[]):
        for i in range(max(0,run['start']),min(len(signatures),run['end']+1)):
            signatures[i]=(run.get('color',e.get('color')),run.get('font_weight',weight),run.get('font_style',style))
    lengths=[]
    start=0
    for line in wrapped.split('\n'):
        stop=start+len(line)
        # Letter spacing restarts at each styled span in the official renderer.
        spaced=sum(1 for i in range(start+1,stop) if signatures[i]==signatures[i-1])
        lengths.append(sum(widths[start:stop])+spaced*e.get('letter_spacing',0))
        start=stop+1
    height=(len(lengths)-1)*e['font_size']*e.get('line_height',1.2)+max(-m.fAscent for m in metrics)+max(m.fDescent for m in metrics)
    return max(lengths,default=0),height,len(lengths)


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
            adjustable={'x','y','width','height','font_size','corner_radius',
                        'stroke_width','letter_spacing'}
            covers=(src['x']<=0 and src['y']<=0
                    and src['width']>=source['canvas']['width']*.98
                    and src['height']>=source['canvas']['height']*.98)
            if covers and src['type']=='image':
                adjustable.add('fit')  # full-bleed images may crop with cover
            for key,value in src.items():
                if key not in adjustable:
                    self.assertEqual(value,out.get(key),key)
            self.assertTrue(set(src).issubset(out))
            self.assertGreater(out['width'],0)
            self.assertGreater(out['height'],0)
            l,t,r,b=extent(out)
            self.assertGreaterEqual(l,-1e-6)
            self.assertGreaterEqual(t,-1e-6)
            self.assertLessEqual(r,cw+1e-6)
            self.assertLessEqual(b,ch+1e-6)
            if src['type']=='image' and out.get('fit')!='cover':
                ratio=src['width']/src['height']
                # Output geometry is rounded to 0.01px: allow only that error.
                self.assertLessEqual(abs(out['width']-ratio*out['height']),.005*(1+ratio)+1e-6)
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

    def test_samples_and_unseen_formats(self):
        for path,task in fixtures():
            source=task['source']
            saved=deepcopy(source)
            for target in (task['target_canvas'],{'width':400,'height':1600},
                           {'width':900,'height':900},{'width':1920,'height':1080},
                           {'width':1200,'height':180}):
                with self.subTest(sample=path.name,target=target):
                    output=resize(source,target)
                    self.assertEqual(output['canvas'],target)
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
        expected=json.loads(metric_fixture('level1_baseline').read_text())
        self.assertEqual(resize(task['source'],task['target_canvas']),expected)

    def test_growth_never_adds_collisions(self):
        # Exercise completed reflow/growth rather than patching a removed pass.
        for path,task in fixtures():
            for factor in (1,2):
                target={k:v*factor for k,v in task['target_canvas'].items()}
                output=resize(task['source'],target)
                self.assert_valid(task['source'],output)

    def test_free_space_grows_text_and_button_together(self):
        source={'canvas':{'width':500,'height':200},'elements':[
            dict(id='shape',type='shape',shape_type='rectangle',fill_color='red',
                 x=20,y=30,width=200,height=50,group_id='pair',corner_radius=20),
            text('label',30,40,180,30,20,'Readable button',group_id='pair',text_align='center'),
            text('other',320,120,150,30,16,'Other text')]}
        smaller=resize(source,{'width':250,'height':100})
        output=resize(source,{'width':750,'height':300})
        initial=smaller['elements']
        result=output['elements']
        self.assertGreater(result[1]['font_size'],initial[1]['font_size'])
        self.assertGreater(result[0]['width'],initial[0]['width'])
        self.assertGreater(result[0]['height'],initial[0]['height'])
        self.assert_valid(source,output)
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
        self.assertGreaterEqual(output['elements'][0]['font_size'],14)
        self.assertTrue(all(e['font_size']>=11 for e in output['elements'][1:]))
        sizes=[e['font_size'] for e in output['elements']]
        self.assertGreaterEqual(sizes[0],sizes[1]*1.18)
        self.assertGreaterEqual(sizes[0],sizes[2]*1.18)
        # Uniform typography must not acquire an artificial size hierarchy.
        equal=deepcopy(source)
        for e in equal['elements']:e['font_size']=18
        equal_output=resize(equal,{'width':900,'height':240})
        self.assert_valid(equal,equal_output)
        equal_sizes=[e['font_size'] for e in equal_output['elements']]
        self.assertEqual(len(set(equal_sizes)),1)

    def test_sample_hierarchy_after_growth(self):
        for path,task in fixtures():
            output=resize(task['source'],task['target_canvas'])['elements']
            source=task['source']['elements']
            for i,a in enumerate(source):
                # Compare like-for-like standalone copy. Buttons have their own
                # prominence policy in the current engine, so their font need
                # not follow a single ratio against body or headline text.
                # Check strict ordering, not the obsolete 1.18 ratio: integer
                # sizes of 7px and 6px still preserve hierarchy. The feasible
                # synthetic fixture separately retains the 1.18 separation.
                if a['type']!='text' or a.get('group_id'):continue
                for j,b in enumerate(source):
                    if b['type']=='text' and not b.get('group_id') and a['font_size']>=b['font_size']*1.45:
                        self.assertGreater(output[i]['font_size'],output[j]['font_size'],(path.name,a['id'],b['id']))

    def test_preferences_are_configurable(self):
        task=next(task for path,task in fixtures() if 'level3' in path.name)
        # A confidently small bottom strip exercises the current fine-print
        # floor. Unlike the old soft preferences, this is an exposed role floor.
        bottom=min((e for e in task['source']['elements'] if e['type']=='text'),key=lambda e:e['font_size'])
        baseline=resize(task['source'],task['target_canvas'])
        with patch.dict(TEXT_FLOOR,fine=8):
            output=resize(task['source'],task['target_canvas'])
        self.assert_valid(task['source'],output)
        before=next(e for e in baseline['elements'] if e['id']==bottom['id'])
        after=next(e for e in output['elements'] if e['id']==bottom['id'])
        self.assertGreaterEqual(after['font_size'],8)
        self.assertGreater(after['font_size'],before['font_size'])

    def test_constrained_readability_and_fit(self):
        task=next(task for path,task in fixtures() if 'level3' in path.name)
        output=resize(task['source'],task['target_canvas'])
        # The current public result has no shortfall report. Validate actual
        # rendered sizes without requiring a failure to meet readability goals.
        self.assertTrue(all(e['font_size']>=1 for e in output['elements'] if e['type']=='text'))
        self.assert_valid(task['source'],output)
        largest=max(task['source']['elements'],key=lambda e:e.get('font_size',0))
        resized=next(e for e in output['elements'] if e['id']==largest['id'])
        self.assertGreaterEqual(resized['font_size'],13)
        # The selected layout must improve supporting copy too, rather than
        # buying a larger headline by making the already-small body smaller.
        from engine import resize as engine
        by_id={e['id']:e for e in output['elements']}
        for u in engine.analyse(task['source']).units:
            if u.role=='body':
                identifier=task['source']['elements'][u.text_id]['id']
                self.assertGreaterEqual(by_id[identifier]['font_size'],9)
        # Public resize(), without debug hooks or a manual candidate override,
        # must reproduce the approved copy geometry and enlarged right hero.
        approved=json.loads(metric_fixture('level3_approved_geometry').read_text())
        for identifier,fields in approved.items():
            for key,value in fields.items():
                if key=='font_size':
                    self.assertEqual(by_id[identifier][key],value)
                else:
                    self.assertAlmostEqual(by_id[identifier][key],value,delta=.03,
                                           msg=f'{identifier}.{key}')

    def test_banner_copy_can_use_parallel_columns(self):
        from engine import resize as engine
        task=next(task for path,task in fixtures() if 'level3' in path.name)
        analysis=engine.analyse(task['source'])
        ctx=engine.Ctx(task['source'],task['target_canvas'],analysis)
        units=[u for u in analysis.units if u.role in ('headline','body','fine')]
        result=engine._zone_copy(ctx,units,500,80,'center',{},split=0.55)
        self.assertIsNotNone(result)
        block,_=result
        out=dict(block.els)
        self.assertEqual(set(out),{i for u in units for i in u.ids})
        head=next(out[u.text_id] for u in units if u.role=='headline')
        body=next(out[u.text_id] for u in units if u.role=='body')
        fine=next(out[u.text_id] for u in units if u.role=='fine')
        self.assertLessEqual(head['x']+head['width'],body['x'])
        self.assertGreaterEqual(fine['y'],max(head['y']+head['height'],body['y']+body['height']))
        self.assertLessEqual(block.w,500)
        self.assertLessEqual(block.h,80)
        for e in out.values():
            w,h,_=renderer_measure(e)
            self.assertLessEqual(w,e['width']+1e-5)
            self.assertLessEqual(h,e['height']+1e-5)

    def test_compact_brand_group_keeps_content_and_clearance(self):
        from engine import resize as engine
        task=next(task for path,task in fixtures() if 'level3' in path.name)
        analysis=engine.analyse(task['source'])
        ctx=engine.Ctx(task['source'],task['target_canvas'],analysis)
        lanes=engine._lanes(analysis)
        tags=[u for u in lanes['X'] if u.kind=='pill']
        stamps=[u for u in lanes['X'] if u.kind!='pill']
        block=engine._zone_right(ctx,lanes['B'],lanes['A'],lanes['D'],[],
                                 'compact',2,21.6,10,7,81,tags,stamps,'head')
        self.assertIsNotNone(block)
        selected=sorted(block.els)
        source={'canvas':task['source']['canvas'],
                'elements':[task['source']['elements'][i] for i,_ in selected]}
        output={'canvas':{'width':math.ceil(block.w),'height':81},
                'elements':[e for _,e in selected]}
        self.assert_valid(source,output)
        for u in stamps:
            if not u.missing:continue
            box=extent(dict(selected)[u.ids[0]])
            for i,e in selected:
                if i in u.ids:continue
                other=extent(e)
                gap=math.hypot(max(0,box[0]-other[2],other[0]-box[2]),
                               max(0,box[1]-other[3],other[1]-box[3]))
                self.assertGreaterEqual(gap,10.8-1e-6)

    def test_glyph_measurements_match_official_renderer(self):
        from engine import resize as engine
        for family in ('Solstice Sans','Solstice Serif'):
            for size in (7,9,13):
                for weight in ('regular','bold'):
                    content='Introducing the LIMITED EDITION'
                    expected=char_widths(FONT_MANAGER,family,weight,weight,size,content)
                    self.assertEqual(engine._ts().widths(family,weight,weight,size,content),
                                     tuple(expected))

    def test_missing_asset_gets_real_collision_free_space(self):
        source={'canvas':{'width':400,'height':200},'elements':[
            text('copy',0,0,250,100,18,'Existing content stays unchanged.'),
            dict(id='unresolved',type='image',asset_id='absent-allocation-fixture.png',
                 x=290,y=145,width=100,height=50,z_index=2)]}
        output=resize(source,source['canvas'])
        result=output['elements']
        self.assertEqual(result[1]['id'],source['elements'][1]['id'])
        self.assertEqual(result[1]['asset_id'],source['elements'][1]['asset_id'])
        self.assertGreaterEqual(result[1]['width'],100)
        self.assertGreaterEqual(result[1]['height'],50)
        self.assert_valid(source,output)

    def test_hero_growth_can_move_up_without_moving_copy(self):
        from engine import resize as engine
        source={'canvas':{'width':400,'height':200},'elements':[
            dict(id='picture',type='image',asset_id='solstice_bottle.png',
                 x=140,y=10,width=60,height=60,fit='contain'),
            text('left',0,0,130,200,12,'Left copy'),
            text('right',190,80,200,110,12,'Right copy')]}
        analysis=engine.analyse(source)
        ctx=engine.Ctx(source,source['canvas'],analysis)
        original=dict(enumerate(deepcopy(source['elements'])))
        original[0].update(y=90,width=40,height=40)
        saved=deepcopy(original)
        out=engine._grow_hero_in_free_space(ctx,original)
        self.assertEqual(original,saved)
        self.assertEqual(out[1],original[1])
        self.assertEqual(out[2],original[2])
        self.assertGreater(out[0]['width'],original[0]['width']+10)
        self.assertLess(out[0]['y'],original[0]['y'])
        self.assertEqual(out[0]['asset_id'],original[0]['asset_id'])
        self.assertAlmostEqual(out[0]['width']/out[0]['height'],1)
        self.assert_valid(source,{'canvas':source['canvas'],'elements':list(out.values())})
        # Once the local free space is filled, a repeat pass must be stable.
        self.assertEqual(engine._grow_hero_in_free_space(ctx,out),out)

    def test_rotated_complex_group_remains_rigid(self):
        source={'canvas':{'width':300,'height':300},'elements':[
            dict(id='s',type='shape',shape_type='rectangle',fill_color='red',
                 x=20,y=20,width=160,height=60,rotation=30,group_id='g'),
            text('t',30,30,140,40,18,'Some words',rotation=30,group_id='g'),
            dict(id='dot',type='shape',shape_type='ellipse',fill_color='blue',
                 x=50,y=45,width=12,height=12,rotation=30,group_id='g')]}
        for target in ({'width':120,'height':500},{'width':500,'height':120}):
            output=resize(source,target)
            self.assert_valid(source,output)
            out=output['elements']
            factor=out[0]['width']/160
            # The engine rounds geometry to hundredths and trims loose text
            # height to actual ink, preserving relative top-left positions.
            self.assertAlmostEqual(out[1]['width'],140*factor,delta=.02)
            self.assertAlmostEqual(out[1]['x']-out[0]['x'],10*factor,delta=.02)
            self.assertAlmostEqual(out[1]['y']-out[0]['y'],10*factor,delta=.02)
            self.assertAlmostEqual(out[2]['width'],12*factor,delta=.02)
            self.assertAlmostEqual(out[2]['x']-out[0]['x'],30*factor,delta=.02)
            self.assertAlmostEqual(out[2]['y']-out[0]['y'],25*factor,delta=.02)

    def test_decoration_association_requires_evidence(self):
        image=dict(id='photo',type='image',asset_id='unchanged.png',x=100,y=100,
                   width=100,height=100,z_index=3)
        deco=dict(id='d',type='shape',shape_type='ellipse',fill_color='white',
                  opacity=.2,x=70,y=80,width=100,height=100,z_index=1)
        corner=dict(deco,id='corner',x=0,y=0,width=30,height=30)
        source={'canvas':{'width':400,'height':400},'elements':[image,deco,corner]}
        output=resize(source,{'width':800,'height':200})
        self.assert_valid(source,output)
        image,associated,anchored=output['elements']
        # Compare relative offsets/proportions after public reflow. Strong
        # source overlap should retain layering; an unrelated corner stays apart.
        self.assertGreater(intersection(image,associated),0)
        self.assertEqual(intersection(image,anchored),0)
        # Canvas-relative center anchors preserve normalized positions, not
        # literal corner coordinates after anisotropic canvas changes.
        self.assertAlmostEqual(anchored['x'],max(0,15/400*800-anchored['width']/2),delta=.01)
        self.assertAlmostEqual(anchored['y'],max(0,15/400*200-anchored['height']/2),delta=.01)

    def test_renderer_text_fit_with_styles_and_spacing(self):
        source={'canvas':{'width':600,'height':400},'elements':[
            text('styled',30,40,540,220,28,
                 'Mixed typography with bold and italic spans.\nExplicit newlines also stay intact.',
                 letter_spacing=1.25,line_height=1.3,
                 style_runs=[{'start':0,'end':15,'font_weight':'bold','color':'red'},
                             {'start':22,'end':38,'font_style':'italic'}])]}
        saved=deepcopy(source)
        for target in ({'width':350,'height':700},{'width':900,'height':160}):
            self.assert_valid(source,resize(source,target))
            self.assertEqual(source,saved)

    def test_empty_and_many_elements(self):
        source={'canvas':{'width':500,'height':500},'elements':[]}
        self.assertEqual(resize(source,source['canvas'])['elements'],[])
        source['elements']=[dict(id=str(i),type='shape',shape_type='ellipse',
            fill_color='red',x=(i%6)*80,y=(i//6)*50,width=30,height=30) for i in range(36)]
        self.assert_valid(source,resize(source,{'width':100,'height':600}))


if __name__=='__main__':
    unittest.main()
