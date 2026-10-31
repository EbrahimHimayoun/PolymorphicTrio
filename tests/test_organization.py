"""Content-independent organization preferences and flexible reflow."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from engine.resize import _arrangement_cost, _organization_penalty, _no_worse_readability, analyse, resize
from tools.synthetic import text_only
from tools.validate import check


class OrganizationTests(unittest.TestCase):
    def test_organization_cannot_buy_alignment_with_smaller_body_text(self):
        unit = SimpleNamespace(kind='text', role='body', font=20, text_id=0)
        ctx = SimpleNamespace(an=SimpleNamespace(units=[unit]), soft={'body': 9}, tcap=lambda role, font: 20)
        self.assertFalse(_no_worse_readability(ctx, {0: {'font_size': 9}}, {0: {'font_size': 8}}))
        self.assertTrue(_no_worse_readability(ctx, {0: {'font_size': 10}}, {0: {'font_size': 9}}))

    def test_brand_alignment_cannot_hide_behind_wide_badges(self):
        aligned = ((35, 0, 80, 25), (10, 35, 130, 20), (0, 65, 150, 30))
        displaced = ((0, 0, 80, 25), (50, 35, 130, 20), (40, 65, 150, 30))
        self.assertEqual(_arrangement_cost(aligned, False), 0)
        self.assertGreater(_arrangement_cost(displaced, False), 0.5)

    def test_rows_columns_and_wrapped_rows_are_equally_organized(self):
        layouts = (
            ((0, 0, 20, 10), (30, 0, 40, 10), (80, 0, 10, 10)),
            ((0, 0, 20, 10), (-10, 20, 40, 10), (5, 40, 10, 10)),
            ((0, 0, 20, 10), (30, 0, 20, 10), (15, 20, 20, 10)))
        for boxes in layouts:
            self.assertAlmostEqual(_arrangement_cost(boxes), 0)
            shifted = tuple((100 + x * 3, 70 + y * 3, w * 3, h * 3) for x, y, w, h in boxes)
            self.assertAlmostEqual(_arrangement_cost(shifted), 0)

    def test_scatter_and_uneven_badge_gaps_cost_more(self):
        row = ((0, 0, 20, 10), (30, 0, 20, 10), (60, 0, 20, 10))
        scatter = ((0, 0, 20, 10), (30, 25, 20, 10), (70, 50, 20, 10))
        uneven = ((0, 0, 20, 10), (30, 0, 20, 10), (110, 0, 20, 10))
        self.assertGreater(_arrangement_cost(scatter), _arrangement_cost(row))
        self.assertGreater(_arrangement_cost(uneven), _arrangement_cost(row))

    def test_soft_penalty_is_bounded_and_source_evidence_matters(self):
        boxes = ((0, 0, 20, 10), (30, 25, 20, 10), (70, 50, 20, 10))
        out = {i: dict(x=x, y=y, width=w, height=h) for i, (x, y, w, h) in enumerate(boxes)}
        ctx = SimpleNamespace(organization_groups=[(((0,), (1,), (2,)), 0, True)])
        cost = _organization_penalty(ctx, out)
        self.assertGreater(cost, 0)
        self.assertLessEqual(cost, 2)
        ctx.organization_groups = []
        self.assertEqual(_organization_penalty(ctx, out), 0)

    def test_different_copy_and_canvases_preserve_content_and_fit(self):
        original = text_only()
        for long in (False, True):
            source = deepcopy(original)
            if long:
                for e in source['elements']:
                    if e.get('id') == 'headline':
                        e['content'] = 'A different collection for long evenings and unforgettable summer celebrations'
            saved = deepcopy(source)
            for w, h in ((900, 600), (400, 1200)):
                with self.subTest(long=long, canvas=(w, h)):
                    out = resize(source, {'width': w, 'height': h})
                    self.assertEqual(check(source, out, (w, h)), [])
                    self.assertEqual(source, saved)

    def test_unseen_badge_text_can_reflow_without_fixed_coordinates(self):
        path = Path(__file__).resolve().parents[1] / 'input/chameleon_sample_level3_advanced.json'
        source = json.loads(path.read_text())['source']
        analysis = analyse(source)
        badges = [u for u in analysis.units if u.role == 'badge']
        for unit, content in zip(badges, ('NEW', 'CRAFTED IN SMALL SEASONAL BATCHES', 'AVAILABLE ONLINE')):
            source['elements'][unit.text_id]['content'] = content
        for i, e in enumerate(source['elements']):
            e['id'] = f'unseen_{i}'
        saved = deepcopy(source)
        for w, h in ((900, 900), (1200, 300)):
            result = resize(source, {'width': w, 'height': h})
            self.assertEqual(check(source, result, (w, h)), [])
            self.assertEqual(source, saved)


if __name__ == '__main__':
    unittest.main()
