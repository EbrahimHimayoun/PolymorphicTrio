"""Post-repair validation and bounded collision recovery regressions."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.resize import (LayoutValidationError, Unit, _final_problems,
                           _review_and_repair, _select_reviewed, resize)


def image(identifier, x, y, size=20):
    return dict(id=identifier, type='image', asset_id='solstice_bottle.png',
                x=x, y=y, width=size, height=size)


def context(elements, groups=None, width=100, height=100):
    units = [Unit(n, ids) for n, ids in enumerate(groups or [[i] for i in range(len(elements))])]
    return SimpleNamespace(els=elements, cw=width, ch=height, floor_mult=1,
                           an=SimpleNamespace(units=units, backgrounds=[]))


class ReviewTests(unittest.TestCase):
    def test_repair_moves_group_together_without_mutating_input(self):
        elements = [image('a', 10, 10), image('b', 12, 12, 5), image('c', 60, 10)]
        ctx = context(elements, [[0, 1], [2]])
        candidate = {i: dict(e) for i, e in enumerate(elements)}
        candidate[2]['x'] = 25
        before = deepcopy(candidate)
        result, issues = _review_and_repair(ctx, candidate)
        self.assertFalse(issues)
        self.assertEqual(candidate, before)
        self.assertEqual(result[1]['x'] - result[0]['x'], 2)
        self.assertEqual(result[1]['y'] - result[0]['y'], 2)
        for i in result:
            self.assertEqual(result[i]['width'], elements[i]['width'])
            self.assertEqual(result[i]['height'], elements[i]['height'])
        self.assertEqual(_review_and_repair(ctx, candidate), (result, issues))

    def test_text_height_repair_cannot_introduce_unchecked_collision(self):
        label = dict(id='label', type='text', content='Hello', font_family='Solstice Sans',
                     color='white', font_size=20, x=10, y=10, width=70, height=1)
        elements = [label, image('product', 10, 20)]
        ctx = context(elements)
        candidate = dict(enumerate(elements))
        _, initial = _review_and_repair(ctx, candidate, rounds=0)
        self.assertTrue(any(p[0] == 'overlap' for p in initial))
        result, issues = _review_and_repair(ctx, candidate)
        self.assertFalse(issues)
        self.assertGreater(result[0]['height'], label['height'])
        self.assertEqual(result[0]['font_size'], 20)

    def test_gate_rejects_content_loss_extra_elements_and_nonfinite_geometry(self):
        elements = [image('a', 10, 10)]
        ctx = context(elements)
        for candidate, diagnostic in (
                ({0: dict(elements[0], asset_id='replacement.png')}, 'preservation'),
                ({0: elements[0], 1: elements[0]}, 'element_count'),
                ({}, 'missing'),
                ({0: dict(elements[0], x=float('nan'))}, 'geometry')):
            with self.subTest(diagnostic=diagnostic):
                self.assertIn(diagnostic, [p[0] for p in _final_problems(ctx, candidate)])

    def test_rejected_high_score_does_not_replace_valid_runner_up(self):
        elements = [image('a', 10, 10)]
        ctx = context(elements)
        corrupted = {0: dict(elements[0], asset_id='replacement.png')}
        valid = {0: dict(elements[0])}
        with patch('engine.resize._score', side_effect=lambda ctx, out, kind: 100 if kind == 'bad' else 0):
            result = _select_reviewed(ctx, [('bad', corrupted, ''), ('good', valid, '')])
        self.assertEqual(result, valid)

    def test_final_gate_enforces_line_limit_even_when_text_box_fits(self):
        label = dict(id='label', type='text', content='Hello\nworld', font_family='Solstice Sans',
                     color='white', font_size=12, x=10, y=10, width=70, height=70, max_lines=1)
        self.assertIn(('max_lines', 0), _final_problems(context([label]), {0: label}))

    def test_exhausted_search_reports_failure_instead_of_returning_overlap(self):
        source = dict(canvas=dict(width=100, height=100),
                      elements=[image('a', 10, 10), image('b', 60, 60)])
        before = deepcopy(source)
        impossible = {i: dict(e, x=0, y=0, width=10, height=10)
                      for i, e in enumerate(source['elements'])}
        with patch('engine.resize._candidates', return_value=[]), \
             patch('engine.resize._fallback', return_value=impossible), \
             patch('engine.resize._grid_fallback', return_value=impossible):
            with self.assertRaises(LayoutValidationError) as caught:
                resize(source, dict(width=10, height=10))
        self.assertIn('overlap', [p[0] for p in caught.exception.problems])
        self.assertEqual(source, before)

    def test_cli_reports_failure_and_continues_other_tasks(self):
        from engine import main
        source = dict(canvas=dict(width=100, height=100), elements=[image('a', 10, 10)])
        task = dict(source=source, target_canvas=source['canvas'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, outputs = root / 'input', root / 'output'
            inputs.mkdir()
            for name in ('a_bad', 'b_good'):
                (inputs / f'{name}.json').write_text(json.dumps(task))
            with patch.object(main, 'INPUT_DIR', inputs), patch.object(main, 'OUTPUT_DIR', outputs), \
                 patch.object(main, 'resize', side_effect=[LayoutValidationError([('overlap', 0, 1)]), source]), \
                 patch.object(main, 'render_to_png', return_value=[]) as render, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main.main(), 1)
            self.assertFalse((outputs / 'a_bad_output.json').exists())
            self.assertTrue((outputs / 'b_good_output.json').exists())
            render.assert_called_once()


if __name__ == '__main__':
    unittest.main()
