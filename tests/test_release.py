import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseChecks(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.rglob('*.py'):
            if '.git' not in path.parts:
                with self.subTest(path=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(encoding='utf-8-sig'))

    def test_help_without_ml_dependencies(self):
        result = subprocess.run([sys.executable, 'main.py', '--help'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--refinement_iterations', result.stdout)

    def test_training_free_rejects_optimizer_flag(self):
        result = subprocess.run([sys.executable, 'main.py', '--method', 'rtts', '--adapt'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('training-free', result.stderr)

    def test_all_presets_generate_training_free_commands(self):
        launcher = load_script('run')
        args = argparse.Namespace(data_dir='.', output='outputs/test', workers=0,
                                  iterations=2, trials=1, seed=0, corruptions='all', debug=False)
        presets = list((ROOT / 'configs').glob('*.json'))
        self.assertEqual(len(presets), 7)
        for path in presets:
            with self.subTest(preset=path.stem):
                config = json.loads(path.read_text())
                command = launcher.build_command(config, args)
                self.assertEqual(command[command.index('--method') + 1], 'rtts')
                self.assertNotIn('--adapt', command)
                self.assertNotIn('--lr', command)
                self.assertEqual(command[command.index('--refinement_iterations') + 1], '2')
                self.assertTrue(config['required_paths'])

    def test_summary_requires_complete_corruption_set(self):
        summarizer = load_script('summarize_results')
        text = 'original, 99.00 +/- 0.00, 99.00 +/- 0.00, 99.00 +/- 0.00\n'
        self.assertEqual(len(summarizer.summarize(text)), 1)
        for corruption in sorted(summarizer.CORRUPTIONS):
            text += f'{corruption}, 10.00 +/- 1.00, 20.00 +/- 1.00, 30.00 +/- 1.00\n'
        rows = summarizer.summarize(text)
        self.assertEqual(rows[-1], ['corruption_mean_15', 10.0, '', 20.0, '', 30.0, ''])

    def test_summary_rejects_duplicate_and_empty_input(self):
        summarizer = load_script('summarize_results')
        row = 'original, 1.00 +/- 0.00, 1.00 +/- 0.00, 1.00 +/- 0.00\n'
        with self.assertRaises(ValueError):
            summarizer.summarize(row + row)
        with self.assertRaises(ValueError):
            summarizer.summarize('')


if __name__ == '__main__':
    unittest.main()
