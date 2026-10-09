import argparse
import ast
import importlib.util
import hashlib
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

    def test_anonymous_source_hashes(self):
        manifest = json.loads((ROOT / 'docs/source-manifest.json').read_text())
        for entry in manifest['files']:
            with self.subTest(path=entry['release_path']):
                data = (ROOT / entry['release_path']).read_bytes()
                if entry['normalize_lf']:
                    data = data.replace(b'\r\n', b'\n')
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])

    def test_launcher_help_without_ml_dependencies(self):
        result = subprocess.run([sys.executable, 'scripts/run.py', '--help'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_presets_use_anonymous_settings(self):
        launcher = load_script('run')
        args = argparse.Namespace(data_dir='.', output='outputs/test', workers=4,
                                  trials=1, seed=0, corruptions='all')
        for path in (ROOT / 'configs').glob('*.json'):
            with self.subTest(preset=path.stem):
                command = launcher.build_command(json.loads(path.read_text()), args)
                self.assertEqual(command[command.index('--method') + 1], 'mlmp')
                self.assertIn('--adapt', command)
                self.assertEqual(command[command.index('--steps') + 1], '10')
                self.assertEqual(command[command.index('--lr') + 1], '0.001')
                self.assertNotIn('--refinement_iterations', command)

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
