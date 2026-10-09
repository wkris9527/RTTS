"""Tensor-level regression against the archived normalization and pooling."""
import ast
from collections import OrderedDict
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, 'PyTorch is required for tensor checks.')
class RefinementChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('refinement', ROOT / 'utils_local/refinement.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        tree = ast.parse((ROOT / 'ovss/clip/model.py').read_text(encoding='utf-8'))
        clip = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CLIP')
        methods = [node for node in clip.body if isinstance(node, ast.FunctionDef) and
                   node.name in ['_distributed_sinkhorn', 'masked_pooling', 'erode_mask_to_center']]
        reference_class = ast.ClassDef(name='Reference', bases=[], keywords=[], body=methods, decorator_list=[])
        namespace = {'torch': torch, 'F': F}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[reference_class], type_ignores=[])), '<reference>', 'exec'), namespace)
        cls.reference = namespace['Reference']()

    def test_sinkhorn_matches_archived_math(self):
        logits = torch.tensor([[2., -3., 7.], [-1., 4., 0.], [1., 1., 1.], [-5., -2., 6.]])
        actual = self.module.sinkhorn_assignments(logits)
        expected = self.reference._distributed_sinkhorn(logits)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(dim=1), torch.ones(4))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual >= 0).all())

    def test_pooling_handles_empty_regions_without_nan(self):
        feature = torch.ones(1, 3, 4, 4)
        masks = torch.zeros(1, 2, 4, 4)
        masks[:, 0] = 1
        actual = self.module.masked_pooling(feature, masks)
        torch.testing.assert_close(actual[0, 0], torch.ones(3))
        torch.testing.assert_close(actual[0, 1], torch.zeros(3))

    def test_no_proposals_preserves_dense_logits(self):
        generator = SimpleNamespace(generate_prompts=lambda *a, **k: [], get_masks=lambda *a, **k: None)
        model = SimpleNamespace(logits2sam=generator)
        logits = torch.randn(1, 1, 3, 4, 4)
        features = torch.randn(1, 17, 8)
        text = torch.randn(1, 3, 8)
        actual = self.module.refine_logits(model, torch.randn(1, 3, 56, 56), logits, features, text)
        torch.testing.assert_close(actual, logits)

    def test_two_rounds_dynamic_grid_and_device(self):
        calls = []
        masks = torch.zeros(1, 2, 56, 56)
        masks[:, 0, :28] = 1
        masks[:, 1, 28:] = 1

        def get_masks(*args, **kwargs):
            calls.append(1)
            return masks

        generator = SimpleNamespace(generate_prompts=lambda *a, **k: [], get_masks=get_masks)
        model = SimpleNamespace(logits2sam=generator, logit_scale=torch.tensor(0.),
                                masked_pooling=self.reference.masked_pooling)
        logits = torch.zeros(1, 1, 3, 4, 4)
        features = F.normalize(torch.randn(1, 17, 8), dim=-1)
        text = F.normalize(torch.randn(1, 3, 8), dim=-1)
        actual = self.module.refine_logits(model, torch.randn(1, 3, 56, 56), logits, features, text, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(actual.shape, logits.shape)
        self.assertEqual(actual.device, logits.device)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float(actual.abs().sum()), 0.)
        self.assertFalse(actual.requires_grad)

    def test_refinement_rejects_multiple_simultaneous_images(self):
        with self.assertRaises(ValueError):
            self.module.refine_logits(None, torch.zeros(2, 3, 56, 56), torch.zeros(1, 2, 3, 4, 4), None, None)

    def test_real_tiny_clip_forward_through_frozen_rtts_wrapper(self):
        # Exercise the actual CLIP classes without importing optional SAM/MMCV.
        # Synthetic proposals isolate the integration contract from checkpoints.
        import numpy as np
        from typing import Tuple, Union
        tree = ast.parse((ROOT / 'ovss/clip/model.py').read_text(encoding='utf-8'))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        namespace = {'torch': torch, 'nn': torch.nn, 'F': F, 'np': np, 'math': math,
                     'OrderedDict': OrderedDict, 'Tuple': Tuple, 'Union': Union}
        exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), '<clip>', 'exec'), namespace)
        model = namespace['CLIP'](embed_dim=8, image_resolution=56, vision_layers=2,
                                  vision_width=64, vision_patch_size=14, context_length=8,
                                  vocab_size=16, transformer_width=32, transformer_heads=1,
                                  transformer_layers=1)
        model.visual.set_params('reduced', 'naclip', 5.0)
        model.logits2sam = SimpleNamespace(generate_prompts=lambda *a, **k: [], get_masks=lambda *a, **k: None)

        def load(*args, **kwargs):
            self.assertTrue(kwargs['enable_rtts'])
            return model, lambda texts: torch.tensor([[1, 2, 15, 0, 0, 0, 0, 0]] * len(texts))

        wrapper_tree = ast.parse((ROOT / 'adapt/rtts.py').read_text(encoding='utf-8'))
        wrapper_class = next(node for node in wrapper_tree.body if isinstance(node, ast.ClassDef))
        import time
        wrapper_namespace = {'torch': torch, 'F': F, 'time': time, 'load_ovss': load,
                             'refine_logits': self.module.refine_logits}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper_class], type_ignores=[])), '<rtts>', 'exec'), wrapper_namespace)
        wrapper = wrapper_namespace['RTTS']('naclip', 'test', ['road', 'person'], prompt_dir=None, device='cpu')
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertFalse(hasattr(wrapper, 'optimizer'))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        result = wrapper.evaluate(torch.randn(2, 3, 56, 56))
        self.assertEqual(result.shape, (2, 2, 56, 56))
        self.assertTrue(torch.isfinite(result).all())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        with self.assertRaises(ValueError):
            wrapper.adapt(torch.randn(1, 3, 56, 56))


if __name__ == '__main__':
    unittest.main()
