"""Training-free RTTS evaluation without an optimizer or parameter updates."""
import time
import torch
import torch.nn.functional as F

from ovss import load_ovss
from utils_local.misc import load_prompts_from_yaml
from utils_local.refinement import refine_logits


class RTTS:
    def __init__(self, ovss_type, ovss_backbone, classes, prompt_dir='prompts.yaml',
                 refinement_iterations=2, runtime_calculation=False, device='cuda'):
        self.device = device
        self.runtime = runtime_calculation
        self.iterations = refinement_iterations
        self.model, tokenize = load_ovss(ovss_type, ovss_backbone, device=device, enable_rtts=True)
        self.model.eval().requires_grad_(False)
        # Refinement is applied explicitly below; disable the legacy forward hook.
        self.model.logits2prompt = False
        prompts = load_prompts_from_yaml(prompt_dir) if prompt_dir else ['a photo of a {}']
        with torch.no_grad():
            embeddings = []
            for name in classes:
                feature = self.model.encode_text(tokenize([p.format(name) for p in prompts]).to(device))
                feature = F.normalize(feature, dim=-1)
                embeddings.append(F.normalize(feature.mean(dim=0), dim=-1))
            self.text_x = torch.stack(embeddings).unsqueeze(0)
        self.adapt_times = []
        self.eval_times = []

    def reset(self):
        """Weights remain frozen; no state restoration is needed."""

    def adapt(self, image):
        raise ValueError('RTTS is training-free. Omit --adapt; use --refinement_iterations.')

    @torch.no_grad()
    def evaluate(self, images):
        start = time.perf_counter()
        predictions = []
        # Cityscapes produces many patches. Process each independently because
        # SAM's predictor and the archived region filtering expect one image.
        for image in images.split(1):
            logits, features, text = self.model(
                image, self.text_x, text_ensemble=True, interpolate=False, vision_out_type='mean'
            )
            logits = refine_logits(self.model, image, logits, features, text, self.iterations)
            prediction = F.interpolate(logits[0].float(), size=image.shape[-2:], mode='bilinear', align_corners=False)
            predictions.append(prediction)
        if self.runtime:
            self.eval_times.append(time.perf_counter() - start)
        return torch.cat(predictions)
