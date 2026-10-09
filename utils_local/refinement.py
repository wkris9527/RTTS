"""Object-level feedback extracted from the anonymous CLIP implementation.

The proposal generator remains in logits_sam.py. This module handles dense
semantic feedback with the original region voting and Sinkhorn operations.
"""
import torch
import torch.nn.functional as F


def sinkhorn_assignments(logits, n_iters=3):
    """Return region-by-category assignments using the snapshot normalization."""
    if logits.ndim != 2 or min(logits.shape) == 0:
        raise ValueError('Expected a nonempty [regions, categories] matrix.')
    q = logits.float().softmax(dim=-1).T
    mass = q.sum().clamp_min(1e-6)
    q = q / mass
    for _ in range(n_iters):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6) / logits.shape[1]
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-6) / mass
    return (q * mass).T


def masked_pooling(features, masks):
    """Original masked mean pooling: [B,D,H,W] -> [B,M,D]."""
    normalized = masks / masks.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (features[:, None] * normalized[:, :, None]).sum(dim=(-2, -1))


@torch.no_grad()
def refine_logits(model, image, logits, image_features, text_features, iterations=2):
    """Apply the existing CRG/MRC/RCA feedback to a single input patch.

    Args:
        logits: [templates, 1, categories, patch_height, patch_width].
        image_features: normalized CLIP tokens, including the CLS token.
        text_features: normalized [templates, categories, embedding_dim].
    """
    if image.shape[0] != 1 or logits.shape[1] != 1:
        raise ValueError('RTTS refinement expects one patch at a time.')
    height, width = logits.shape[-2:]
    features = image_features[:, 1:].transpose(1, 2).reshape(1, -1, height, width)
    spatial = [features[0].permute(1, 2, 0)]
    scoring_text = text_features.mean(dim=0)
    for _ in range(iterations):
        prompts = model.logits2sam.generate_prompts(logits, original_img_shape=image.shape[-2:])
        masks = model.logits2sam.get_masks(
            image, prompts, logits, text_embeddings=scoring_text, spatial_features=spatial
        )
        if masks is None or masks.shape[1] == 0:
            break
        masks = F.interpolate(masks.float(), size=(height, width), mode='bilinear', align_corners=False, antialias=True)
        masks = masks.to(device=logits.device, dtype=logits.dtype)
        masks = masks[:, masks[0].sum(dim=(-2, -1)) != 0]
        if masks.shape[1] == 0:
            break
        voted = logits.clone()
        for index in range(masks.shape[1] - 1, -1, -1):
            region = masks[0, index] > 0
            if region.any():
                voted[:, 0, :, region] = logits[:, 0, :, region].mean(dim=-1, keepdim=True)
        prototypes = model.masked_pooling(features, masks).float()
        projected = []
        for text in text_features:
            similarities = prototypes @ text.float().T
            assignment = sinkhorn_assignments(similarities[0]).unsqueeze(0)
            projected.append(torch.einsum('bmc,bmhw->bchw', assignment, masks.float()))
        region_logits = model.logit_scale.exp() * torch.stack(projected)
        logits = logits + 0.5 * (voted + region_logits)
    return logits
