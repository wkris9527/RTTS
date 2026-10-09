import ovss.clip as clip
from ovss.clip import tokenize as clip_tokenize


def load_ovss(ovss_type, ovss_backbone, device='cpu', enable_rtts=False):
    """
    Load the OVSS model based on the specified type and backbone.

    Args:
        ovss_type: Type of the OVSS model.
        ovss_backbone: Backbone architecture of the OVSS model.
        device: Device to load the model on (e.g., 'cpu' or 'cuda').

    Returns:
        ovss_model: Loaded OVSS model.
    """
    if ovss_type == 'clip':
        arch = "vanilla"
        attn_strategy = "vanilla"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'sclip':
        arch = "vanilla"
        attn_strategy = "csa"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'naclip':
        arch = "reduced"
        attn_strategy = "naclip"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    else:
        raise ValueError(f"Unsupported OVSS type: {ovss_type}")

    if enable_rtts:
        configure_refinement(ovss_model)
    return ovss_model, tokenize


def configure_refinement(model):
    """Enable the archived feedback hook for explicitly requested combinations."""
    from utils_local.logits_sam import SAMwithlogits
    model.logits2sam = SAMwithlogits(
        num_sample_points=10, response_thresh_ratio=0.5,
        min_prob_thresh=0.1, min_mask_area=100, prompt_type='point',
        enable_enhanced=True, use_semantic=True, alpha=0.3, beta=0.7,
        Sinkhorn_clustered=True,
    )
    model.logits2prompt = True
