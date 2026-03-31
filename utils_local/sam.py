import torch


def _patch_batched_nms_for_sam():
    import torchvision.ops.boxes as tv_boxes

    if getattr(tv_boxes, "_mlmp_batched_nms_patched", False):
        safe = tv_boxes.batched_nms
    else:
        _orig = tv_boxes.batched_nms

        def _batched_nms_safe(boxes, scores, idxs, iou_threshold):
            device = boxes.device
            boxes = boxes.detach().float().to(device)
            if not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores, device=device, dtype=torch.float32)
            else:
                scores = scores.detach().float().to(device)
            if not isinstance(idxs, torch.Tensor):
                idxs = torch.as_tensor(idxs, device=device, dtype=torch.long)
            else:
                idxs = idxs.detach().to(device).long()
            keep = _orig(
                boxes.cpu(),
                scores.cpu(),
                idxs.cpu(),
                iou_threshold,
            )
            return keep.to(device=device)

        tv_boxes.batched_nms = _batched_nms_safe
        tv_boxes._mlmp_batched_nms_patched = True
        safe = _batched_nms_safe

    try:
        import segment_anything.automatic_mask_generator as amg

        amg.batched_nms = safe
    except ImportError:
        pass


_patch_batched_nms_for_sam()

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

_patch_batched_nms_for_sam()

import numpy as np
import os
import cv2
import timm
from PIL import Image
from timm.models.vision_transformer import PatchEmbed
from torchvision import transforms
import torch.nn.functional as F
class SamMask:
    def __init__(self):
        self.sam = sam_model_registry['vit_h'](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(device='cuda:0')
        self.predictor = SamPredictor(self.sam)
        self.min_mask_area=100
        self.maskgenerator  = SamAutomaticMaskGenerator(
        model=self.sam,
        points_per_side=16,  
        pred_iou_thresh=0.86,  
        stability_score_thresh=0.92,  
        crop_n_layers=0,  
        crop_n_points_downscale_factor=1,
        min_mask_region_area=self.min_mask_area,  
    )
    def get_masks(self,image):
        image = image.cpu()
        image =np.array(image)
        image = self.restruct_x(torch.tensor(image)).numpy().astype(np.uint8).transpose(0,2,3,1)
        mask_list = []
        for i in range(image.shape[0]):
            single_image = image[i]
            self.predictor.set_image(single_image)
            masks = self.maskgenerator.generate(image=single_image)
            masks = sorted(masks, key=lambda m: m.get('predicted_iou', 0.0), reverse=True)
        
            mask_list.append(masks)
        return mask_list
    def restruct_x(self,x):
        CLIP_MEAN = [122.7709, 116.7460, 104.0937]
        CLIP_STD  = [68.5005, 66.6322, 70.3232]
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1).to(x.device)
        re_x = x * std + mean
        re_x = re_x.clamp(0, 255)
        return re_x


    