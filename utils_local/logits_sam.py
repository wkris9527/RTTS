from segment_anything import SamPredictor,SamAutomaticMaskGenerator,sam_model_registry
import torch
import math
import numpy as np
import os
import cv2
import timm
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.ops import batched_nms
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

class SAMwithlogits:
    
    def __init__(self, response_thresh_ratio=0.5, num_sample_points=10, 
                 min_prob_thresh=0.1, min_mask_area=100, prompt_type='point',
                 response_thresh_mode='ratio', entropy_max_frac=0.85,
                 enable_enhanced=False, sample_radius=4, num_neighbor_samples=8,
                 num_high_quality=5, num_mid_quality=15, num_low_quality=3, mid_cluster_num=4,
                 alpha=0.0, beta=1.0, exp=2.0, use_semantic=False,
                 enable_nms=True, nms_iou_threshold=0.5,Sinkhorn_clustered = False,
                 soft_merging = False,):
        self.response_thresh_ratio = response_thresh_ratio
        _modes = ('ratio', 'ratio_entropy', 'otsu', 'otsu_entropy')
        if response_thresh_mode not in _modes:
            raise ValueError(f"response_thresh_mode must be one of {_modes}")
        self.response_thresh_mode = response_thresh_mode
        self.entropy_max_frac = entropy_max_frac
        self.num_sample_points = num_sample_points
        self.min_prob_thresh = min_prob_thresh
        self.min_mask_area = min_mask_area
        self.device = 'cuda'
        
        self.enable_enhanced = enable_enhanced
        self.sample_radius = sample_radius
        self.num_neighbor_samples = num_neighbor_samples
        self.num_high_quality = num_high_quality
        self.num_mid_quality = num_mid_quality
        self.num_low_quality = num_low_quality
        self.mid_cluster_num = mid_cluster_num
        
        self.alpha = alpha
        self.beta = beta
        self.exp = exp
        self.use_semantic = use_semantic

        self.Sinkhorn_clustered = Sinkhorn_clustered
        self.enable_nms = enable_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.soft_merging = soft_merging 
        
        self.sam = sam_model_registry['vit_h'](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(device='cuda:0')
        self.predictor = SamPredictor(self.sam)
        # self.auto_mask_generator = SamAutomaticMaskGenerator(
        #     model=self.sam,
        #     points_per_side=16,           
        #     pred_iou_thresh=0.86,         
        #     stability_score_thresh=0.92,  
        #     crop_n_layers=0,
        #     crop_n_points_downscale_factor=1,
        #     min_mask_region_area=100,     
        # )
    @staticmethod
    def _pixel_entropy_from_probs(probs):
        eps = 1e-8
        p = probs.clamp(min=eps)
        return -(p * p.log()).sum(dim=1)

    @staticmethod
    def _otsu_threshold_prob_map(prob_map):
        x = prob_map.detach().float().cpu().numpy()
        x = np.clip(x, 0.0, 1.0)
        if x.size == 0:
            return 0.5
        u8 = (x * 255.0).astype(np.uint8)
        t8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(t8) / 255.0

    def _high_response_mask(self, prob_map, pixel_entropy_map, num_classes):
        max_prob = prob_map.max()
        mode = self.response_thresh_mode

        ratio_mask = prob_map > (self.response_thresh_ratio * max_prob)

        if mode in ('otsu', 'otsu_entropy'):
            t = self._otsu_threshold_prob_map(prob_map)
            base = prob_map > t
            if base.sum() == 0:
                base = ratio_mask
        else:
            base = ratio_mask

        if mode in ('ratio_entropy', 'otsu_entropy'):
            max_h = math.log(float(max(num_classes, 2)))
            ent_cap = self.entropy_max_frac * max_h
            return base & (pixel_entropy_map <= ent_cap)
        return base
    

    
    
    def generate_prompts(self, logits, original_img_shape=(224, 224)):
        logits_agg = logits.mean(dim=0)  # (batch_size, #classes, W, H)
        
        logits_agg = F.interpolate(
            logits_agg, 
            size=original_img_shape, 
            mode='bilinear', 
            align_corners=False
        )  # (batch_size, #classes, H', W')
        
        probs = logits_agg.softmax(dim=1)
        need_entropy = self.response_thresh_mode in ('ratio_entropy', 'otsu_entropy')
        H_map = self._pixel_entropy_from_probs(probs) if need_entropy else None
        
        batch_prompts = []
        for b_idx in range(probs.shape[0]):
            all_points = []
            all_class_ids = []
            
            for cls_idx in range(probs.shape[1]):
                prob_map = probs[b_idx, cls_idx]
                max_prob = prob_map.max()
                if max_prob < self.min_prob_thresh:
                    continue
                
                H_b = H_map[b_idx] if need_entropy else None
                high_response_mask = self._high_response_mask(
                    prob_map, H_b, probs.shape[1]
                )
                
                if high_response_mask.sum() < self.min_mask_area:
                    continue
                
                ys, xs = torch.where(high_response_mask)
                vals = prob_map[ys, xs]
                k = min(len(vals), self.num_sample_points)
                _, topk_idx = torch.topk(vals, k)
                
                points = torch.stack([xs[topk_idx].float(), ys[topk_idx].float()], dim=1)
                all_points.append(points)
                all_class_ids.extend([cls_idx] * k)
            
            if all_points:
                batch_prompts.append({
                    'points': torch.cat(all_points, dim=0),
                    'class_ids': all_class_ids
                })
            else:
                batch_prompts.append({'points': None, 'class_ids': []})
        
        return batch_prompts
    
    def generate_box_prompts(self, logits, original_img_shape=(224, 224)):

        logits_agg = logits.mean(dim=0)  # (batch_size, #classes, W, H)
        
     
        logits_agg = F.interpolate(
            logits_agg, 
            size=original_img_shape, 
            mode='bilinear', 
            align_corners=False
        )  # (batch_size, #classes, H', W')
        
        
        probs = logits_agg.softmax(dim=1)
        
        batch_prompts = []
        for b_idx in range(probs.shape[0]):
            all_boxes = []
            all_class_ids = []
            
            for cls_idx in range(probs.shape[1]):
                prob_map = probs[b_idx, cls_idx]
                max_prob = prob_map.max()
                
                if max_prob < self.min_prob_thresh:
                    continue
                
              
                thresh = self.response_thresh_ratio * max_prob
                high_response_mask = (prob_map > thresh)
                
                if high_response_mask.sum() < self.min_mask_area:
                    continue
                
    
                mask_np = high_response_mask.cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_mask_area:
                        continue
                    
          
                    x, y, w, h = cv2.boundingRect(contour)
                    box = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
                    all_boxes.append(box)
                    all_class_ids.append(cls_idx)
            
            if all_boxes:
                batch_prompts.append({
                    'boxes': torch.stack(all_boxes, dim=0),
                    'class_ids': all_class_ids
                })
            else:
                batch_prompts.append({'boxes': None, 'class_ids': []})
        
        return batch_prompts
    
    def get_masks(self, image, prompts, logits=None, text_embeddings=None, spatial_features=None, masks=None):
        if self.enable_enhanced and logits is not None:
            return self.get_masks_enhanced(image, prompts, logits, text_embeddings, spatial_features, masks)
        
        image = image.cpu()
        image = np.array(image)
        image = self.restruct_x(torch.tensor(image)).numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        
        mask_list = []
        for i in range(image.shape[0]):
            single_image = image[i]
 
            if 'points' in prompts[i]:
    
                point_coords = prompts[i]['points']
                class_ids = prompts[i]['class_ids']
                

                if point_coords is None or len(point_coords) == 0:
                    h, w = single_image.shape[:2]
                    empty_mask = np.zeros((1, h, w), dtype=image.dtype)
                    mask_list.append(torch.from_numpy(empty_mask))
                    print("No valid points found for image.")
                    continue
                

                if isinstance(point_coords, np.ndarray):
                    point_coords = torch.from_numpy(point_coords)
                point_coords = point_coords.float()
                
   
                self.predictor.set_image(single_image)
                img_h, img_w = single_image.shape[:2]
                points_transformed = self.predictor.transform.apply_coords_torch(
                    point_coords.unsqueeze(1), (img_h, img_w)
                )  # (N, 1, 2)
                
        
                class_to_points = {}
                for idx, cls_id in enumerate(class_ids):
                    if cls_id not in class_to_points:
                        class_to_points[cls_id] = []
                    class_to_points[cls_id].append(idx)
                

                all_masks = []
                

                for cls_id, point_indices in sorted(class_to_points.items()):
                    cls_points = points_transformed[point_indices].to(device='cuda:0')  # (N_cls, 1, 2)
                    cls_labels = torch.ones(len(point_indices), 1, dtype=torch.int, device=cls_points.device)
             
                    masks, scores, _ = self.predictor.predict_torch(
                        point_coords=cls_points,  # (N_cls, 1, 2)
                        point_labels=cls_labels,  # (N_cls, 1)
                        boxes=None,
                        multimask_output=True
                    )

                    for pt_idx in range(masks.shape[0]):
                        pt_masks = masks[pt_idx]  # (3, H, W)
                        pt_scores = scores[pt_idx]  # (3,)
                        best_idx = torch.argmax(pt_scores)
                        best_mask = pt_masks[best_idx].cpu().numpy()  # (H, W)
                        mask_area = best_mask.sum()
                        if mask_area>0:
                            all_masks.append(best_mask)

                

                if all_masks:
                    all_masks = np.stack(all_masks, axis=0)  # (num_points, H, W)
                    mask_list.append(torch.from_numpy(all_masks.astype(dtype=image.dtype)))
                else:
                    return None
                
            elif 'boxes' in prompts[i]:

                boxes = prompts[i]['boxes']
                class_ids = prompts[i]['class_ids']

                if boxes is None or len(boxes) == 0:
                    h, w = single_image.shape[:2]
                    empty_mask = np.zeros((1, h, w), dtype=image.dtype)
                    mask_list.append(torch.from_numpy(empty_mask))
                    print("No valid boxes found for image.")
                    continue
                

                if isinstance(boxes, np.ndarray):
                    boxes = torch.from_numpy(boxes)
                boxes = boxes.float()
                

                self.predictor.set_image(single_image)
                img_h, img_w = single_image.shape[:2]
                boxes_transformed = self.predictor.transform.apply_boxes_torch(boxes, (img_h, img_w))

                class_to_boxes = {}
                for idx, cls_id in enumerate(class_ids):
                    if cls_id not in class_to_boxes:
                        class_to_boxes[cls_id] = []
                    class_to_boxes[cls_id].append(idx)
                
  
                all_masks = []
                

                for cls_id, box_indices in sorted(class_to_boxes.items()):
                    cls_boxes = boxes_transformed[box_indices].to('cuda:0')  # (N_cls, 4)
                    

                    masks, scores, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=cls_boxes,  
                        multimask_output=True
                    )
        
         
                    for box_idx in range(masks.shape[0]):
                        box_masks = masks[box_idx]  # (3, H, W)
                        box_scores = scores[box_idx]  # (3,)
                        best_idx = torch.argmax(box_scores)
                        best_mask = box_masks[best_idx].cpu().numpy()  # (H, W)
                        if best_mask.sum()>0:
                            all_masks.append(best_mask)    
  
                if all_masks:
                    all_masks = np.stack(all_masks, axis=0)  # (num_boxes, H, W)
                    mask_list.append(torch.from_numpy(all_masks.astype(dtype=image.dtype)))
            else:
               return None
        
        return torch.stack(mask_list, dim=0)  # (B, num_masks, H, W)
    def restruct_x(self,x):
        CLIP_MEAN = [122.7709, 116.7460, 104.0937]
        CLIP_STD  = [68.5005, 66.6322, 70.3232]
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1).to(x.device)
        re_x = x * std + mean
        re_x = re_x.clamp(0, 255)
        return re_x
    
    
    def compute_semantic_score(self, mask, visual_feature_map, text_embedding, device):
        if visual_feature_map is None or text_embedding is None:
            return 0.0
        
        try:
            feat_h, feat_w = visual_feature_map.shape[:2]
            

            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (feat_w, feat_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            mask_tensor = torch.from_numpy(mask_resized).to(device)
            

            if mask_tensor.sum() == 0:
                return 0.0
            
            masked_features = visual_feature_map[mask_tensor]  # (N_pixels, D)
            avg_visual_feat = masked_features.mean(dim=0)  # (D,)
            
  
            avg_visual_feat = F.normalize(avg_visual_feat, p=2, dim=-1)
            text_emb_norm = F.normalize(text_embedding, p=2, dim=-1)
            

            cosine_sim = torch.dot(avg_visual_feat, text_emb_norm).item()

            semantic_score = max(0.0, cosine_sim)
            
            return semantic_score
            
        except Exception as e:
            print(f"[ERROR] Failed to compute semantic score: {e}")
            return 0.0
    
    def sample_around_points(self, points, img_shape, class_ids=None, prob_map=None):
        if points is None or len(points) == 0:
            return points, class_ids
        
        h, w = img_shape
        expanded_points = []
        expanded_class_ids = []
        

        angles = np.linspace(0, 2*np.pi, self.num_neighbor_samples, endpoint=False)
        
        for i, point in enumerate(points):

            expanded_points.append(point)
            if class_ids is not None:
                expanded_class_ids.append(class_ids[i])
            
         
            x, y = point[0].item(), point[1].item()
            for angle in angles:
                new_x = x + self.sample_radius * np.cos(angle)
                new_y = y + self.sample_radius * np.sin(angle)
                
          
                new_x = max(0, min(w - 1, new_x))
                new_y = max(0, min(h - 1, new_y))
                if prob_map is not None:
                    cls_id = class_ids[i]
                    new_intx, new_inty = int(new_x), int(new_y)
                    if prob_map[cls_id][new_inty, new_intx] < self.min_prob_thresh:
                        continue
                
                expanded_points.append(torch.tensor([new_x, new_y], dtype=torch.float32,device=points.device))
                if class_ids is not None:
                    expanded_class_ids.append(class_ids[i])
        
        expanded_points = torch.stack(expanded_points, dim=0)
        return expanded_points, expanded_class_ids
    
    def evaluate_mask_quality(self, mask, sam_score, point, prob_map, high_response_mask,
                             visual_feature_map=None, text_embedding=None):
        if mask.sum() == 0:
            return 0.0, {'purity': 0.0, 'coverage': 0.0, 'semantic': 0.0, 
                         'geometric': 0.0, 'score': 0.0}
        
        device = prob_map.device
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)
        mask_area = mask_tensor.sum() + 1e-6
        
    
        purity = (mask_tensor * prob_map).sum() / mask_area
        
    
        intersection = (mask_tensor * high_response_mask.float()).sum()
        coverage = intersection / (high_response_mask.float().sum() + 1e-6)
        
      
        semantic_score = 0.0
        if self.use_semantic and visual_feature_map is not None and text_embedding is not None:
            semantic_score = self.compute_semantic_score(
                mask, 
                visual_feature_map, 
                text_embedding, 
                device
            )
        
  
        geometric_score = purity.item() * (coverage.item() ** self.exp)
        
      
        quality_score = self.alpha * semantic_score + self.beta * geometric_score
        
        metrics = {
            'purity': purity.item(),
            'coverage': coverage.item(),
            'semantic': semantic_score,
            'geometric': geometric_score,
            'score': quality_score
        }
        
        return quality_score, metrics
    
    def mask_to_box(self, mask):
     
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
    
            return torch.tensor([0., 0., 0., 0.])
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
    
    def apply_nms_to_masks(self, masks, scores, class_ids, iou_threshold=0.5):
        if len(masks) == 0:
            return [], [], [], []
        
     
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        class_ids_tensor = torch.tensor(class_ids, dtype=torch.long)
        
  
        boxes = []
        for mask in masks:
            box = self.mask_to_box(mask)
            boxes.append(box)
        boxes_tensor = torch.stack(boxes)  # (N, 4)
        
      
        keep = batched_nms(boxes_tensor, scores_tensor, class_ids_tensor, iou_threshold)
        
 
        keep_indices = keep.cpu().numpy().tolist()
        filtered_masks = [masks[i] for i in keep_indices]
        filtered_scores = [scores[i] for i in keep_indices]
        filtered_class_ids = [class_ids[i] for i in keep_indices]
        
        return filtered_masks, filtered_scores, filtered_class_ids, keep_indices
    
    def compute_mask_similarity_matrix(self, masks, image_feature):
        n = len(masks)
        if n == 0:
            return np.array([])
        
        feat_h, feat_w = image_feature.shape[:2]
        feature_dim = image_feature.shape[2]
        
       
        mask_features_list = []
        for mask in masks:
        
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (feat_w, feat_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            mask_tensor = torch.from_numpy(mask_resized).to(image_feature.device)
            
        
            if mask_tensor.sum() > 0:
                mask_feature = image_feature[mask_tensor].mean(dim=0)
            else:
                mask_feature = torch.zeros(feature_dim, device=image_feature.device)
            
      
            mask_feature = F.normalize(mask_feature, p=2, dim=0)
            mask_features_list.append(mask_feature)
        
   
        mask_features = torch.stack(mask_features_list, dim=0)
        
        
        similarity_matrix = torch.mm(mask_features, mask_features.t())
        
        similarity_matrix = torch.clamp(similarity_matrix, min=0.0)
        
        return similarity_matrix.cpu().numpy()
        
            
    
    def cluster_and_merge_masks(self, masks, num_clusters, quality_scores=None,image_feature=None):
        if len(masks) == 0:
            return []
        
        if len(masks) == 1:
            return masks
        

        num_clusters = min(num_clusters, len(masks))
        
  
        n = len(masks)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                intersection = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                if union > 0:
                    iou = intersection / union
                else:
                    iou = 0.0
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
  
        np.fill_diagonal(iou_matrix, 1.0)

        distance_matrix = 1.0 - iou_matrix
        mask_similarity_matrix = self.compute_mask_similarity_matrix(masks, image_feature)
        cluster_weight_matrix = distance_matrix * (1 - mask_similarity_matrix)
        
        try:
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(cluster_weight_matrix)
        except:
            if quality_scores is not None:
                merged = self._weighted_merge_masks(masks, quality_scores)
            else:
                merged = np.logical_or.reduce(masks)
            return [merged]
        
        merged_masks = []
        for cluster_id in np.unique(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_masks = [masks[i] for i in cluster_indices]
            
            if len(cluster_masks) == 1:
                merged = cluster_masks[0]
            else:
                
                if quality_scores is not None:
                    cluster_scores = [quality_scores[i] for i in cluster_indices]
                    
                    center_idx = np.argmax(cluster_scores)
                    center_mask = cluster_masks[center_idx]
                    
             
                    weights = []
                    for i, mask in enumerate(cluster_masks):
                        if i == center_idx:
       
                            weights.append(cluster_scores[center_idx])
                        else:

                            intersection = np.logical_and(mask, center_mask).sum()
                            union = np.logical_or(mask, center_mask).sum()
                            iou = intersection / (union + 1e-6)
                            weights.append(cluster_scores[i] * iou)
                    
        
                    weights = np.array(weights)
                    weights = weights / (weights.sum() + 1e-6)  
                    
                    masks_array = np.stack(cluster_masks, axis=0).astype(np.float32)
                    merged = np.sum(masks_array * weights[:, None, None], axis=0)

     
                    merged = (merged > 0.5).astype(np.uint8)
                else:
                    merged = np.logical_or.reduce(cluster_masks).astype(np.uint8)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            merged = cv2.morphologyEx(merged.astype(np.uint8), 
                                     cv2.MORPH_CLOSE, kernel).astype(bool)
            
            merged_masks.append(merged)
        
        return merged_masks
    
    def merge_low_quality_masks(self, masks, quality_scores=None):
        if len(masks) == 0:
            return None
        
        if len(masks) == 1:
            return masks[0]
        
      
        if quality_scores is not None:
            merged = self._weighted_merge_masks(masks, quality_scores)
        else:
            
            merged = np.logical_or.reduce(masks).astype(np.uint8)
        
       
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        merged = cv2.morphologyEx(merged.astype(np.uint8), 
                                 cv2.MORPH_CLOSE, kernel)
        merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        return merged.astype(bool)
    
    def _weighted_merge_masks(self, masks, quality_scores):
        masks_array = np.stack(masks, axis=0).astype(np.float32)  # (N, H, W)
        scores = np.array(quality_scores)  # (N,)
        
       
        weights = scores / (scores.sum() + 1e-6)  # (N,)
        
        
        merged = np.sum(masks_array * weights[:, None, None], axis=0)  # (H, W)
        
      
        merged = (merged > 0.5).astype(np.uint8)
        
        return merged

    def _compute_semantic_ios(self, masks, obj_sim):
       
        n_masks = masks.shape[0]  
        if n_masks == 0:
            return torch.zeros(0, device=masks.device)


        flat_masks = masks.flatten(1).float()

        inter_num = flat_masks @ flat_masks.t()
        inter_num.fill_diagonal_(0.0)
        inter_num = torch.tril(inter_num, diagonal=0)
        pos_num = flat_masks.sum(dim=1)
        _ios = inter_num / (pos_num[:, None] + 1e-6)
        if obj_sim is not None:
            _ios = _ios * obj_sim
            
        ios = _ios.max(dim=-1)[0]
        return ios
    
    def get_masks_enhanced(self, image, prompts, logits, text_embeddings=None, spatial_features=None, masks=None):
        visual_features_batch = spatial_features if self.use_semantic else None
        

        image_for_sam = image.cpu()
        image_for_sam = np.array(image_for_sam)
        image_for_sam = self.restruct_x(torch.tensor(image_for_sam)).numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        

        logits_agg = logits.mean(dim=0)  # (B, #classes, W, H)
        img_h, img_w = image_for_sam.shape[1:3]
        logits_agg = F.interpolate(
            logits_agg, 
            size=(img_h, img_w), 
            mode='bilinear', 
            align_corners=False
        )
        probs = logits_agg.softmax(dim=1)  # (B, #classes, H, W)
        mask_list = []
        mask_scores_list = []
        
        for b_idx in range(image_for_sam.shape[0]):
            single_image = image_for_sam[b_idx]
            
 
            visual_feature_map = None
            if self.use_semantic and visual_features_batch is not None:
                visual_feature_map = visual_features_batch[b_idx]
            
            if masks is not None:
                if b_idx >= masks.shape[0]:
                    h, w = single_image.shape[:2]
                    mask_list.append(torch.from_numpy(np.zeros((1, h, w), dtype=np.uint8)))
                    continue
                pre = masks[b_idx].detach().float()
                if pre.dim() == 2:
                    pre = pre.unsqueeze(0)
                if pre.shape[-2:] != (img_h, img_w):
                    pre = F.interpolate(
                        pre.unsqueeze(1), size=(img_h, img_w), mode='nearest'
                    ).squeeze(1)
                probs_b = probs[b_idx]
                dev = probs_b.device
                all_masks = []
                all_scores = []
                all_class_ids_list = []
                for i in range(pre.shape[0]):
                    pi = pre[i]
                    p_cpu = pi.cpu().numpy()
                    if p_cpu.max() <= 1.0 + 1e-5:
                        bi = (p_cpu > 0.5).astype(np.uint8)
                    else:
                        bi = (p_cpu > 0).astype(np.uint8)
                    if bi.sum() == 0:
                        continue
                    all_masks.append(bi)
                    mb_t = torch.from_numpy(bi.astype(bool)).to(dev)
                    pc = probs_b[:, mb_t].reshape(probs_b.shape[0], -1).mean(dim=1)
                    all_scores.append(float(pc.max().item()))
                    all_class_ids_list.append(int(pc.argmax().item()))
                all_sam_scores = list(all_scores)
                all_points_orig = [None] * len(all_masks)
            else:
       
                if 'points' not in prompts[b_idx]:
                 
                    h, w = single_image.shape[:2]
                    empty_mask = np.zeros((1, h, w), dtype=np.uint8)
                    mask_list.append(torch.from_numpy(empty_mask))
                    continue
                
                point_coords = prompts[b_idx]['points']
                class_ids = prompts[b_idx]['class_ids']
                
                if point_coords is None or len(point_coords) == 0:
                    h, w = single_image.shape[:2]
                    empty_mask = np.zeros((1, h, w), dtype=np.uint8)
                    mask_list.append(torch.from_numpy(empty_mask))
                    continue
                
               
                if isinstance(point_coords, np.ndarray):
                    point_coords = torch.from_numpy(point_coords)
                point_coords = point_coords.float()
                
                expand_point = False
                if expand_point:
        
                    expanded_points, expanded_class_ids = self.sample_around_points(
                        point_coords, 
                        (img_h, img_w), 
                        class_ids,
                        probs[b_idx]
                    )
                
                else:
             
                    expanded_points = point_coords
                    expanded_class_ids = class_ids
        
                self.predictor.set_image(single_image)
                points_transformed = self.predictor.transform.apply_coords_torch(
                    expanded_points.unsqueeze(1), (img_h, img_w)
                )  # (N, 1, 2)
                
                all_masks = []
                all_scores = []
                all_sam_scores = []
                all_points_orig = []
                all_class_ids_list = []
                
              
                class_to_points = {}
                for idx, cls_id in enumerate(expanded_class_ids):
                    if cls_id not in class_to_points:
                        class_to_points[cls_id] = []
                    class_to_points[cls_id].append(idx)
                
               
                for cls_id, point_indices in sorted(class_to_points.items()):
                    cls_points = points_transformed[point_indices].to(device='cuda:0')
                    cls_labels = torch.ones(len(point_indices), 1, dtype=torch.int, device=cls_points.device)
                    

                    sam_pred_masks, sam_scores, _ = self.predictor.predict_torch(
                        point_coords=cls_points,
                        point_labels=cls_labels,
                        boxes=None,
                        multimask_output=True
                    )
                    
                    prob_map = probs[b_idx, cls_id]  # (H, W)
                    max_prob = prob_map.max()
                    thresh = self.response_thresh_ratio * max_prob
                    high_response_mask = (prob_map > thresh)  # (H, W)
                    
                  
                    text_embedding = None
                    if self.use_semantic and text_embeddings is not None:
                        text_embedding = text_embeddings[cls_id]  # (D,)


                    # for pt_idx, orig_idx in enumerate(point_indices):
                    #     pt_masks = sam_pred_masks[pt_idx]  # (3, H, W)
                #     pt_scores = sam_scores[pt_idx]  # (3,)
                    
                #     
                #     best_quality_score = -1
                #     best_mask = None
                #     best_sam_score = None
                #     best_metrics = None
                    
                #     
                #     for layer_idx in range(3):
                #         
                #         candidate_mask = pt_masks[layer_idx].cpu().numpy()  # (H, W)
                #         candidate_sam_score = pt_scores[layer_idx].item()
                        
                #         
                #         if candidate_mask.sum() == 0:
                #             continue
                        
                #        
                #         point_orig = expanded_points[orig_idx].cpu().numpy()
                #         quality_score, metrics = self.evaluate_mask_quality(
                #             candidate_mask, 
                #             candidate_sam_score, 
                #             point_orig, 
                #             prob_map, 
                #             high_response_mask,
                #             visual_feature_map=visual_feature_map,
                #             text_embedding=text_embedding
                #         )
                        
                #        
                #         if quality_score > best_quality_score:
                #             best_quality_score = quality_score
                #             best_mask = candidate_mask
                #             best_sam_score = candidate_sam_score
                #             best_metrics = metrics
                    
                #    
                #     if best_mask is not None and best_quality_score > 0.1:
                #         all_masks.append(best_mask)
                #         all_scores.append(best_quality_score)
                #         all_sam_scores.append(best_sam_score)
                #         all_points_orig.append(point_orig)
                #         all_class_ids_list.append(cls_id)
                

                    for pt_idx, orig_idx in enumerate(point_indices):
    
                        pt_masks = sam_pred_masks[pt_idx]  # (3, H, W)
                        pt_scores = sam_scores[pt_idx]  # (3,)
                        

                        best_idx = torch.argmax(pt_scores)
                        best_mask = pt_masks[best_idx].cpu().numpy()  # (H, W)
                        best_sam_score = pt_scores[best_idx].item()
                        
      
                        if best_mask.sum() == 0:
                            continue
                        
     
                        point_orig = expanded_points[orig_idx].cpu().numpy()
                        quality_score, metrics = self.evaluate_mask_quality(
                            best_mask, 
                            best_sam_score, 
                            point_orig, 
                            prob_map, 
                            high_response_mask,
                            visual_feature_map=visual_feature_map,
                            text_embedding=text_embedding
                        )
                        
                        all_masks.append(best_mask)
                        all_scores.append(quality_score)
                        all_sam_scores.append(best_sam_score)
                        all_points_orig.append(point_orig)
                        all_class_ids_list.append(cls_id)
         
            if len(all_masks) == 0:
                h, w = single_image.shape[:2]
                empty_mask = np.zeros((1, h, w), dtype=np.uint8)
                mask_list.append(torch.from_numpy(empty_mask))
                mask_scores_list.append(torch.zeros(1))
                continue

            validation_mode = 'merge'    #  'merge', 'top10', 'random10'
            
            if validation_mode == 'random10':
                all_scores = np.array(all_scores)
                total_masks = len(all_scores)
                
         
                sorted_indices = np.argsort(all_scores)[::-1]  
                final_masks = []
                top_n = min(10, total_masks)
                random_indices = np.random.choice(total_masks, size=top_n, replace=False)
                final_masks = [all_masks[i] for i in random_indices]
                if len(final_masks) > 0:
                    final_masks_array = np.stack(final_masks, axis=0).astype(np.uint8)
                    mask_list.append(torch.from_numpy(final_masks_array))
                    
            elif validation_mode == 'top10':
         
                all_scores = np.array(all_scores)
                total_masks = len(all_scores)
                

                sorted_indices = np.argsort(all_scores)[::-1]  
                final_masks = []
                top_n = min(10, total_masks)
                top_indices = sorted_indices[:top_n]
                final_masks = [all_masks[i] for i in top_indices]
                if len(final_masks) > 0:
                    final_masks_array = np.stack(final_masks, axis=0).astype(np.uint8)
                    mask_list.append(torch.from_numpy(final_masks_array))
   
                # all_scores = np.array(all_scores)
                # total_masks = len(all_scores)
                
                
                # mask_areas = np.array([mask.sum() for mask in all_masks])
                
               
                # sorted_indices = np.argsort(mask_areas)[::-1]  
                # final_masks = []
                # top_n = min(10, total_masks)
                # top_indices = sorted_indices[:top_n]
                # final_masks = [all_masks[i] for i in top_indices]
                # if len(final_masks) > 0:
                #     final_masks_array = np.stack(final_masks, axis=0).astype(np.uint8)
                #     mask_list.append(torch.from_numpy(final_masks_array))

            else:  # validation_mode == 'merge'

                if self.enable_nms and self.Sinkhorn_clustered:
                # if self.enable_nms :

                    nms_masks, nms_scores, nms_class_ids, keep_indices = self.apply_nms_to_masks(
                        all_masks, all_scores, all_class_ids_list, self.nms_iou_threshold
                    )
                    # print(f"  [NMS] Batch {b_idx}: {len(all_masks)} masks -> {len(nms_masks)} masks (IoU={self.nms_iou_threshold})")
                    
        
                    all_masks = nms_masks
                    all_scores = nms_scores
                    all_sam_scores = [all_sam_scores[i] for i in keep_indices]
                    all_points_orig = [all_points_orig[i] for i in keep_indices]
                    all_class_ids_list = nms_class_ids

                if self.soft_merging:
                   
                    keep = len(all_masks)
                    scores_out = all_scores 
                    
                    masks_out_binary = all_masks  
                    masks_out_binary = torch.stack([
                        torch.from_numpy(m) if isinstance(m, np.ndarray) else m 
                        for m in masks_out_binary
                    ], dim=0).to('cuda')
                    
                    feat_H, feat_W, D = spatial_features[0].shape
                    
                    masks_small = F.interpolate(
                        masks_out_binary.unsqueeze(1).float(),  
                        size=(feat_H, feat_W),                  
                        mode='nearest'                          
                    ).squeeze(1).bool()
               
                    
        
                    obj_feats_out = []
                    for i in range(keep):
                  
                        m = masks_small[i] 
                        
                        if m.sum() > 0:
              
                            f = spatial_features[0][m].mean(dim=0)
               
                        else:

                            f = torch.zeros(D, device=self.device, dtype=spatial_features[0].dtype)
                        
                        obj_feats_out.append(f)
                    
    
                    obj_feats_out = torch.stack(obj_feats_out)
             
    
                    obj_feats_out = F.normalize(obj_feats_out, p=2, dim=-1)
            
           
                    obj_sim = obj_feats_out @ obj_feats_out.t()
                    
                    
                    obj_sim = obj_sim.clamp(min=0.0)
                    
                    if isinstance(scores_out, list):
                        scores_out = torch.tensor(scores_out, device='cuda', dtype=torch.float32)
                    elif isinstance(scores_out, np.ndarray):
                        scores_out = torch.from_numpy(scores_out).to('cuda').float()
                    
                    sorted_vals, sorted_idx = torch.sort(scores_out, descending=True)
                    
          
                    masks_out_binary_sorted = masks_out_binary[sorted_idx]
                    obj_sim_sorted = obj_sim[sorted_idx][:, sorted_idx]
                    
         
                    ios = self._compute_semantic_ios(masks_out_binary_sorted, obj_sim_sorted)
            
                    
       
                    score_decay = 1.0 - ios
                 
            
                
                    scores_decayed = sorted_vals * torch.pow(score_decay, 0.5)
                   
                    
             
                    top_k = min(10, len(masks_out_binary_sorted))
   
                    final_masks_tensor = masks_out_binary_sorted[:top_k]
        
                    final_scores_tensor = scores_decayed[:top_k]

                    
                    final_masks = [m for m in final_masks_tensor]
        
                    final_scores = final_scores_tensor.tolist()
                    
  
                    if len(final_masks) > 0:
    
                        final_masks_cpu = [m.cpu().numpy() if isinstance(m, torch.Tensor) else m for m in final_masks]
                        final_masks_array = np.stack(final_masks_cpu, axis=0).astype(np.uint8)
                        mask_list.append(torch.from_numpy(final_masks_array))
                        mask_scores_list.append(final_scores_tensor)
                    


                else:
                    all_scores = np.array(all_scores)
                    total_masks = len(all_scores)
                    
                    sorted_indices = np.argsort(all_scores)[::-1] 
                    final_masks = []
                    num_high = min(self.num_high_quality, total_masks)
                    num_mid = min(self.num_mid_quality, max(0, total_masks - num_high))
                    num_low = min(self.num_low_quality, max(0, total_masks - num_high - num_mid))
                    
                   
                    high_indices = sorted_indices[:num_high]
                    mid_indices = sorted_indices[num_high:num_high + num_mid]
                    low_indices = sorted_indices[num_high + num_mid:num_high + num_mid + num_low]
                    

                    
                
                    

                    if num_high > 0:
                        high_masks = [all_masks[i] for i in high_indices]
                        final_masks.extend(high_masks)

                    

                    if num_mid > 0:
                        mid_masks = [all_masks[i] for i in mid_indices]
                        mid_scores = [all_scores[i] for i in mid_indices]
                        clustered = self.cluster_and_merge_masks(mid_masks, self.mid_cluster_num, mid_scores,spatial_features[0])
                        final_masks.extend(clustered)
  
                    
   
                    if num_low > 0:
                        low_masks = [all_masks[i] for i in low_indices]
                        low_scores = [all_scores[i] for i in low_indices]
                        merged_low = self.merge_low_quality_masks(low_masks, low_scores)
                        if merged_low is not None:
                            final_masks.append(merged_low)

                    

                    if len(final_masks) > 0:
                        final_masks_array = np.stack(final_masks, axis=0).astype(np.uint8)
                        mask_list.append(torch.from_numpy(final_masks_array))
           
                    else:
                        h, w = single_image.shape[:2]
                        empty_mask = np.zeros((1, h, w), dtype=np.uint8)
                        mask_list.append(torch.from_numpy(empty_mask))
        
        return torch.stack(mask_list, dim=0)   # (B, num_masks, H, W)
    
    def get_masks_vis(self, image, prompts, logits=None, text_embeddings=None, spatial_features=None):
        

        image = image.cpu()
        image = np.array(image)
        image = self.restruct_x(torch.tensor(image)).numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        
        mask_list = []
        for i in range(image.shape[0]):
            single_image = image[i]
            

            if 'points' in prompts[i]:
           
                point_coords = prompts[i]['points']
                class_ids = prompts[i]['class_ids']
                
       
                if point_coords is None or len(point_coords) == 0:
                    h, w = single_image.shape[:2]
                    empty_mask = np.zeros((1, h, w), dtype=image.dtype)
                    mask_list.append(torch.from_numpy(empty_mask))
                    print("No valid points found for image.")
                    continue
                
   
                if isinstance(point_coords, np.ndarray):
                    point_coords = torch.from_numpy(point_coords)
                point_coords = point_coords.float()
                
   
                self.predictor.set_image(single_image)
                img_h, img_w = single_image.shape[:2]
                points_transformed = self.predictor.transform.apply_coords_torch(
                    point_coords.unsqueeze(1), (img_h, img_w)
                )  # (N, 1, 2)
                

                class_to_points = {}
                for idx, cls_id in enumerate(class_ids):
                    if cls_id not in class_to_points:
                        class_to_points[cls_id] = []
                    class_to_points[cls_id].append(idx)
                
     
                all_masks = []
                

                for cls_id, point_indices in sorted(class_to_points.items()):
                    cls_points = points_transformed[point_indices].to(device='cuda:0')  # (N_cls, 1, 2)
                    cls_labels = torch.ones(len(point_indices), 1, dtype=torch.int, device=cls_points.device)
                    
   
                    masks, scores, _ = self.predictor.predict_torch(
                        point_coords=cls_points,  # (N_cls, 1, 2)
                        point_labels=cls_labels,  # (N_cls, 1)
                        boxes=None,
                        multimask_output=True
                    )
    
     
                    for pt_idx in range(masks.shape[0]):
                        pt_masks = masks[pt_idx]  # (3, H, W)
                        pt_scores = scores[pt_idx]  # (3,)
                        best_idx = torch.argmax(pt_scores)
                        best_mask = pt_masks[best_idx].cpu().numpy()  # (H, W)
                        mask_area = best_mask.sum()
                        if mask_area>0:
                            all_masks.append(best_mask)

                

                if all_masks:
                    all_masks = np.stack(all_masks, axis=0)  # (num_points, H, W)
                    mask_list.append(torch.from_numpy(all_masks.astype(dtype=image.dtype)))
                else:
                    return None
        
        return torch.stack(mask_list, dim=0)  # (B, num_masks, H, W)
    
    def get_masks_auto(self, image, spatial_features=None):

        image_for_sam = image.cpu()
        image_for_sam = np.array(image_for_sam)
        image_for_sam = self.restruct_x(torch.tensor(image_for_sam)).numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        
        mask_list = []
        
        for b_idx in range(image_for_sam.shape[0]):
            single_image = image_for_sam[b_idx]  # (H, W, 3)
            img_h, img_w = single_image.shape[:2]
            
          
            masks_data = self.auto_mask_generator.generate(single_image)
            
            if len(masks_data) == 0:
                h, w = single_image.shape[:2]
                empty_mask = np.zeros((1, h, w), dtype=np.uint8)
                mask_list.append(torch.from_numpy(empty_mask))
                continue
            

            all_masks = [m['segmentation'].astype(np.uint8) for m in masks_data]
            all_scores = [m['predicted_iou'] * m['stability_score'] for m in masks_data]
     
            if self.enable_nms:
               
                dummy_class_ids = [0] * len(all_masks)
                nms_masks, nms_scores, _, keep_indices = self.apply_nms_to_masks(
                    all_masks, all_scores, dummy_class_ids, self.nms_iou_threshold
                )
                all_masks = nms_masks
                all_scores = nms_scores
            
        
            validation_mode = 'merge'  
            
            if validation_mode == 'random10':
                all_scores = np.array(all_scores)
                total_masks = len(all_scores)
                top_n = min(10, total_masks)
                random_indices = np.random.choice(total_masks, size=top_n, replace=False)
                final_masks = [all_masks[i] for i in random_indices]
                
                if len(final_masks) > 0:
                    final_masks_array = np.stack(final_masks, axis=0).astype(np.uint8)
                    mask_list.append(torch.from_numpy(final_masks_array))
                else:
                    empty_mask = np.zeros((1, img_h, img_w), dtype=np.uint8)
                    mask_list.append(torch.from_numpy(empty_mask))
                    
            elif validation_mode == 'top10':
                all_scores = np.array(all_scores)
                sorted_indices = np.argsort(all_scores)[::-1]
                top_n = min(10, len(all_scores))
                final_masks = [all_masks[i] for i in sorted_indices[:top_n]]
                
                if len(final_masks) > 0:
                    final_masks_array = np.stack(final_masks, axis=0).astype(np.uint8)
                    mask_list.append(torch.from_numpy(final_masks_array))
                else:
                    empty_mask = np.zeros((1, img_h, img_w), dtype=np.uint8)
                    mask_list.append(torch.from_numpy(empty_mask))
                    
            else:  
                all_scores = np.array(all_scores)
                total_masks = len(all_scores)
                
               
                sorted_indices = np.argsort(all_scores)[::-1]
                

                num_high = min(self.num_high_quality, total_masks)
                num_mid = min(self.num_mid_quality, max(0, total_masks - num_high))
                num_low = min(self.num_low_quality, max(0, total_masks - num_high - num_mid))
                
                high_indices = sorted_indices[:num_high]
                mid_indices = sorted_indices[num_high:num_high + num_mid]
                low_indices = sorted_indices[num_high + num_mid:num_high + num_mid + num_low]
                
                final_masks = []
                
     
                if num_high > 0:
                    high_masks = [all_masks[i] for i in high_indices]
                    final_masks.extend(high_masks)
                
        
                if num_mid > 0:
                    mid_masks = [all_masks[i] for i in mid_indices]
                    mid_scores = [all_scores[i] for i in mid_indices]

                    feat = spatial_features[b_idx] if spatial_features else None
                    clustered = self.cluster_and_merge_masks(mid_masks, self.mid_cluster_num, mid_scores, feat)
                    final_masks.extend(clustered)
                
         
                if num_low > 0:
                    low_masks = [all_masks[i] for i in low_indices]
                    low_scores = [all_scores[i] for i in low_indices]
                    merged_low = self.merge_low_quality_masks(low_masks, low_scores)
                    if merged_low is not None:
                        final_masks.append(merged_low)
 
                if len(final_masks) > 0:
                    final_masks_array = np.stack(final_masks, axis=0).astype(np.uint8)
                    mask_list.append(torch.from_numpy(final_masks_array))
                else:
                    empty_mask = np.zeros((1, img_h, img_w), dtype=np.uint8)
                    mask_list.append(torch.from_numpy(empty_mask))
        
        return torch.stack(mask_list, dim=0)  # (B, num_masks, H, W)


