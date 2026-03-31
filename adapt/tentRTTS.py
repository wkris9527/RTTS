import time
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils_local.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters
from utils_local.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters
from torchvision import transforms
from utils_local.logits_sam import SAMwithlogits         ##sam1
import numpy as np

REFERENCE_PROMPT = 'a photo of a {}'


class TENT:
    """
    Test-time adaptation for open-vocabulary semantic segmentation (OVSS) models using TENT.

    Performs iterative optimization of the visual encoder LayerNorm parameters to reduce predictive uncertainty 
    based on the softmax output distribution.

    Inspired by TENT GitHub: https://github.com/DequanWang/tent
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=10, 
                 prompt_dir=None, runtime_calculation=False,
                 device='cpu', 
                 ):
        """
        Initialize the TENT adaptation module.

        Args:
            ovss_type (str): Identifier for the open-vocabulary segmentation model to load.
            ovss_backbone (str): Name of the backbone architecture within the OVSS model.
            lr (float): Learning rate for the LayerNorm optimizer.
            classes (List[str]): List of class names for prompt generation.
            steps (int, optional): Number of adaptation iterations per sample. Defaults to 10.
            prompt_dir (str or None, optional): Path to YAML file with prompt templates. Defaults to None.
            runtime_calculation (bool, optional): Whether to record adaptation/evaluation runtimes. Defaults to False.
            device (str, optional): Compute device, e.g., 'cpu' or 'cuda'. Defaults to 'cpu'.
        """

        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr

        if classes is not None:
            self.classes = classes
        else:
            raise Exception("Classes are required in the init")
        
        self.prompt_dir = prompt_dir
        self.steps = steps
        self.runtime = runtime_calculation
        self.device = device

        # Load the OVSS model and tokenizer
        self.model, self.tokenize = load_ovss(self.ovss_type, self.ovss_backbone, device=self.device)

        if self.prompt_dir:
            # Load the prompt templates
            self.prompt_templates = load_prompts_from_yaml(self.prompt_dir)
            # print the number of prompt templates
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # Set the gradients for LayerNorm layers only for visual encoder
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)

        self.model.visual = self.set_ln_grads(self.model.visual)

        # Collect the LayerNorm parameters
        params, _ = self.collect_ln_params(self.model.visual)

        # print the parameters
        print_clip_parameters(self.model)

        # Set the optimizer
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        # print the parameters passed to the optimizer
        print_optimizer_parameters(self.optimizer, self.model)

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        # extracting text features
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(self.classes, self.prompt_templates, average=False).squeeze() # (class, 512)

        # define variables to store adaptation and evaluation duration
        if self.runtime:
            self.adapt_times = []
            self.eval_times = []
            
        self.use_sam = False    
        self.voted = False      
        self.clsvoted = False   
        self.logits2prompt = True  
        self.logits2prompt_iterations = 2 
        self.random_mask = False  
        self.Sinkhorn_clustered = True  
        self.prompt_type = 'point' 
        self.logits2sam = SAMwithlogits(
                num_sample_points=10,
                response_thresh_ratio=0.5,
                min_prob_thresh=0.1,
                min_mask_area=100,
                prompt_type=self.prompt_type,
                enable_enhanced=True,
                use_semantic=True,       
                alpha=0.3,               
                beta=0.7,                 
                Sinkhorn_clustered = self.Sinkhorn_clustered,
            )
        self.patch_size = 32 


    def adapt(self, x):
        """
        Forward pass with adaptation.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            List[float]: Loss values recorded at each adaptation iteration.
        """

        self.reset()
        loss_report = self.perform_adaptation(x)
        return loss_report

    @torch.no_grad() 
    def evaluate(self, x):
        """
        Forward pass without adaptation.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Per-class logits of shape (batch_size, num_classes, H, W).

        """

        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x, True, 
                                  interpolate=False) # (#template, batch_size, #classes, H, W)
        image_features = self.model.encode_image(x, return_vanilla_cls=False, out_type="mean")  # (batch_size, num_patches+1, D)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.updatelogits(logits, x, self.text_x, image_features)
        logits = self.interpolate_logits(logits, x)
        logits = logits[0]
        t2 = time.time()
        if self.runtime:
            self.eval_times.append(t2-t1)

        return logits

    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)

    def perform_adaptation(self, x):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).
        
        Returns:
            List[float]: Recorded loss values for each adaptation iteration.
        """

        t1 = time.time()
        loss_report = []
        for iter in range(self.steps):
            logits, _, _ = self.model(x, self.text_x, True, 
                                      interpolate=False)  # (#template, batch_size, #classes, H, W)
            
            # adapt
            entropy_per_pixel = self.softmax_entropy(logits)  # Shape: (#template, batch_size, H, W)
            # Average over all prompts, pixels and batch samples
            loss = entropy_per_pixel.mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        t2 = time.time()
        if self.runtime:
            self.adapt_times.append(t2-t1)

        return loss_report

    def extract_text_embeddings(self, class_names, prompts, average=True):
        """
        Extracts text embeddings for given class names and prompts.
        Args:
            class_names: List of class names to generate text embeddings for.
            prompts: List of prompt templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different templates for each class.
        Returns:
            text_features: Tensor of text embeddings for the given class names and prompts.
        """
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)  # Shape: (#templates, 512)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                class_embeddings_avg = class_embeddings.mean(dim=0)  # Shape: (512,)
                class_embeddings_avg = class_embeddings_avg / class_embeddings_avg.norm()
                # add the averaged embeddings to the original embeddings
                class_embeddings = torch.cat([class_embeddings, class_embeddings_avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features

    @staticmethod
    def set_ln_grads(model):
        """
        Set gradient settings for LayerNorm layers within the model, disabling gradients globally except for these LN layers.
        Args:
            model: The model whose LayerNorm layers' gradients are to be set.
        Returns:
            The model with modified gradient settings.
        """
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        """
        Collect the affine scale and shift parameters from LayerNorm layers.
        Args:
            model: The model from which to collect LayerNorm parameters.
        Returns:
            params: List of LayerNorm parameters.
            names: List of parameter names.
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """
        Copy the model and optimizer states for resetting after adaptation.
        Args:
            model: The model to copy.
            optimizer: The optimizer to copy.
        Returns:
            model_state: Copied state of the model.
            optimizer_state: Copied state of the optimizer.
        """
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """
        Restore the model and optimizer states from copies.
        Args:
            model: The model to restore.
            optimizer: The optimizer to restore.
            model_state: The state to restore the model to.
            optimizer_state: The state to restore the optimizer to.
        """
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits.
            x : torch.Tensor : logits of shape (#templates, batch_size, num_classes, H, W)
        """
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)
    
    def interpolate_logits(self, logits, image):
        patch_size = self.patch_size
        w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
        temp_dim = logits.shape[0]
        b_dim = logits.shape[1]
        out_dim = logits.shape[2]
        # Perform interpolation
        logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
        logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#templates*batch_size, #class, W', H')

        # Reshape back to include template and batch dimensions
        logits = logits.view(temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#templates, batch_size, #class, W', H')
        return logits
    def updatelogits(self,logits,image,text_features,image_features):
        batch_size = logits.shape[1]
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.unsqueeze(0) if text_features.dim() == 2 else text_features  
        text_embeddings_for_scoring = text_features.mean(dim=0) if text_features.dim() == 3 else text_features  
        spatial_features_batch = []
        if self.logits2sam.use_semantic:
            for b_idx in range(batch_size):
                spatial_feat = self._convert_tokens_to_spatial_features(image_features, batch_idx=b_idx)
                spatial_features_batch.append(spatial_feat)
        else:
            spatial_features_batch = None
        
        for iteration in range(self.logits2prompt_iterations):
            if self.prompt_type == 'box':
                batch_prompts = self.logits2sam.generate_box_prompts(logits, original_img_shape=image.shape[-2:])
            elif self.prompt_type == 'point':
                batch_prompts = self.logits2sam.generate_prompts(logits, original_img_shape=image.shape[-2:])
            else:
                raise ValueError(f"Unknown prompt_type: {self.prompt_type}. Must be 'point' or 'box'.")
            
            masks  = self.logits2sam.get_masks(
                image, 
                batch_prompts, 
                logits,
                text_embeddings=text_embeddings_for_scoring,  
                spatial_features=spatial_features_batch       
            )
            # if masks is not None:
            #     _, num_masks, mask_h, mask_w = masks.shape
            #     resize_transform = transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)##对于VITL14
            #     masks = resize_transform(masks).to(image.dtype)##torch.Size([b, m, 16, 16]) 
            #     batch_masks_filtered = []
            #     for b in range(masks.shape[0]):
            #         mask_no_empty_ = []
            #         for m in range(masks.shape[1]):
            #             mask = masks[b][m]
            #             if mask.sum() != 0:
            #                 mask_no_empty_.append(mask)
            #         if len(mask_no_empty_) > 0:
            #             batch_masks_filtered.append(torch.stack(mask_no_empty_, dim=0))
            #         else:
            #             batch_masks_filtered.append(torch.zeros((1, 16, 16), dtype=masks.dtype, device=masks.device))

            #     max_num_masks = max(m.shape[0] for m in batch_masks_filtered)
            #     padded_batch_masks = []
            #     for batch_mask in batch_masks_filtered:
            #         num_m = batch_mask.shape[0]
            #         if num_m < max_num_masks:
            #             padding = torch.zeros((max_num_masks - num_m, 16, 16), dtype=batch_mask.dtype, device=batch_mask.device)
            #             padded_mask = torch.cat([batch_mask, padding], dim=0)
            #             padded_batch_masks.append(padded_mask)
            #         else:
            #             padded_batch_masks.append(batch_mask)

            #     masks = torch.stack(padded_batch_masks, dim=0)
                
            #     if max_num_masks == 1 and all(m[0].sum() == 0 for m in batch_masks_filtered):
            #         masks = None
            if masks is not None:
                _, num_masks, mask_h, mask_w = masks.shape
                resize_transform = transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)##vit14
                masks = resize_transform(masks).to(image.dtype)
                mask_no_empty_ = []
                for m in range(masks.shape[1]):
                    mask = masks[0][m]
                    if mask.sum() !=0:
                            mask_no_empty_.append(mask)
                if len(mask_no_empty_)==0:
                    masks = None
                else:
                    masks = torch.stack( mask_no_empty_, dim=0).unsqueeze(0) 
                if masks is not None:
                    _, num_masks, mask_h, mask_w = masks.shape
                    output_logits = logits.clone()
                    for b in range(batch_size):
                        for m in range(num_masks - 1, -1, -1):  
                            mask_binary = masks[b, m] > 0  
                            if mask_binary.sum() > 0:  
                                mask_logits = logits[:, b, :, mask_binary]  
                                voted_logits = mask_logits.mean(dim=-1, keepdim=True)  
                                output_logits[:, b, :, mask_binary] = voted_logits
                    if self.Sinkhorn_clustered:
                        image_features_masked = self.masked_pooling(
                            image_features[:,1:,].permute(0,2,1).reshape(
                                image_features.shape[0], image_features.shape[2], 16, 16
                            ), 
                            masks
                        )
                        
                        image_features_masked_sim = image_features_masked.float()
                        num_templates_cluster = text_features.shape[0]
                        num_classes_cluster = text_features.shape[1]
                        B_cluster = image_features_masked.shape[0]
                        num_masks_cluster = image_features_masked.shape[1]
                        
                        logits_clustered_list = []
                        
                        for t_idx in range(num_templates_cluster):
                            text_features_sim =  text_features[t_idx].float()
                            similarity_logits = image_features_masked_sim @ text_features_sim.T
                            
                            similarity_flat = similarity_logits.reshape(-1, num_classes_cluster)  # [B*num_masks, num_classes]
                            
                            assignments = self._distributed_sinkhorn(similarity_flat, n_iters=3)  # [B*num_masks, num_classes]
                            assignments = assignments.reshape(B_cluster, num_masks_cluster, num_classes_cluster)  # [B, num_masks, num_classes]
                            
                            spatial_logits = torch.einsum('bmc,bmhw->bchw', assignments.to(masks.device), masks.float()).to("cuda")
                            logits_clustered_list.append(spatial_logits)
                        
                        logits_clustered = torch.stack(logits_clustered_list, dim=0)
                        logits_clustered = logit_scale * logits_clustered
                    
                        logits =  logits + 0.5 * (output_logits + logits_clustered) 
                    else:
                        logits = output_logits
                else:
                    break
            else:
                break
        return logits
    
    def _convert_tokens_to_spatial_features(self, image_features, batch_idx=0):
        
        if image_features.dim() == 4:
            features = image_features[-1, batch_idx]  
        elif image_features.dim() == 3:
            features = image_features[batch_idx]  
        else:
            raise ValueError(f"Unexpected image_features dimension: {image_features.dim()}")
        
        spatial_tokens = features[1:, :]  
    
        n_patches = spatial_tokens.shape[0]
        grid_size = int(n_patches ** 0.5)
        
        if grid_size * grid_size != n_patches:
            raise ValueError(f"Cannot reshape {n_patches} patches to square grid")
        
        spatial_feature_map = spatial_tokens.reshape(grid_size, grid_size, -1)
        
        return spatial_feature_map
          
    def erode_mask_to_center(self, masks, min_mask_area=20, erosion_kernel_size=3, erosion_iterations=1):
        """
        """
        B, num_masks, H, W = masks.shape
        processed_masks = []
        
        for b in range(B):
            batch_processed = []
            for m in range(num_masks):
                mask = masks[b, m]  # [H, W]
                
                mask_binary = (mask > 0.5).float()
                mask_area = mask_binary.sum().item()
                
                if mask_area < min_mask_area:
                    batch_processed.append(mask)
                    continue
                
                eroded_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                kernel = torch.ones(1, 1, erosion_kernel_size, erosion_kernel_size, 
                                   device=masks.device, dtype=masks.dtype)
                padding = erosion_kernel_size // 2
                
                for _ in range(erosion_iterations):
                    conv_result = F.conv2d(eroded_mask, kernel, padding=padding)
                    
                    threshold = kernel.sum() * 0.8 
                    eroded_mask = (conv_result >= threshold).float()
                
                eroded_mask = eroded_mask.squeeze(0).squeeze(0)  # [H, W]
                
                if eroded_mask.sum() < 1e-6:
                    batch_processed.append(mask)
                else:
                    batch_processed.append(eroded_mask)
            
            processed_masks.append(torch.stack(batch_processed, dim=0))
        
        return torch.stack(processed_masks, dim=0)  # [B, num_masks, H, W]
    def masked_pooling(self, features, masks, use_center_only=True, min_mask_area=20, 
                      erosion_kernel_size=3, erosion_iterations=1):
        B, C, H, W = features.shape
        num_masks = masks.shape[1]
        if use_center_only:
            masks = self.erode_mask_to_center(masks, min_mask_area, 
                                             erosion_kernel_size, erosion_iterations)
        masks = masks.to(device = features.device, dtype = features.dtype)
        mask_sums = masks.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)  # [B, num_masks, 1, 1]
        normalized_masks = masks / mask_sums  # [B, num_masks, H, W]
        features_expanded = features.unsqueeze(1)  # [B, 1, C, H, W]
        masks_expanded = normalized_masks.unsqueeze(2)  # [B, num_masks, 1, H, W]
        pooled = (features_expanded * masks_expanded).sum(dim=(3, 4))  # [B, num_masks, C]
        return pooled
    
    def _distributed_sinkhorn(self, logits, n_iters=3):
       
        Q = logits.softmax(dim=-1).T  # [K, N]
        K = logits.shape[1]  # 
        
        sum_Q = torch.sum(Q)
        sum_Q = torch.clamp(sum_Q, min=1e-6)
        B = sum_Q
        Q = Q / sum_Q
        
        for _ in range(n_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            sum_of_rows = torch.clamp(sum_of_rows, min=1e-6)
            Q = Q / sum_of_rows
            Q = Q / K
            
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            sum_of_cols = torch.clamp(sum_of_cols, min=1e-6)
            Q = Q / sum_of_cols
            Q = Q / B
        
        Q = Q * B
        
        return Q.t()  # [N, K]
    
    def masked_center(self, features, masks, use_center_only=True, min_mask_area=20, 
                      erosion_kernel_size=3, erosion_iterations=1):
        """
        """
        B, C, H, W = features.shape  
        num_masks = masks.shape[1]  
        if use_center_only:
            masks = self.erode_mask_to_center(masks, min_mask_area, 
                                             erosion_kernel_size, erosion_iterations)
        
        masks = masks.to(device=features.device, dtype=features.dtype)
        
        y_coords = torch.arange(H, dtype=features.dtype, device=features.device).view(H, 1).expand(H, W)
        x_coords = torch.arange(W, dtype=features.dtype, device=features.device).view(1, W).expand(H, W)
        
        pooled = torch.zeros(B, num_masks, C, device=features.device, dtype=features.dtype)
        
        for b in range(B):
            for m in range(num_masks):
                mask = masks[b, m]  # [H, W]
                mask_sum = mask.sum()
                
                if mask_sum < 1e-6:
                    continue
                
                center_y = (mask * y_coords).sum() / mask_sum
                center_x = (mask * x_coords).sum() / mask_sum
                
                cy = int(torch.round(center_y).item())
                cx = int(torch.round(center_x).item())
                
                cy = max(0, min(cy, H - 1))
                cx = max(0, min(cx, W - 1))
                
                pooled[b, m] = features[b, :, cy, cx]
        
        return pooled  # [B, num_masks, C]
            

