import time
import copy
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from ovss import load_ovss

from torchvision import transforms
from utils_local.logits_sam import SAMwithlogits         ##sam1
import numpy as np

CLS_EOS_ID = 49407
REFERENCE_PROMPT = 'a photo of a {}'


class TPT(nn.Module):
    """
    Test-Time Prompt Tuning (TPT) for open-vocabulary semantic segmentation (OVSS) models.

    Inspired by the official TPT repository (https://github.com/azshue/TPT), this module
    injects a small set of learnable soft prompt tokens into the CLIP text encoder and
    adapts them per test image via entropy minimization, keeping the backbone frozen.
    """

    def __init__(self, ovss_type, ovss_backbone, classes, lr=5e-3, n_ctx=4, steps=1, 
                 runtime_calculation=False, device= "cuda",
                 ):
        """
        Initialize the TPT adaptation module.

        Args:
            ovss_type (str): Identifier for the OVSS model to load.
            ovss_backbone (str): Name of the backbone architecture.
            classes (List[str]): Ordered list of class names for segmentation.
            lr (float, optional): Learning rate for soft-prompt optimizer. Defaults to 5e-3.
            n_ctx (int, optional): Number of learnable prompt tokens. Defaults to 4.
            steps (int, optional): Adaptation steps per test sample. Defaults to 1.
            runtime_calculation (bool, optional): Record runtimes if True. Defaults to False.
            device (str, optional): Compute device ("cpu" or "cuda"). Defaults to "cuda".
        """
        super().__init__()

        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr

        if classes is not None:
            self.classes = classes
        else:
            raise Exception("Classes are required in the init")
        
        assert n_ctx == 4, "The default hand‑crafted prompt has exactly 4 tokens. If you change n_ctx, be sure the prompt template length matches."
        self.n_ctx = n_ctx

        self.steps = steps
        self.prompt = REFERENCE_PROMPT
        self.runtime = runtime_calculation
        self.device = device

        # ---------- OVSS Model ----------
        self.model, self.tokenize = load_ovss(self.ovss_type, self.ovss_backbone, device=self.device)

        # ---------- learnable soft tokens (initialised from "a photo of a") ----------
        with torch.no_grad():
            tmpl_ids = self.tokenize("a photo of a")[0]            # (77,)
            tmpl_ids = tmpl_ids.to(device)
            ctx_ids = tmpl_ids[1:1 + n_ctx]                        # remove <SOS>, take next 4 ids
            init_ctx = self.model.token_embedding(ctx_ids).clone()  # 4 × D (D = 512/768)
        self.ctx = nn.Parameter(init_ctx.to(device), requires_grad=True)
        self.D = self.ctx.shape[1]

        # ---------- optimiser ----------
        self.optimizer = optim.Adam([self.ctx], lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        # ---------- save pristine state for reset() ----------
        self._clean_state = {
            "model": copy.deepcopy(self.model.state_dict()),
            "ctx": self.ctx.data.clone(),
            "optim": copy.deepcopy(self.optimizer.state_dict()),
        }

        # ---------- pre‑tokenise every class once ----------
        self._token_ids = {c: self.tokenize(self.prompt.format(c))[0] for c in self.classes}

        # ---------- define variables to store adaptation and evaluation duration ----------
        if self.runtime:
            self.adapt_times = []
            self.eval_times = []
        
        self.use_sam = False    ##是否使用sam生成mask
        self.voted = False      ##是否对logits进行平均（相当于maskpooling）
        self.clsvoted = False   ##对logits实现基于类别的投票
        self.logits2prompt = True  ##是否使用logits生成prompt再生成mask
        self.logits2prompt_iterations = 2  ##logits优化的迭代次数
        self.random_mask = False  ##是否使用随机mask替代从logits生成的mask方法
        self.Sinkhorn_clustered = True  ##是否使用Sinkhorn对logits进行聚类
        self.prompt_type = 'point'  # 提示类型: 'point' 或 'box'
        self.logits2sam = SAMwithlogits(
                num_sample_points=10,
                response_thresh_ratio=0.5,
                min_prob_thresh=0.1,
                min_mask_area=100,
                prompt_type=self.prompt_type,
                enable_enhanced=True,
                use_semantic=True,       # 启用语义一致性评分
                alpha=0.3,               # 语义得分权重 30%
                beta=0.7,                 # 几何得分权重 70%
                Sinkhorn_clustered = self.Sinkhorn_clustered,
            )
        self.patch_size = 14


    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================
    @lru_cache(maxsize=None)
    def _sos_rest_eos(self, cls_name: str):
        """Return (<SOS> ids, rest ids, eos_position_in_original_sentence)."""
        ids = self._token_ids[cls_name].clone()
        eos_idx = (ids == CLS_EOS_ID).nonzero(as_tuple=True)[0].item()
        sos = ids[:1]       # keep the first token (SOS)
        rest = ids[1:]      # rest incl. <EOS> & <PAD>
        return sos, rest, eos_idx

    def _build_text_features(self):
        """Encode class names using current prompt ➜ (C, D_proj)."""
        C = len(self.classes)
        device = self.device

        sos_list, body_list, eos_positions = [], [], []
        for cls in self.classes:
            sos, body, eos_idx = self._sos_rest_eos(cls)
            # Drop the first n_ctx tokens of the body (they were replaced)
            body_trimmed = body[self.n_ctx:]
            sos_list.append(sos)
            body_list.append(body_trimmed)
            eos_positions.append(eos_idx)  # position unchanged because we replaced, not inserted

        ids_sos = torch.stack(sos_list).to(device)             # C × 1
        ids_body = torch.stack(body_list).to(device)           # C × (76‑4) = 72

        emb_sos = self.model.token_embedding(ids_sos)           # C × 1 × D
        emb_body = self.model.token_embedding(ids_body)         # C × 72 × D

        # Broadcast soft prompt to all classes
        ctx = self.ctx.unsqueeze(0).expand(C, -1, -1)          # C × 4 × D

        # Concatenate: seq length = 1 + 4 + 72 = 77
        x = torch.cat([emb_sos, ctx, emb_body], dim=1)         # C × 77 × D

        # Add positional encodings (77 tokens available)
        x = x + self.model.positional_embedding

        # Transformer expects (seq, batch, dim)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x.half())
        x = x.permute(1, 0, 2)                                # C × 77 × D

        eos_idx_tensor = torch.tensor(eos_positions, device=device)
        x = x[torch.arange(C, device=device), eos_idx_tensor]  # C × D
        x = self.model.ln_final(x)
        x = x @ self.model.text_projection
        x = x / x.norm(dim=-1, keepdim=True)
        return x


    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def reset(self):
        """Restore backbone, prompt, and optimiser to their initial states."""
        self.model.load_state_dict(self._clean_state["model"], strict=True)
        self.ctx.data.copy_(self._clean_state["ctx"])
        self.optimizer.load_state_dict(self._clean_state["optim"])

    @torch.no_grad()
    def evaluate(self, images):
        """
        Forward pass without adaptation.

        Args:
            images (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Per-class logits of shape (batch_size, num_classes, H, W).

        """
        t0 = time.time()
        images = images.to(self.device)
        img_feat = self.model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = self._build_text_features()           # C×D
        logits = self.model.logit_scale.exp() * img_feat @ txt_feat.T

        logits = logits[:, 1:]
        patch_size = self.model.visual.patch_size
        w, h = images[0].shape[-2] // patch_size, images[0].shape[-1] // patch_size
        b_dim = logits.shape[0]
        out_dim = logits.shape[-1]
        
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
        logits = logits.unsqueeze(0)  # add template dim
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        logits = self.updatelogits(logits, images, txt_feat, image_features=img_feat)
        logits = self.interpolate_logits(logits, images)
        logits = logits[0]  # remove template dim

        # # interpolate to original image size
        # logits = nn.functional.interpolate(logits, size=(images[0].shape[-2], images[0].shape[-1]), mode='bilinear', align_corners=False)

        if self.runtime:
            self.eval_times.append(time.time() - t0)

        return logits                   

    def adapt(self, images):
        """
        Forward pass with adaptation.

        Args:
            images (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            List[float]: Loss values recorded at each adaptation iteration.
        """
        
        self.reset()
        loss_report = []
        images = images.to(self.device)
        t0 = time.time()
        for _ in range(self.steps):
            img_feat = self.model.encode_image(images)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = self._build_text_features()
            logits = self.model.logit_scale.exp() * img_feat @ txt_feat.T



            logits = logits[:, 1:]
            patch_size = self.model.visual.patch_size
            w, h = images[0].shape[-2] // patch_size, images[0].shape[-1] // patch_size
            b_dim = logits.shape[0]
            out_dim = logits.shape[-1]
            
            logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)



            loss = self.softmax_entropy(logits).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.runtime:
            self.adapt_times.append(time.time() - t0)

        return loss_report


    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim=-3) -> torch.Tensor:
        """Entropy of softmax distribution from logits.
            x : torch.Tensor : logits of shape (#templates, batch_size, num_classes, H, W)
        """
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)
    
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
        # 准备语义评分所需的特征（如果启用）
        # 1. 文本特征：取多模板的平均作为每个类别的代表
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.unsqueeze(0) if text_features.dim() == 2 else text_features  # (T, C, D) 或 (C, D)
        text_embeddings_for_scoring = text_features.mean(dim=0) if text_features.dim() == 3 else text_features  # (#classes, D)
        
        # 2. 视觉特征：将token特征转换为空间特征图
        spatial_features_batch = []
        if self.logits2sam.use_semantic:
            for b_idx in range(batch_size):
                # 使用已有的 _convert_tokens_to_spatial_features 方法
                spatial_feat = self._convert_tokens_to_spatial_features(image_features, batch_idx=b_idx)
                spatial_features_batch.append(spatial_feat)
        else:
            spatial_features_batch = None
        
        # 迭代优化过程
        for iteration in range(self.logits2prompt_iterations):
            # 根据prompt_type选择相应的生成方法
            if self.prompt_type == 'box':
                batch_prompts = self.logits2sam.generate_box_prompts(logits, original_img_shape=image.shape[-2:])
            elif self.prompt_type == 'point':
                batch_prompts = self.logits2sam.generate_prompts(logits, original_img_shape=image.shape[-2:])
            else:
                raise ValueError(f"Unknown prompt_type: {self.prompt_type}. Must be 'point' or 'box'.")
            
            # 调用get_masks，传递预处理好的空间特征（避免冗余计算）
            masks  = self.logits2sam.get_masks(
                image, 
                batch_prompts, 
                logits,
                text_embeddings=text_embeddings_for_scoring,  # 传递文本嵌入
                spatial_features=spatial_features_batch       # 传递空间特征图
            )
            # if masks is not None:
            #     _, num_masks, mask_h, mask_w = masks.shape
            #     resize_transform = transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)##对于VITL14
            #     masks = resize_transform(masks).to(image.dtype)##torch.Size([b, m, 16, 16])
            #     # 对每个 batch 分别过滤空 mask
            #     batch_masks_filtered = []
            #     for b in range(masks.shape[0]):
            #         mask_no_empty_ = []
            #         for m in range(masks.shape[1]):
            #             mask = masks[b][m]
            #             if mask.sum() != 0:
            #                 mask_no_empty_.append(mask)
            #         # 在每个 batch 内部进行判断并追加结果（修复缩进导致只处理最后一个 batch 的问题）
            #         if len(mask_no_empty_) > 0:
            #             # 该 batch 有非空 mask
            #             batch_masks_filtered.append(torch.stack(mask_no_empty_, dim=0))
            #         else:
            #             # 该 batch 没有非空 mask，添加一个空的占位 mask
            #             batch_masks_filtered.append(torch.zeros((1, 16, 16), dtype=masks.dtype, device=masks.device))

            #     # 找到最大的 mask 数量并 padding
            #     max_num_masks = max(m.shape[0] for m in batch_masks_filtered)
            #     padded_batch_masks = []
            #     for batch_mask in batch_masks_filtered:
            #         num_m = batch_mask.shape[0]
            #         if num_m < max_num_masks:
            #             # 需要 padding
            #             padding = torch.zeros((max_num_masks - num_m, 16, 16), dtype=batch_mask.dtype, device=batch_mask.device)
            #             padded_mask = torch.cat([batch_mask, padding], dim=0)
            #             padded_batch_masks.append(padded_mask)
            #         else:
            #             padded_batch_masks.append(batch_mask)

            #     # 堆叠所有 batch 的 masks
            #     masks = torch.stack(padded_batch_masks, dim=0)
                
            #     # 检查是否所有 batch 都没有有效 mask
            #     if max_num_masks == 1 and all(m[0].sum() == 0 for m in batch_masks_filtered):
            #         masks = None
            if masks is not None:
                _, num_masks, mask_h, mask_w = masks.shape
                resize_transform = transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)
                masks = resize_transform(masks).to(image.dtype)##torch.Size([b, m, 16, 16])
                mask_no_empty_ = []
                for m in range(masks.shape[1]):
                    mask = masks[0][m]
                    if mask.sum() !=0:
                            mask_no_empty_.append(mask)
                if len(mask_no_empty_)==0:
                    masks = None
                    # print(f"Iteration {iteration+1}/{self.logits2prompt_iterations}: No masks generated for image.")
                else:
                    masks = torch.stack( mask_no_empty_, dim=0).unsqueeze(0) # 直接使用torch.stack拼接tensor
                if masks is not None:
                    _, num_masks, mask_h, mask_w = masks.shape
                    output_logits = logits.clone()
                    # 对每个 mask 区域内的 logits 进行聚合
                    # 倒序处理：排名靠前的mask得分更高，最后处理以覆盖低分mask的结果
                    for b in range(batch_size):
                        for m in range(num_masks - 1, -1, -1):  # 倒序遍历
                        # for m in range(num_masks):  # 正序遍历
                            mask_binary = masks[b, m] > 0  # (h, w)
                            if mask_binary.sum() > 0:  # 确保 mask 不为空
                                # 提取该 mask 内的所有 logits 进行投票
                                # logits: (templates, batch_size, classes, h, w)
                                # mask_binary: (h, w)
                                mask_logits = logits[:, b, :, mask_binary]  # (templates, classes, num_pixels_in_mask)
                                
                                # 投票：对 mask 内的所有像素取平均值（也可以用 max 或其他聚合方式）
                                voted_logits = mask_logits.mean(dim=-1, keepdim=True)  # (templates, classes, 1)
                                # 将投票结果赋给该 mask 内的所有位置，使其保持一致
                                output_logits[:, b, :, mask_binary] = voted_logits
                    if self.Sinkhorn_clustered:
                        # ========== 基于 Sinkhorn 聚类的 logits 计算 ==========
                        # 1. 获取 mask 池化特征: [B, num_masks, D]
                        image_features_masked = self.masked_pooling(
                            image_features[:,1:,].permute(0,2,1).reshape(
                                image_features.shape[0], image_features.shape[2], 16, 16
                            ), 
                            masks
                        )
                        
                        # # 2. 归一化 mask 特征
                        # image_features_masked_norm = F.normalize(image_features_masked.float(), dim=-1)  # [B, num_masks, D]
                        image_features_masked_sim = image_features_masked.float()
                        # 3. 获取维度信息
                        num_templates_cluster = text_features.shape[0]
                        num_classes_cluster = text_features.shape[1]
                        B_cluster = image_features_masked.shape[0]
                        num_masks_cluster = image_features_masked.shape[1]
                        
                        # 4. 对每个模板计算聚类分配并映射回空间
                        logits_clustered_list = []
                        
                        for t_idx in range(num_templates_cluster):
                            # 归一化当前模板的文本特征: [num_classes, D]
                            # text_features_norm = F.normalize(text_features[t_idx].float(), dim=-1)
                            text_features_sim =  text_features[t_idx].float()
                            # 计算相似度: [B, num_masks, D] @ [D, num_classes] -> [B, num_masks, num_classes]
                            similarity_logits = image_features_masked_sim @ text_features_sim.T
                            
                            # 展平后做 Sinkhorn 聚类
                            similarity_flat = similarity_logits.reshape(-1, num_classes_cluster)  # [B*num_masks, num_classes]
                            
                            # Sinkhorn 软分配算法
                            assignments = self._distributed_sinkhorn(similarity_flat, n_iters=3)  # [B*num_masks, num_classes]
                            assignments = assignments.reshape(B_cluster, num_masks_cluster, num_classes_cluster)  # [B, num_masks, num_classes]
                            
                            # 使用 einsum 将分配映射回空间维度
                            # assignments: [B, num_masks, num_classes], masks: [B, num_masks, H, W]
                            # 输出: [B, num_classes, H, W] - 每个像素对每个类别的软分数
                            spatial_logits = torch.einsum('bmc,bmhw->bchw', assignments.to(masks.device), masks.float()).to("cuda")
                            logits_clustered_list.append(spatial_logits)
                        
                        # 5. 堆叠所有模板的 logits: [T, B, num_classes, H, W]
                        logits_clustered = torch.stack(logits_clustered_list, dim=0)
                        logits_clustered = logit_scale * logits_clustered
                    
                        # 更新logits用于下一轮迭代（融合投票 logits 和聚类 logits）
                        logits =  logits + 0.5 * (output_logits + logits_clustered) 
                    else:
                        logits = output_logits
                    # print(f"Iteration {iteration+1}/{self.logits2prompt_iterations}: Logits optimized with {num_masks} masks.")
                else:
                    # 如果没有生成mask，则跳出迭代
                    break
            else:
                # 如果没有生成mask，则跳出迭代
                # print(f"Iteration {iteration+1}/{self.logits2prompt_iterations}: No masks generated, stopping iteration.")
                break
        return logits
    
    def _convert_tokens_to_spatial_features(self, image_features, batch_idx=0):
        """
        将 token 序列转换为空间特征图，用于语义一致性计算
        
        Args:
            image_features: torch.Tensor
                - 如果是3D: (batch_size, tokens, D)
                - 如果是4D: (outlayers, batch_size, tokens, D)
            batch_idx: 要提取的batch索引
            
        Returns:
            spatial_feature_map: torch.Tensor (feat_h, feat_w, D)
        """
        # 处理不同维度的情况
        if image_features.dim() == 4:
            # (outlayers, batch_size, tokens, D) -> 取最后一层
            features = image_features[-1, batch_idx]  # (tokens, D)
        elif image_features.dim() == 3:
            # (batch_size, tokens, D)
            features = image_features[batch_idx]  # (tokens, D)
        else:
            raise ValueError(f"Unexpected image_features dimension: {image_features.dim()}")
        
        # 去除CLS token
        spatial_tokens = features[1:, :]  # (N_patches, D)
    
        # 计算网格尺寸并重塑为空间特征图
        n_patches = spatial_tokens.shape[0]
        grid_size = int(n_patches ** 0.5)
        
        if grid_size * grid_size != n_patches:
            # 如果不是完美平方数，尝试根据 patch_size 计算
            raise ValueError(f"Cannot reshape {n_patches} patches to square grid")
        
        # 重塑为 (H, W, D)
        spatial_feature_map = spatial_tokens.reshape(grid_size, grid_size, -1)
        
        return spatial_feature_map
          
    def erode_mask_to_center(self, masks, min_mask_area=20, erosion_kernel_size=3, erosion_iterations=1):
        """
        对掩码进行腐蚀操作，只保留中心区域，去除边缘部分
        
        参数:
            masks: [B, num_masks, H, W] 掩码张量，值在0-1之间
            min_mask_area: 最小掩码面积阈值（像素数），小于该值的掩码不进行腐蚀
            erosion_kernel_size: 腐蚀核的大小，必须是奇数（3, 5, 7等）
            erosion_iterations: 腐蚀迭代次数
        
        返回:
            处理后的掩码 [B, num_masks, H, W]
        """
        B, num_masks, H, W = masks.shape
        processed_masks = []
        
        for b in range(B):
            batch_processed = []
            for m in range(num_masks):
                mask = masks[b, m]  # [H, W]
                
                # 计算掩码面积（对于0-1之间的值，使用阈值0.5来计算有效面积）
                mask_binary = (mask > 0.5).float()
                mask_area = mask_binary.sum().item()
                
                # 如果掩码太小，不进行腐蚀，直接使用原始掩码
                if mask_area < min_mask_area:
                    batch_processed.append(mask)
                    continue
                
                # 对掩码进行腐蚀操作
                eroded_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # 创建腐蚀核（全1的卷积核）
                kernel = torch.ones(1, 1, erosion_kernel_size, erosion_kernel_size, 
                                   device=masks.device, dtype=masks.dtype)
                padding = erosion_kernel_size // 2
                
                # 多次腐蚀迭代
                for _ in range(erosion_iterations):
                    # 使用卷积模拟腐蚀操作
                    # 腐蚀：只有当核覆盖区域全为1时，中心点才为1
                    # 这里使用平均池化的反向逻辑：如果平均值小于1，说明有边缘
                    conv_result = F.conv2d(eroded_mask, kernel, padding=padding)
                    
                    # 对于0-1之间的值，使用阈值来判断是否保留
                    # 只有当邻域内的平均值接近1时才保留（即都是前景）
                    threshold = kernel.sum() * 0.8  # 80%的核区域为前景时才保留
                    eroded_mask = (conv_result >= threshold).float()
                
                eroded_mask = eroded_mask.squeeze(0).squeeze(0)  # [H, W]
                
                # 检查腐蚀后是否还有剩余区域
                if eroded_mask.sum() < 1e-6:
                    # 如果腐蚀后没有剩余，使用原始掩码
                    batch_processed.append(mask)
                else:
                    # 使用腐蚀后的掩码
                    batch_processed.append(eroded_mask)
            
            processed_masks.append(torch.stack(batch_processed, dim=0))
        
        return torch.stack(processed_masks, dim=0)  # [B, num_masks, H, W]
    def masked_pooling(self, features, masks, use_center_only=True, min_mask_area=20, 
                      erosion_kernel_size=3, erosion_iterations=1):
        B, C, H, W = features.shape##特征图，B是batch size，C是通道数，H,W是空间尺寸。
        num_masks = masks.shape[1]##num_masks 个mask，每个mask覆盖 H×W，通常是0/1或[0,1]的权重
        # 如果启用中心区域优化，先对掩码进行腐蚀处理
        if use_center_only:
            masks = self.erode_mask_to_center(masks, min_mask_area, 
                                             erosion_kernel_size, erosion_iterations)
        masks = masks.to(device = features.device, dtype = features.dtype)
        mask_sums = masks.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)  # [B, num_masks, 1, 1]
        #对每个mask在空间维度 (H, W) 上求和，相当于算每个mask的面积（或总权重），得到形状 [B, num_masks]，加上 keepdim=True 变成 [B, num_masks, 1, 1]。
        normalized_masks = masks / mask_sums  # [B, num_masks, H, W]
        #每个mask变成一个在空间上加和为1的权重分布
        #把原来的二值/实值mask变成归一化的加权系数，后面对特征做的是“加权平均”，而不是简单求和。
        features_expanded = features.unsqueeze(1)  # [B, 1, C, H, W]
        masks_expanded = normalized_masks.unsqueeze(2)  # [B, num_masks, 1, H, W]
        #为了广播乘法，对维度做扩展
    
        pooled = (features_expanded * masks_expanded).sum(dim=(3, 4))  # [B, num_masks, C]
        ##按 mask 在空间上做加权平均池化
        ##features = [1,2;3,4]
        ##mask1 = [1,0;0,1] mask2 = [0,1;1,0]
        ##norm_mask1 = [0.5,0;0,0.5] norm_mask2 = [0,0.5;0.5,0]
        ##masked_pooled = [0.5*1+0.5*3,0.5*2+0.5*3] = [2.5,2.5]
    
        return pooled
    
    def _distributed_sinkhorn(self, logits, n_iters=3):
        """
        Sinkhorn-Knopp 软聚类分配算法
        
        参数:
            logits: [N, K] 相似度矩阵, N 是样本数, K 是类别数
            n_iters: Sinkhorn 迭代次数
            
        返回:
            Q: [N, K] 软分配矩阵, 每行和为1
        """
        # 转换为概率分布
        Q = logits.softmax(dim=-1).T  # [K, N]
        K = logits.shape[1]  # 类别数量
        
        # 使矩阵和为1
        sum_Q = torch.sum(Q)
        sum_Q = torch.clamp(sum_Q, min=1e-6)
        B = sum_Q
        Q = Q / sum_Q
        
        for _ in range(n_iters):
            # 行归一化：每个原型的总权重必须为 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            sum_of_rows = torch.clamp(sum_of_rows, min=1e-6)
            Q = Q / sum_of_rows
            Q = Q / K
            
            # 列归一化：每个样本的总权重必须为 1/B
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            sum_of_cols = torch.clamp(sum_of_cols, min=1e-6)
            Q = Q / sum_of_cols
            Q = Q / B
        
        # 使列和为1，使 Q 成为有效分配矩阵
        Q = Q * B
        
        return Q.t()  # [N, K]
    
    def masked_center(self, features, masks, use_center_only=True, min_mask_area=20, 
                      erosion_kernel_size=3, erosion_iterations=1):
        """
        直接提取mask中心点对应的特征，而不是对整个mask区域做平均池化
        """
        B, C, H, W = features.shape  # 特征图：B batch size, C 通道数, H,W 空间尺寸
        num_masks = masks.shape[1]  # num_masks 个mask
        if use_center_only:
            masks = self.erode_mask_to_center(masks, min_mask_area, 
                                             erosion_kernel_size, erosion_iterations)
        
        masks = masks.to(device=features.device, dtype=features.dtype)
        
        # 创建坐标网格
        y_coords = torch.arange(H, dtype=features.dtype, device=features.device).view(H, 1).expand(H, W)
        x_coords = torch.arange(W, dtype=features.dtype, device=features.device).view(1, W).expand(H, W)
        
        # 初始化输出tensor
        pooled = torch.zeros(B, num_masks, C, device=features.device, dtype=features.dtype)
        
        # 对每个batch和每个mask计算质心并提取特征
        for b in range(B):
            for m in range(num_masks):
                mask = masks[b, m]  # [H, W]
                mask_sum = mask.sum()
                
                if mask_sum < 1e-6:
                    # 如果mask为空，使用零向量
                    continue
                
                # 计算质心坐标（加权平均）
                center_y = (mask * y_coords).sum() / mask_sum
                center_x = (mask * x_coords).sum() / mask_sum
                
                # 转换为整数索引（四舍五入）
                cy = int(torch.round(center_y).item())
                cx = int(torch.round(center_x).item())
                
                # 确保索引在有效范围内
                cy = max(0, min(cy, H - 1))
                cx = max(0, min(cx, W - 1))
                
                # 直接提取中心点特征
                pooled[b, m] = features[b, :, cy, cx]
        
        return pooled  # [B, num_masks, C]
            
