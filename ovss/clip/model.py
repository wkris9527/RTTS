from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.cluster import KMeans
from PIL import Image
from timm.models.vision_transformer import PatchEmbed
from torchvision import transforms
import torch.nn.functional as F
# from utils_local.sam import SamMask
# from utils_local.dino import Dino
# from utils_local.vit import Extractor
from utils_local.logits_sam import SAMwithlogits         ##sam1
# from utils_local.logits_samcity import SAMwithlogits
# from utils_local.logits_sam2 import SAMwithlogits         ##sam2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):####修改的地方
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size#14
        self.output_dim = output_dim#768
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.arch, self.attn_strategy, self.gaussian_std = None, None, 0
        self.addition_cache = dict()
        self.purge = False
        # self.dino_patch_size = 16
        # self.dino_cluster = 16
        # self.dino = Dino()
        # self.vit = Extractor()
        # self.sammask = SamMask()

    # nonly: Neighbourhood Only, kk: KK-Similarity, csa: SCLIP, vanilla: CLIP
    def set_params(self, arch, attn_strategy, gaussian_std):
        assert arch in ['reduced', 'vanilla']
        assert attn_strategy in ['naclip', 'nonly', 'kk', 'csa', 'vanilla']
        assert attn_strategy != 'csa' or arch == 'vanilla'
        assert gaussian_std > 0 or attn_strategy not in ['naclip', 'nonly']
        self.arch, self.attn_strategy, self.gaussian_std = arch, attn_strategy, gaussian_std

    def forward(self, x: torch.Tensor, output_layers=(-1,), out_type="mean", return_vanilla_cls=False, weights=None,mask_list=None):
        # import pdb; pdb.set_trace()
        B, nc, w, h = x.shape  ##torch.Size([2, 3, 224, 224])
        # mask_list = self.dino.get_dino_masks(x, n_clusters=16)
        masks = None

        if mask_list is not None:
            masks = mask_list
            if masks is not None and masks.dtype != x.dtype:
                masks = masks.to(x.dtype)
        
   
        n_patches = (w // self.patch_size, h // self.patch_size)##(16,16)

        x = self.conv1(x) 
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        

        # import pdb; pdb.set_trace()
        if self.purge:
            if x.shape[1] != self.positional_embedding.shape[0]:
                pos = self.interpolate_pos_encoding(x, w, h).to(x.dtype)
            else:
                pos = self.positional_embedding.to(x.dtype)
            pos = pos.unsqueeze(0).expand(x.shape[0],-1,-1)
        else:
            if x.shape[1] != self.positional_embedding.shape[0]:
                x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
            else:
                x = x + self.positional_embedding.to(x.dtype)
        # import pdb; pdb.set_trace()


      
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND ##torch.Size([257, 2, 1024])

        # convert negative indices to positive if applicable
        num_layers = len(self.transformer.resblocks) ##24       
        output_layers = tuple(num_layers + idx if idx < 0 else idx for idx in output_layers) #(23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6)
        last_layer_idx = max(output_layers)

        out_features = [] 
        for idx, blk in enumerate(self.transformer.resblocks[:last_layer_idx+1]):
            blk = self.transformer.resblocks[idx]
            if self.purge and idx in [0]:
                x = self.cls_purge_forward(blk.attn, x, pos=pos, purge_size =16,blk=blk)
            if idx != last_layer_idx:
                # we always use the vanilla attention for the intermediate layers beacuse in one experiment if we feed any other attention the performance of the next layers drops
                reduced = self.custom_attn("vanilla", blk.attn, blk.ln_1(x), n_patches) ## for the purpose of evaluation we can compute 2 types, one for feedforward and one for the evaluation
                x = x + reduced
                x = x + blk.mlp(blk.ln_2(x))

            else: 
                reduced = self.custom_attn(self.attn_strategy, blk.attn, blk.ln_1(x), n_patches)
                final_x = x + reduced
                final_x = final_x + blk.mlp(blk.ln_2(final_x))
                if self.attn_strategy != 'vanilla' and return_vanilla_cls:
                    vanilla_cls = blk(x)[0]

                    

            if idx in output_layers:
                if self.arch == 'reduced':
                    out_features.append(reduced)
                elif self.arch == 'vanilla':
                    if idx == last_layer_idx:
                        out_features.append(final_x)
                else:
                    raise NotImplemented(f'arch {self.arch} is not implemented')
                
        if out_type=="mean":
            if len(out_features) > 1:
                x = torch.mean(torch.stack(out_features), dim=0)
            else:
                x = out_features[0]


            x = x.permute(1, 0, 2)  # LND -> NLD

            if return_vanilla_cls:
                return self.ln_post(x) @ self.proj, self.ln_post(vanilla_cls) @ self.proj
            else:
                return self.ln_post(x) @ self.proj

        elif out_type=="all":
            out_features = [x.permute(1, 0, 2) for x in out_features]
            out_features = [self.ln_post(x) @ self.proj for x in out_features]
            # stack out_features on a new dim
            out_features = torch.stack(out_features, dim=0)
            if return_vanilla_cls:
                return out_features, self.ln_post(vanilla_cls) @ self.proj
            else:
                return out_features
        
        elif out_type=="weighted_mean":
            len(weights) == len(out_features)
            
            # Compute the final output as the weighted average of the selected reduced features
            x = torch.stack(out_features, dim=0)
            x = torch.sum(x * weights.unsqueeze(1).unsqueeze(-1), dim=0)
            x = x.permute(1, 0, 2)  # LND -> NLD


            return self.ln_post(x) @ self.proj




    @staticmethod
    def gaussian_window(dim1, dim2, std=1.):
        constant = 1 / (std * math.sqrt(2))
        ks = list()
        for dim in [dim1, dim2]:
            start = -(dim - 1) / 2.0
            k = torch.linspace(start=start * constant,
                               end=(start + (dim - 1)) * constant,
                               steps=dim,
                               dtype=torch.float)
            ks.append(k)
        dist_square_to_mu = (torch.stack(torch.meshgrid(*ks, indexing='ij')) ** 2).sum(0)
        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window, adjust_for_cls=True):
        m = torch.einsum('ij,kl->ijkl', torch.eye(dim1), torch.eye(dim2))
        m = m.permute((0, 3, 1, 2)).contiguous()  # m[ijkl] = 1 iff (i, j) == (k, l)
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1), window.unsqueeze(0).unsqueeze(1), padding='same').squeeze(1)
        out = out.view(dim1 * dim2, dim1 * dim2)
        if adjust_for_cls:
            v_adjusted = torch.vstack([torch.zeros((1, dim1 * dim2)), out])
            out = torch.hstack([torch.zeros((dim1 * dim2 + 1, 1)), v_adjusted])
        return out

    def custom_attn(self, attn_strategy, attn_layer, x, n_patches, return_attn=False, with_attn=False):

        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5
        if x.dtype != attn_layer.in_proj_weight.dtype:
            x = x.to(attn_layer.in_proj_weight.dtype)

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if attn_strategy in ['naclip', 'nonly']:
            addition = self.addition_cache.get(n_patches)
            if addition is None:
                window_size = [side * 2 - 1 for side in n_patches]
                window = VisionTransformer.gaussian_window(*window_size, std=self.gaussian_std)
                addition = VisionTransformer.get_attention_addition(*n_patches, window).unsqueeze(0).to(x.dtype).to(
                    x.device)
                self.addition_cache[n_patches] = addition

            if attn_strategy == 'naclip':
                attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale
                omega = addition
            elif attn_strategy == 'nonly':
                attn_weights = torch.zeros((num_heads, num_tokens, num_tokens)).to(x.dtype).to(x.device)
                omega = addition * (scale * torch.einsum('hop,hPO->hpP', q.norm(dim=2).unsqueeze(1),
                                                         k.norm(dim=2).unsqueeze(2)).mean().item())
            else:
                raise NotImplemented

            attn_weights += omega
            attn_weights = F.softmax(attn_weights, dim=-1)

        elif attn_strategy == 'csa':
            q_attn = torch.bmm(q, q.transpose(1, 2)) * scale
            k_attn = torch.bmm(k, k.transpose(1, 2)) * scale
            attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)
        elif attn_strategy == 'vanilla':
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif attn_strategy == 'kk':
            attn_weights = torch.bmm(k * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            raise NotImplemented(f'attn_strategy {self.attn_strategy} is not implemented')

        if return_attn:
            return attn_weights

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)

        if with_attn:
            return attn_output, attn_weights

        return attn_output

    def erode_mask_to_center(self, masks, min_mask_area=20, erosion_kernel_size=3, erosion_iterations=1):
  
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
    
    def masked_center(self, features, masks, use_center_only=True, min_mask_area=20, 
                      erosion_kernel_size=3, erosion_iterations=1):
     
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
        
        return pooled  
    def cls_purge_forward(self, attn_layer,x,pos=None,purge_size=3,blk = None):
        # import pdb; pdb.set_trace()
        x = x.permute(1, 0, 2) 
        B, N, C = x.shape
        cls_token = x[:, 0:1]  
        pos_cls = pos[:, 0:1]  
        cls = blk.ln_1(cls_token + pos_cls)
        cls_query =  (
            F.linear(cls, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        )[0]
        token = x[:, 1:]  # B N-1 C
        pos_token = pos[:,1:]
        patches = blk.ln_1(token + pos_token)
        token_key = (
            F.linear(patches, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        )[1]
        cosine_distance = self.cosine_distance(cls_query, token_key)
        to_keep_indices = cosine_distance.argsort(dim=-1)[
                    :, : N - 1 - purge_size
        ]
        to_keep_indices = to_keep_indices.sort(dim=-1)[0]
        to_keep_indices = to_keep_indices.unsqueeze(-1).expand(-1, -1, C)

        masked_x = torch.gather(x[:, 1:], 1, to_keep_indices)
        masked_pose = torch.gather(pos[:, 1:], 1, to_keep_indices)
        x = torch.cat([x[:, 0:1], masked_x], dim=1)
        pos = torch.cat([pos[:, 0:1], masked_pose], dim=1)
        x = x + pos
        x = x.permute(1, 0, 2) 

        return x
    def cosine_distance(self,tokens, mean, std=0):
        tokens_norm = F.normalize(tokens, p=2, dim=-1)  
        mean_norm = F.normalize(mean, p=2, dim=-1)  
        cosine_sim = (tokens_norm * mean_norm).sum(dim=-1)  
        cosine_dist = 1 - cosine_sim  
        return cosine_dist
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), mode='bicubic',
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length
        self.use_sam = False    
        self.voted = False      
        self.clsvoted = False   
        self.logits2prompt = True  
        self.auto_mask = False  
        self.logits2prompt_iterations = 2 
        self.random_mask = False  
        self.Sinkhorn_clustered = True 

        self.prompt_type = 'point' 
        if self.use_sam:
            self.sammask = SamMask()
        if self.logits2prompt or self.auto_mask:
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
        if self.random_mask:
            self.sammask = SamMask()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.weights_track = []
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

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

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, output_layers=(-1,), return_vanilla_cls=False, out_type="mean", weights=None,mask=None): 
        if mask is not None:
            return self.visual(image.type(self.dtype), output_layers=output_layers, return_vanilla_cls=return_vanilla_cls, out_type=out_type, weights=weights,mask_list=mask)
        else:
            return self.visual(image.type(self.dtype), output_layers=output_layers, return_vanilla_cls=return_vanilla_cls, out_type=out_type, weights=weights)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, text_ensemble=False, vision_outputs=(-1,), 
                return_vanilla_cls=False, interpolate=False, vision_out_type="mean",
                save_weights=False, K=3, topk_equal_weights=False, return_all_cls=False):
        
        logit_scale = self.logit_scale.exp()

        if text_ensemble:
            text_features = text
            # text_features = text_features.T
        else:
            text_features = self.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(0) # (#templates, #classes, #features)

        mask_top_k = 101
        if mask_top_k == 0:
            masks = None
        else:
            if self.use_sam:
                mask_list = self.sammask.get_masks(image)
                batch_masks = []
                for img_masks in mask_list:
                    mask_data = [{'segmentation': m['segmentation'], 'area': m.get('area', 0)} for m in img_masks]
                    if mask_top_k<100:
                        mask_data_sorted = sorted(mask_data, key=lambda x: x['area'], reverse=True)
                        if mask_top_k > 0 :
                            mask_data_sorted = mask_data_sorted[mask_top_k:mask_top_k+1]
                        segmentations = [item['segmentation'] for item in mask_data_sorted]
                        batch_masks.append(np.stack(segmentations, axis=0) if segmentations else np.empty((0, image.shape[-2], image.shape[-1])))
                    else:
                        segmentations = [item['segmentation'] for item in mask_data]
                        batch_masks.append(np.stack(segmentations, axis=0) if segmentations else np.empty((0, image.shape[-2], image.shape[-1])))
                masks = np.stack(batch_masks, axis=0)  # [B, num_masks, H, W]
                masks = torch.from_numpy(masks).float().to(device=image.device,dtype=image.dtype) 
                # print(f"Masks shape: {masks.shape}")
                if masks.shape[1]==0:
                    print("No masks generated for image.")
            else:
                masks = None

        if masks is not None:
            resize_transform = transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)
            masks = resize_transform(masks)
            mask_no_empty = []
            for m in range(masks.shape[1]):
                mask = masks[0][m]
                if mask.sum() !=0:
                    mask_no_empty.append(mask)
            if len(mask_no_empty)==0:
                masks = None
                print("No masks generated for image.")
            else:
                masks = torch.stack(mask_no_empty, dim=0).unsqueeze(0) 


        if vision_out_type == "mean":
            if return_vanilla_cls:
                if self.use_sam and masks is not None:
                    # import pdb; pdb.set_trace()
                    image_features, vanilla_cls_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="mean",mask=masks)#image.shape:torch.Size([2, 3, 224, 224])
                else:
                    image_features, vanilla_cls_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="mean")
            else:
                image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="mean")

            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            if self.use_sam and masks is not None:
                after_mask_pooling = False
                if after_mask_pooling:
                    image_feature = self.masked_center(image_features[:,1:,].permute(0,2,1).reshape(image_features.shape[0],image_features.shape[2],16,16),masks)
                    image_features = torch.cat([image_features[:,0:1,:],image_feature],dim=1)
                else:
                    image_features = image_features
                if self.voted:
                    logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)
                    logits = logits[:, :, 1:] # (#templates, batch_size, tokens-1, #classes)
                    
                    templates, batch_size, tokens, classes = logits.shape
                    _, num_masks, h, w = masks.shape
                    temp_dim = templates
                    b_dim = batch_size
                    out_dim = classes
                    
                    logits = logits.permute(0, 1, 3, 2).reshape(templates, batch_size, classes, h, w)  
                    
                    output_logits = logits.clone()
                    if self.clsvoted:
                        for b in range(batch_size):
                            for m in range(num_masks):
                                mask_binary = masks[b, m] > 0  # (h, w)
                                if mask_binary.sum() > 0:  
                                    mask_logits = logits[:, b, :, mask_binary]  # (templates, classes, num_pixels_in_mask)
                                    
                                    avg_over_templates = mask_logits.mean(dim=0)  # (classes, num_pixels_in_mask)
                                    
                                    pixel_classes = avg_over_templates.argmax(dim=0)  # (num_pixels_in_mask,)
                                    
                                    most_common_class = torch.bincount(pixel_classes).argmax()
                                    
                                    class_mask = (pixel_classes == most_common_class)  # 该类别的像素位置
                                    voted_logits = mask_logits[:, :, class_mask].mean(dim=-1, keepdim=True)  # (templates, classes, 1)
                                    
                                    output_logits[:, b, :, mask_binary] = voted_logits
                    else:
                        for b in range(batch_size):
                            for m in range(num_masks):
                                mask_binary = masks[b, m] > 0  # (h, w)
                                if mask_binary.sum() > 0:  
                                    mask_logits = logits[:, b, :, mask_binary]  # (templates, classes, num_pixels_in_mask)
                                    
                                    voted_logits = mask_logits.mean(dim=-1, keepdim=True)  # (templates, classes, 1)
                                    
                                    output_logits[:, b, :, mask_binary] = voted_logits
                    
                    logits1 = output_logits
                    if self.logits2prompt:
                        batch_size = logits.shape[1]
                        if self.prompt_type == 'box':
                            batch_prompts = self.logits2sam.generate_box_prompts(logits, original_img_shape=image.shape[-2:])
                        elif self.prompt_type == 'point':
                            batch_prompts = self.logits2sam.generate_prompts(logits, original_img_shape=image.shape[-2:])
                        else:
                            raise ValueError(f"Unknown prompt_type: {self.prompt_type}. Must be 'point' or 'box'.")
                        
                        spatial_features_list = []
                        for b_idx in range(batch_size):
                            spatial_feat = self._convert_tokens_to_spatial_features(image_features, batch_idx=b_idx)
                            spatial_features_list.append(spatial_feat)
                        
                        masks = self.logits2sam.get_masks(image, batch_prompts,
                                                         image_feature=spatial_features_list,
                                                         text_feature=text_features)
                        if masks is not None:
                            _, num_masks, h_mask, w_mask = masks.shape
                            resize_transform = transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)
                            masks = resize_transform(masks).to(image.dtype)
                            output_logits = logits1.clone()
                            
                            for b in range(batch_size):
                                for m in range(num_masks):
                                    mask_binary = masks[b, m] > 0  # (h, w)
                                    if mask_binary.sum() > 0:  
                                        mask_logits = logits1[:, b, :, mask_binary]  # (templates, classes, num_pixels_in_mask)
                                        
                                        voted_logits = mask_logits.mean(dim=-1, keepdim=True)  # (templates, classes, 1)
                                        output_logits[:, b, :, mask_binary] = voted_logits
                            logits = output_logits
                        else:
                            logits = logits1
                    else:
                        logits = logits1
                else:
                    logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)
                    logits = logits[:, :, 1:] # (#templates, batch_size, tokens-1, #classes)
                    all_cls = logits[:, :, 0] # (#templates, batch_size, #classes)
                    temp_dim = logits.shape[0]
                    b_dim = logits.shape[1]
                    out_dim = logits.shape[-1]

                    templates, batch_size, tokens, classes = logits.shape
                    _, num_masks, h, w = masks.shape
                    output_logits = torch.full((templates, batch_size, 30, h, w),
                                -10.0, device=logits.device, dtype=logits.dtype)
                    logits_rearranged = logits.permute(0, 1, 3, 2)  # [7, 2, 30, 6]
                    for b in range(batch_size):
                        for m in range(num_masks):
                            mask_binary = masks[b, m] > 0
                            output_logits[:, b, :, mask_binary] = logits_rearranged[:, b, :, m:m+1]

                    logits = output_logits  # (#templates, batch_size, #classes, H, W)
            else:
                logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)
                logits = logits[:, :, 1:] # (#templates, batch_size, tokens-1, #classes)
                all_cls = logits[:, :, 0] # (#templates, batch_size, #classes)
                # logits = logits - 0.05 * all_cls.unsqueeze(2)
                patch_size = self.visual.patch_size
                w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size###16,16
                temp_dim = logits.shape[0]
                b_dim = logits.shape[1]
                out_dim = logits.shape[-1]
                logits = logits.permute(0, 1, 3, 2).reshape(logits.shape[0], logits.shape[1], out_dim, w, h) # (#templates, batch_size, #class, W, H)

            
            if interpolate:
                # Perform interpolation
                logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
                logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#templates*batch_size, #class, W', H')

                # Reshape back to include template and batch dimensions
                logits = logits.view(temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#templates, batch_size, #class, W', H')

            if return_vanilla_cls:
                vanilla_cls_features = vanilla_cls_features / vanilla_cls_features.norm(dim=-1, keepdim=True)
                vanilla_cls_logits = logit_scale * torch.einsum('bd,tcd->tbc', vanilla_cls_features, text_features) # (#templates, batch_size, #classes)
                return logits, image_features, text_features, vanilla_cls_logits
            elif return_all_cls:
                return logits, image_features, text_features, all_cls
            
            return logits, image_features, text_features
        
        elif vision_out_type == "adaptive_weighted_mean":
            image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="all")
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # (#outlayers, batch_size, tokens, #features)
            logits = logit_scale * torch.einsum('obsd,tcd->otbsc', image_features, text_features) # (#outlayers, #templates, batch_size, tokens, #classes)
            logits = logits[:, :, :, 1:] # (#outlayers, #templates, batch_size, tokens-1, #classes)
            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            layers_dim = logits.shape[0]
            temp_dim = logits.shape[1]
            b_dim = logits.shape[2]
            out_dim = logits.shape[-1]
            logits = logits.permute(0, 1, 2, 4, 3).reshape(logits.shape[0], logits.shape[1], logits.shape[2], out_dim, w, h) # (#outlayers, #templates, batch_size, #class, W, H)
            
            ent = -(logits.softmax(-3) * logits.log_softmax(-3)).sum(-3) 
            ent_weights = torch.mean(ent, dim=[1, 3, 4]) 
            ent_weights = -ent_weights 
            ent_weights = F.softmax(ent_weights, dim=0) # (#outlayers, batch_size)

            # save the entropy weights to a list (detach them to avoid backpropagation)
            if save_weights: # just to be sure the weights are saved only during evaluation
                self.weights_track.append(ent_weights.detach().cpu().numpy()) # (#outlayers, batch_size)

            # now recalcualte the logits based on the entropy weights
            if self.use_sam and masks is not None:
                image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="weighted_mean", weights=ent_weights,mask=masks) # (batch_size, tokens, #features)
            else:
                image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="weighted_mean", weights=ent_weights)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            if self.use_sam and masks is not None:
                after_mask_pooling = False
                if after_mask_pooling:
                    image_feature = self.masked_center(image_features[:,1:,].permute(0,2,1).reshape(image_features.shape[0],image_features.shape[2],16,16),masks)
                    image_features = torch.cat([image_features[:,0:1,:],image_feature],dim=1)
                else:
                    image_features = image_features
                if self.voted:
                    logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)
                    logits = logits[:, :, 1:] # (#templates, batch_size, tokens-1, #classes)
                    
                    templates, batch_size, tokens, classes = logits.shape
                    _, num_masks, h, w = masks.shape
                    temp_dim = templates
                    b_dim = batch_size
                    out_dim = classes
                    
                    logits = logits.permute(0, 1, 3, 2).reshape(templates, batch_size, classes, h, w)  # (#templates, batch_size, #classes, h, w)
                    
                    output_logits = logits.clone()
                    if self.clsvoted:
                        for b in range(batch_size):
                            for m in range(num_masks):
                                mask_binary = masks[b, m] > 0  # (h, w)
                                if mask_binary.sum() > 0:  
                                    mask_logits = logits[:, b, :, mask_binary]  # (templates, classes, num_pixels_in_mask)
                                    
                                    avg_over_templates = mask_logits.mean(dim=0)  # (classes, num_pixels_in_mask)
                                    
                                    probs = torch.softmax(avg_over_templates, dim=0)  # (classes, num_pixels_in_mask)
                                    
                                    class_votes = probs.sum(dim=1)  # (classes,)
                                    
                                    most_common_class = class_votes.argmax()
                                    
                                    class_probs = probs[most_common_class, :]  # (num_pixels_in_mask,)
                                    
                                    weights = class_probs / class_probs.sum()  # (num_pixels_in_mask,)
                                    voted_logits = (mask_logits * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True)
                                    # (templates, classes, 1)
                                    
                                    output_logits[:, b, :, mask_binary] = voted_logits 
                        logits1 = output_logits
                    else:
                        for b in range(batch_size):
                            for m in range(num_masks):
                                mask_binary = masks[b, m] > 0  # (h, w)
                                if mask_binary.sum() > 0:  
                                    mask_logits = logits[:, b, :, mask_binary]  # (templates, classes, num_pixels_in_mask)
                                    
                                    voted_logits = mask_logits.mean(dim=-1, keepdim=True)  # (templates, classes, 1)
                                    
                                    output_logits[:, b, :, mask_binary] = voted_logits
                            
                        logits1 = output_logits
                    if self.logits2prompt:
                        batch_size = logits.shape[1]
                        if self.prompt_type == 'box':
                            batch_prompts = self.logits2sam.generate_box_prompts(logits, original_img_shape=image.shape[-2:])
                        elif self.prompt_type == 'point':
                            batch_prompts = self.logits2sam.generate_prompts(logits, original_img_shape=image.shape[-2:])
                        else:
                            raise ValueError(f"Unknown prompt_type: {self.prompt_type}. Must be 'point' or 'box'.")
                        spatial_features_list = []
                        for b_idx in range(batch_size):
                            spatial_feat = self._convert_tokens_to_spatial_features(image_features, batch_idx=b_idx)
                            spatial_features_list.append(spatial_feat)
 
                        
                        masks = self.logits2sam.get_masks(image, batch_prompts,
                                                         image_feature=spatial_features_list,
                                                         text_feature=text_features)
                        if masks is not None:
                            _, num_masks, h_mask, w_mask = masks.shape
                            resize_transform = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)
                            masks = resize_transform(masks).to(image.dtype)  # (B, num_masks, h, w)
                            output_logits = logits.clone()
                            
                            for b in range(batch_size):
                                for m in range(num_masks):
                                    mask_binary = masks[b, m] > 0
                                    if mask_binary.sum() > 0:
                                        mask_logits = logits1[:, b, :, mask_binary]
                                        voted_logits = mask_logits.mean(dim=-1, keepdim=True)
                                        output_logits[:, b, :, mask_binary] = voted_logits
                            
                            logits = output_logits
                        else:
                            logits = logits1
                    else:
                        logits = logits1

                else:
                    logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)
                    logits = logits[:, :, 1:] # (#templates, batch_size, tokens-1, #classes)
                    temp_dim = logits.shape[0]
                    b_dim = logits.shape[1]
                    out_dim = logits.shape[-1]
                    templates, batch_size, tokens, classes = logits.shape
                    _, num_masks, h, w = masks.shape
                    output_logits = torch.full((templates, batch_size, classes, h, w),
                    -10.0, device=logits.device, dtype=logits.dtype)
                    logits_rearranged = logits.permute(0, 1, 3, 2)  # [7, 2, 30, 6]
                    for b in range(batch_size):
                        for m in range(num_masks):
                            mask_binary = masks[b, m] > 0
                            output_logits[:, b, :, mask_binary] = logits_rearranged[:, b, :, m:m+1]

                    logits = output_logits
            else:
                logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)
                logits = logits[:, :, 1:] # (#templates, batch_size, tokens-1, #classes)
                patch_size = self.visual.patch_size
                w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
                temp_dim = logits.shape[0]
                b_dim = logits.shape[1]
                out_dim = logits.shape[-1]
                logits = logits.permute(0, 1, 3, 2).reshape(logits.shape[0], logits.shape[1], out_dim, w, h) # (#templates, batch_size, #class, W, H)
                if self.logits2prompt:
                    batch_size = logits.shape[1]
                    
                    text_embeddings_for_scoring = text_features.mean(dim=0) if text_features.dim() == 3 else text_features  # (#classes, D)
                    
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
                        #     # resize_transform = transforms.Resize((14, 14), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)##对于VITB16
                        #     # resize_transform = transforms.Resize((7, 7), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)##对于VITB32
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
                            resize_transform = transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)
                            masks = resize_transform(masks).to(image.dtype)
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
                                for b in range(batch_size):
                                    for m in range(num_masks - 1, -1, -1):  
                                        mask_binary = masks[b, m] > 0  # (h, w)
                                        if mask_binary.sum() > 0:  
                                            mask_logits = logits[:, b, :, mask_binary]  # (templates, classes, num_pixels_in_mask)
                                            
                                            voted_logits = mask_logits.mean(dim=-1, keepdim=True)  # (templates, classes, 1)
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
                                    logits = logits + 0.5 * output_logits
                                # print(f"Iteration {iteration+1}/{self.logits2prompt_iterations}: Logits optimized with {num_masks} masks.")
                            else:
                                break
                        else:
                            # print(f"Iteration {iteration+1}/{self.logits2prompt_iterations}: No masks generated, stopping iteration.")
                            break
                elif self.random_mask:
                    batch_size = logits.shape[1]
                    mask_list = self.sammask.get_masks(image)
                    batch_masks = []
                    for img_masks in mask_list:
                        mask_data = [{'segmentation': m['segmentation'], 'area': m.get('area', 0), 'predicted_iou': m.get('predicted_iou', 0)} for m in img_masks]
                        mask_data_iou_sorted = sorted(mask_data, key=lambda x: x['predicted_iou'], reverse=True)
                        number_of_masks = min(len(mask_data_iou_sorted), 2)
                        mask_data_sorted = mask_data_iou_sorted[:number_of_masks]
                        segmentations = [item['segmentation'] for item in mask_data_sorted]
                        batch_masks.append(np.stack(segmentations, axis=0) if segmentations else np.empty((0, image.shape[-2], image.shape[-1])))
                    masks = np.stack(batch_masks, axis=0)  # [B, num_masks, H, W]
                    masks = torch.from_numpy(masks).float().to(device=image.device,dtype=image.dtype) 
                    if masks is not None:
                        resize_transform = transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)
                        masks = resize_transform(masks)
                        mask_no_empty = []
                        for m in range(masks.shape[1]):
                            mask = masks[0][m]
                            if mask.sum() !=0:
                                mask_no_empty.append(mask)
                        if len(mask_no_empty)==0:
                            masks = None
                            print("No masks generated for image.")
                        else:
                            masks = torch.stack(mask_no_empty, dim=0).unsqueeze(0) 
                    if masks is not None:
                        _, num_masks, mask_h, mask_w = masks.shape
                        output_logits = logits.clone()
                        for b in range(batch_size):
                            for m in range(num_masks):
                                mask_binary = masks[b, m] > 0  # (h, w)
                                if mask_binary.sum() > 0:  
                                    mask_logits = logits[:, b, :, mask_binary]  # (templates, classes, num_pixels_in_mask)
                                    
                                    voted_logits = mask_logits.mean(dim=-1, keepdim=True)  # (templates, classes, 1)
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
                            
                            logits = 0 * output_logits + 1 * logits_clustered
                        else:
                            logits = output_logits
                elif self.auto_mask:
                    batch_size = logits.shape[1]
                    
                    spatial_features_batch = []
                    if self.logits2sam.use_semantic:
                        for b_idx in range(batch_size):
                            spatial_feat = self._convert_tokens_to_spatial_features(image_features, batch_idx=b_idx)
                            spatial_features_batch.append(spatial_feat)
                    else:
                        spatial_features_batch = None
                    
                    masks = self.logits2sam.get_masks_auto(image, spatial_features=spatial_features_batch)
                    
                    if masks is not None:
                        resize_transform = transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
                        masks = resize_transform(masks).to(image.dtype)
                        
                        mask_no_empty = []
                        for m in range(masks.shape[1]):
                            mask = masks[0][m]
                            if mask.sum() != 0:
                                mask_no_empty.append(mask)
                        
                        if len(mask_no_empty) == 0:
                            masks = None
                        else:
                            masks = torch.stack(mask_no_empty, dim=0).unsqueeze(0)
                    
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
                                image_features[:, 1:, ].permute(0, 2, 1).reshape(
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
                                text_features_sim = text_features[t_idx].float()
                                similarity_logits = image_features_masked_sim @ text_features_sim.T
                                similarity_flat = similarity_logits.reshape(-1, num_classes_cluster)
                                assignments = self._distributed_sinkhorn(similarity_flat, n_iters=3)
                                assignments = assignments.reshape(B_cluster, num_masks_cluster, num_classes_cluster)
                                spatial_logits = torch.einsum('bmc,bmhw->bchw', assignments.to(masks.device), masks.float()).to("cuda")
                                logits_clustered_list.append(spatial_logits)
                            
                            logits_clustered = torch.stack(logits_clustered_list, dim=0)
                            logits_clustered = logit_scale * logits_clustered
                            logits = logits + 0.5 * (output_logits + logits_clustered)
                        else:
                            logits = logits + 0.5 * output_logits
                else:
                    logits = logits
            
            if interpolate:
                # Perform interpolation
                logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
                logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#templates*batch_size, #class, W', H')

                # Reshape back to include template and batch dimensions
                logits = logits.view(temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#templates, batch_size, #class, W', H')

            if return_vanilla_cls:
                vanilla_cls_features = vanilla_cls_features / vanilla_cls_features.norm(dim=-1, keepdim=True)
                vanilla_cls_logits = logit_scale * torch.einsum('bd,tcd->tbc', vanilla_cls_features, text_features) # (#templates, batch_size, #classes)
                return logits, image_features, text_features, vanilla_cls_logits
            
            return logits, image_features, text_features
        
    def erode_mask_to_center(self, masks, min_mask_area=20, erosion_kernel_size=3, erosion_iterations=1):
        
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
        K = logits.shape[1]  
        
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

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
