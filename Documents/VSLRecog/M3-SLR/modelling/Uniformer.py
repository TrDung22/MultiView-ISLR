from modelling.Uniformer_base import build_uniformer_small
from modelling.maskUniformer_base import build_mask_uniformer_small
import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from flash_attn import flash_attn_qkvpacked_func
from transformers import pipeline

class UFOneView(nn.Module):
    def __init__(self, num_classes=199, maskFeat=False, pretraiend=False, pretrained_name=None, device=None, **kwargs):
        super().__init__()
        self.maskFeat = maskFeat
        if maskFeat:
            self.model = build_mask_uniformer_small(
                num_classes=num_classes, 
                pretrained=pretraiend, 
                pretrained_name=pretrained_name, 
                device=device
            )
        else:
            self.model = build_uniformer_small(
                num_classes=num_classes, 
                pretrained=pretraiend, 
                pretrained_name=pretrained_name, 
                device=device
            )
    
    def forward_ft(self, x, return_features=False):
        features = {}

        # Stage 1
        x = self.model.patch_embed1(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks1:
            x = blk(x)
        features['stage1'] = x.clone()

        # Stage 2
        x = self.model.patch_embed2(x)
        for blk in self.model.blocks2:
            x = blk(x)
        features['stage2'] = x.clone()

        # Stage 3
        x = self.model.patch_embed3(x)
        for blk in self.model.blocks3:
            x = blk(x)
        features['stage3'] = x.clone()

        # Stage 4
        x = self.model.patch_embed4(x)
        for blk in self.model.blocks4:
            x = blk(x)
        features['stage4'] = x.clone()

        x = self.model.norm(x)
        x = self.model.pre_logits(x)
        ft = x.flatten(2).mean(-1)

        # Kiểm tra hierarchical_simkd từ UsimKD
        if return_features:
            return ft, features
        return ft

    def forward(self, clip=None, gloss=None):
        if self.maskFeat:
            preds, labels = self.model(clip)
            return {'preds': preds, 'labels': labels}
        else:
            logits = self.model(clip)
            return {'logits': logits}
    
# --- DCA helper functions and modules ---

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult)
        self.act = GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)
    def forward(self, x, **kwargs):
        x = self.w1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w2(x)
        return self.dropout(x)

class FlashAttentionForFusion(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dimension must be divisible by number of heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear layers cho Q, K, V
        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Tạo Q, K, V trong một lần
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape  # x: [B, N, C]
        B, M, C = context.shape  # context: [B, M, C]

        # Tính QKV từ x và context
        qkv_x = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        qkv_context = self.to_qkv(context).reshape(B, M, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, heads, M, head_dim]

        # Lấy Q từ x, K và V từ context
        q = qkv_x[0]  # [B, heads, N, head_dim]
        k = qkv_context[1]  # [B, heads, M, head_dim]
        v = qkv_context[2]  # [B, heads, M, head_dim]

        # Ép kiểu sang fp16 (yêu cầu của flash_attn_qkvpacked_func)
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

        # Tạo tensor QKV packed cho x (chỉ dùng Q, K và V giả để tương thích API)
        qkv_packed = torch.stack([q, k, v], dim=2).reshape(B, N, 3, self.num_heads, self.head_dim)  # [B, N, 3, heads, head_dim]

        # Gọi flash_attn_qkvpacked_func
        attn_output = flash_attn_qkvpacked_func(
            qkv_packed,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False
        )  # [B, N, heads, head_dim]

        # Reshape và ép kiểu về fp32
        x = attn_output.reshape(B, N, self.num_heads * self.head_dim).to(torch.float32)

        # Áp dụng projection và dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- FlashAttentionBlock ---
class FlashAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, FlashAttentionForFusion(dim, num_heads=num_heads, attn_drop=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, dropout=dropout))
    
    def forward(self, x, context=None):
        x = x + self.attn(x, context=context)
        x = x + self.ff(x)
        return x

# --- Helper functions to reshape feature maps ---
def flatten_features(x):
    # x: [B, C, T, H, W] -> [B, N, C] với N = T*H*W
    B, C, T, H, W = x.shape
    return x.flatten(2).transpose(1,2)

def reshape_features(x, orig_shape):
    # Ngược lại: [B, N, C] -> [B, C, T, H, W]
    B, C, T, H, W = orig_shape
    return x.transpose(1,2).view(B, C, T, H, W)

class UFThreeView(nn.Module):
    def __init__(self, num_classes=199, embed_size=512, pseudo_context=False, prompt_type=None, pretraiend=True, 
                 pretrained_left=None, pretrained_center=None, pretrained_right=None, 
                 co_attetion=True, maskFeat=False, device=None, **kwargs):
        """
        num_classes: số lớp đầu ra.
        embed_size: kích thước embedding (dùng cho fusion cuối cùng).
        device: thiết bị chạy mô hình.
        """
        super().__init__()
        self.device = device
        self.co_attention = co_attetion
        self.maskFeat = maskFeat
        self.pseudo_context = pseudo_context
        self.prompt_type = prompt_type
        # 3 backbone Uniformer-S 
        self.left_backbone   = UFOneView(num_classes=num_classes, maskFeat=maskFeat, pretraiend=pretraiend, 
                                             pretrained_name=pretrained_left, device=device)
        self.center_backbone = UFOneView(num_classes=num_classes, maskFeat=maskFeat, pretraiend=pretraiend, 
                                             pretrained_name=pretrained_center, device=device)
        self.right_backbone  = UFOneView(num_classes=num_classes, maskFeat=maskFeat, pretraiend=pretraiend, 
                                             pretrained_name=pretrained_right, device=device)

        # Fusion head 
        self.fusion_ft_size = embed_size * 3
        self.head = nn.Linear(self.fusion_ft_size, num_classes) if num_classes > 0 else nn.Identity()
        if device is not None:
            self.head.to(device)
        
        if co_attetion:
            self.attention_blocks = nn.ModuleDict({
                "1": nn.ModuleDict({
                    "left": FlashAttentionBlock(dim=64, num_heads=1, dropout=0.),
                    "right": FlashAttentionBlock(dim=64, num_heads=1, dropout=0.),
                    "center_left": FlashAttentionBlock(dim=64, num_heads=1, dropout=0.),
                    "center_right": FlashAttentionBlock(dim=64, num_heads=1, dropout=0.)
                }),
                "2": nn.ModuleDict({
                    "left": FlashAttentionBlock(dim=128, num_heads=2, dropout=0.),
                    "right": FlashAttentionBlock(dim=128, num_heads=2, dropout=0.),
                    "center_left": FlashAttentionBlock(dim=128, num_heads=2, dropout=0.),
                    "center_right": FlashAttentionBlock(dim=128, num_heads=2, dropout=0.)
                }),
                "3": nn.ModuleDict({
                    "left": FlashAttentionBlock(dim=320, num_heads=5, dropout=0.),
                    "right": FlashAttentionBlock(dim=320, num_heads=5, dropout=0.),
                    "center_left": FlashAttentionBlock(dim=320, num_heads=5, dropout=0.),
                    "center_right": FlashAttentionBlock(dim=320, num_heads=5, dropout=0.)
                }),
                "4": nn.ModuleDict({
                    "left": FlashAttentionBlock(dim=512, num_heads=8, dropout=0.),
                    "right": FlashAttentionBlock(dim=512, num_heads=8, dropout=0.),
                    "center_left": FlashAttentionBlock(dim=512, num_heads=8, dropout=0.),
                    "center_right": FlashAttentionBlock(dim=512, num_heads=8, dropout=0.)
                })
            })

        if self.pseudo_context:
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
            pipe = pipeline(
                "text-generation", 
                model=model_id, 
                torch_dtype=torch.bfloat16, 
                device_map=device
            )
            self.generator = pipe
            self.text_embedder = pipe.model
            self.tokenizer = pipe.tokenizer

    
    def pyramid_fusion(self, Zl, Zc, Zr, gloss, stage):
        # Flatten các đặc trưng: [B, C, T, H, W] -> [B, N, C]
        Zl_flat = flatten_features(Zl)
        Zc_flat = flatten_features(Zc)
        Zr_flat = flatten_features(Zr)
        
        # Left view: fusion với context từ center
        left_fusion = self.attention_blocks[str(stage)]["left"](Zl_flat, context=Zc_flat)
        left_new = left_fusion
        
        # Right view: fusion với context từ center
        right_fusion = self.attention_blocks[str(stage)]["right"](Zr_flat, context=Zc_flat)
        right_new = right_fusion
        
        # Center view: fusion từ left và right (2 nhánh)
        center_fusion_left = self.attention_blocks[str(stage)]["center_left"](Zc_flat, context=Zl_flat)
        center_fusion_right = self.attention_blocks[str(stage)]["center_right"](Zc_flat, context=Zr_flat)
        center_new = center_fusion_left + center_fusion_right - Zc_flat
        
        # Reshape lại về [B, C, T, H, W]
        left_new = reshape_features(left_new, Zl.shape)
        center_new = reshape_features(center_new, Zc.shape)
        right_new = reshape_features(right_new, Zr.shape)
        return left_new, center_new, right_new
    
    def forward_ft(self, rgb_left=None, rgb_center=None, rgb_right=None, gloss=None, return_features=False):
            features = {}

            # --- Stage 1 ---
            left = self.left_backbone.model.patch_embed1(rgb_left)
            center = self.center_backbone.model.patch_embed1(rgb_center)
            right = self.right_backbone.model.patch_embed1(rgb_right)

            left = self.left_backbone.model.pos_drop(left)
            center = self.center_backbone.model.pos_drop(center)
            right = self.right_backbone.model.pos_drop(right)

            for blk in self.left_backbone.model.blocks1:
                left = blk(left)
            for blk in self.center_backbone.model.blocks1:
                center = blk(center)
            for blk in self.right_backbone.model.blocks1:
                right = blk(right)

            # if self.co_attention:
            #     left, center, right = self.pyramid_fusion(left, center, right, gloss, stage=1)
            # else:
            #     left = left
            #     center = center
            #     right = right

            features['stage1'] = {
                'left': left.clone(),
                'center': center.clone(),
                'right': right.clone()
            }
            
            # --- Stage 2 ---
            left = self.left_backbone.model.patch_embed2(left)
            center = self.center_backbone.model.patch_embed2(center)
            right = self.right_backbone.model.patch_embed2(right)
            for blk in self.left_backbone.model.blocks2:
                left = blk(left)
            for blk in self.center_backbone.model.blocks2:
                center = blk(center)
            for blk in self.right_backbone.model.blocks2:
                right = blk(right)
            
            # if self.co_attention:
            #     left, center, right = self.pyramid_fusion(left, center, right, gloss, stage=2)
            # else:
            #     left = left
            #     center = center
            #     right = right

            features['stage2'] = {
                'left': left.clone(),
                'center': center.clone(),
                'right': right.clone()
            }
            
            # --- Stage 3 ---
            left = self.left_backbone.model.patch_embed3(left)
            center = self.center_backbone.model.patch_embed3(center)
            right = self.right_backbone.model.patch_embed3(right)
            for blk in self.left_backbone.model.blocks3:
                left = blk(left)
            for blk in self.center_backbone.model.blocks3:
                center = blk(center)
            for blk in self.right_backbone.model.blocks3:
                right = blk(right)
            
            if self.co_attention:
                left, center, right = self.pyramid_fusion(left, center, right, gloss, stage=3)
            else:
                left = left
                center = center
                right = right
            
            features['stage3'] = {
                'left': left.clone(),
                'center': center.clone(),
                'right': right.clone()
            }
            
            # --- Stage 4 ---
            left = self.left_backbone.model.patch_embed4(left)
            center = self.center_backbone.model.patch_embed4(center)
            right = self.right_backbone.model.patch_embed4(right)
            for blk in self.left_backbone.model.blocks4:
                left = blk(left)
            for blk in self.center_backbone.model.blocks4:
                center = blk(center)
            for blk in self.right_backbone.model.blocks4:
                right = blk(right)
            
            if self.co_attention:
                left, center, right = self.pyramid_fusion(left, center, right, gloss, stage=4)
            else:
                left = left
                center = center
                right = right
            
            features['stage4'] = {
                'left': left.clone(),
                'center': center.clone(),
                'right': right.clone()
            }

            left = self.left_backbone.model.norm(left)
            center = self.center_backbone.model.norm(center)
            right = self.right_backbone.model.norm(right)

            left = self.left_backbone.model.pre_logits(left)
            center = self.center_backbone.model.pre_logits(center)
            right = self.right_backbone.model.pre_logits(right)
            
            left_ft = left.flatten(2).mean(-1)   # [B, embed_dim]
            center_ft = center.flatten(2).mean(-1)
            right_ft = right.flatten(2).mean(-1)
            
            fusion_ft = torch.cat([left_ft, center_ft, right_ft], dim=-1) 
            if return_features:
                return fusion_ft, features

            return fusion_ft
    
    def forward_maskfeat(self, rgb_left=None, rgb_center=None, rgb_right=None):
            rgb_left, mask_left = rgb_left
            rgb_center, mask_center = rgb_center
            rgb_right, mask_right = rgb_right

            raw_left = rgb_left.clone()
            raw_center = rgb_center.clone()
            raw_right = rgb_right.clone()

            rgb_left = self.left_backbone.model.patch_embed1(rgb_left)
            rgb_center = self.center_backbone.model.patch_embed1(rgb_center)
            rgb_right = self.right_backbone.model.patch_embed1(rgb_right)

            B, C, T, H, W = rgb_center.shape

            tokens_left = rgb_left.flatten(2).transpose(1, 2)
            tokens_center = rgb_center.flatten(2).transpose(1, 2)
            tokens_right = rgb_right.flatten(2).transpose(1, 2)

            float_mask_left = mask_left.type_as(tokens_left)
            float_mask_left = F.interpolate(float_mask_left.unsqueeze(1), size=(T,H,W), mode='nearest').squeeze(1)
            float_mask_left = float_mask_left.flatten(1).unsqueeze(-1)

            float_mask_center = mask_center.type_as(tokens_center)
            float_mask_center = F.interpolate(float_mask_center.unsqueeze(1), size=(T,H,W), mode='nearest').squeeze(1)
            float_mask_center = float_mask_center.flatten(1).unsqueeze(-1)

            float_mask_right = mask_right.type_as(tokens_right)
            float_mask_right = F.interpolate(float_mask_right.unsqueeze(1), size=(T,H,W), mode='nearest').squeeze(1)
            float_mask_right = float_mask_right.flatten(1).unsqueeze(-1)

            output_masks_left = self.left_backbone.model._get_multiscale_mask(mask_left)
            output_masks_center = self.center_backbone.model._get_multiscale_mask(mask_center)
            output_masks_right = self.right_backbone.model._get_multiscale_mask(mask_right)

            labels_left = self.left_backbone.model._get_hog_label_3d(raw_left.detach(), output_masks_left)
            labels_center = self.center_backbone.model._get_hog_label_3d(raw_center.detach(), output_masks_center)
            labels_right = self.right_backbone.model._get_hog_label_3d(raw_right.detach(), output_masks_right)

            mask_tokens_left = self.left_backbone.model.mask_token.expand(B, tokens_center.size(1), -1)
            mask_tokens_center = self.center_backbone.model.mask_token.expand(B, tokens_center.size(1), -1)
            mask_tokens_right = self.right_backbone.model.mask_token.expand(B, tokens_center.size(1), -1)

            tokens_left = tokens_left * (1 - float_mask_left) + mask_tokens_left * float_mask_left
            tokens_center = tokens_center * (1 - float_mask_center) + mask_tokens_center * float_mask_center
            tokens_right = tokens_right * (1 - float_mask_right) + mask_tokens_right * float_mask_right

            rgb_left = tokens_left.transpose(1, 2).view(B,C,T,H,W)
            rgb_center = tokens_center.transpose(1, 2).view(B,C,T,H,W)
            rgb_right = tokens_right.transpose(1, 2).view(B,C,T,H,W)

            block_outputs_left = []
            block_outputs_center = []
            block_outputs_right = []
            current_depth_left = 0
            current_depth_center = 0
            current_depth_right = 0

            for i, blk in enumerate(self.left_backbone.model.blocks1):
                rgb_left = blk(rgb_left)
                if current_depth_left in self.left_backbone.model.pretrain_depth:
                    block_outputs_left.append(rgb_left)
                current_depth_left += 1
            
            for i, blk in enumerate(self.center_backbone.model.blocks1):
                rgb_center = blk(rgb_center)
                if current_depth_center in self.center_backbone.model.pretrain_depth:
                    block_outputs_center.append(rgb_center)
                current_depth_center += 1

            for i, blk in enumerate(self.right_backbone.model.blocks1):
                rgb_right = blk(rgb_right)
                if current_depth_right in self.right_backbone.model.pretrain_depth:
                    block_outputs_right.append(rgb_right)
                current_depth_right += 1
            
            if self.co_attention:
                rgb_left, rgb_center, rgb_right = self.pyramid_fusion(rgb_left, rgb_center, rgb_right, gloss, stage=1)
            else:
                rgb_left = rgb_left
                rgb_center = rgb_center
                rgb_right = rgb_right

            rgb_left = self.left_backbone.model.patch_embed2(rgb_left)
            rgb_center = self.center_backbone.model.patch_embed2(rgb_center)
            rgb_right = self.right_backbone.model.patch_embed2(rgb_right)

            for i, blk in enumerate(self.left_backbone.model.blocks2):
                rgb_left = blk(rgb_left)
                if current_depth_left in self.left_backbone.model.pretrain_depth:
                    block_outputs_left.append(rgb_left)
                current_depth_left += 1
            
            for i, blk in enumerate(self.center_backbone.model.blocks2):
                rgb_center = blk(rgb_center)
                if current_depth_center in self.center_backbone.model.pretrain_depth:
                    block_outputs_center.append(rgb_center)
                current_depth_center += 1

            for i, blk in enumerate(self.right_backbone.model.blocks2):
                rgb_right = blk(rgb_right)
                if current_depth_right in self.right_backbone.model.pretrain_depth:
                    block_outputs_right.append(rgb_right)
                current_depth_right += 1
            
            if self.co_attention:
                rgb_left, rgb_center, rgb_right = self.pyramid_fusion(rgb_left, rgb_center, rgb_right, gloss, stage=2)
            else:
                rgb_left = rgb_left
                rgb_center = rgb_center
                rgb_right = rgb_right

            rgb_left = self.left_backbone.model.patch_embed3(rgb_left)
            rgb_center = self.center_backbone.model.patch_embed3(rgb_center)
            rgb_right = self.right_backbone.model.patch_embed3(rgb_right)

            for i, blk in enumerate(self.left_backbone.model.blocks3):
                rgb_left = blk(rgb_left)
                if current_depth_left in self.left_backbone.model.pretrain_depth:
                    block_outputs_left.append(rgb_left)
                current_depth_left += 1
            
            for i, blk in enumerate(self.center_backbone.model.blocks3):
                rgb_center = blk(rgb_center)
                if current_depth_center in self.center_backbone.model.pretrain_depth:
                    block_outputs_center.append(rgb_center)
                current_depth_center += 1

            for i, blk in enumerate(self.right_backbone.model.blocks3):
                rgb_right = blk(rgb_right)
                if current_depth_right in self.right_backbone.model.pretrain_depth:
                    block_outputs_right.append(rgb_right)
                current_depth_right += 1
            
            if self.co_attention:
                rgb_left, rgb_center, rgb_right = self.pyramid_fusion(rgb_left, rgb_center, rgb_right, gloss, stage=3)
            else:
                rgb_left = rgb_left
                rgb_center = rgb_center
                rgb_right = rgb_right

            rgb_left = self.left_backbone.model.patch_embed4(rgb_left)
            rgb_center = self.center_backbone.model.patch_embed4(rgb_center)
            rgb_right = self.right_backbone.model.patch_embed4(rgb_right)

            for i, blk in enumerate(self.left_backbone.model.blocks4):
                rgb_left = blk(rgb_left)
                if current_depth_left in self.left_backbone.model.pretrain_depth:
                    block_outputs_left.append(rgb_left)
                current_depth_left += 1
            
            for i, blk in enumerate(self.center_backbone.model.blocks4):
                rgb_center = blk(rgb_center)
                if current_depth_center in self.center_backbone.model.pretrain_depth:
                    block_outputs_center.append(rgb_center)
                current_depth_center += 1

            for i, blk in enumerate(self.right_backbone.model.blocks4):
                rgb_right = blk(rgb_right)
                if current_depth_right in self.right_backbone.model.pretrain_depth:
                    block_outputs_right.append(rgb_right)
                current_depth_right += 1
            
            if self.co_attention:
                rgb_left, rgb_center, rgb_right = self.pyramid_fusion(rgb_left, rgb_center, rgb_right, gloss, stage=4)
            else:
                rgb_left = rgb_left
                rgb_center = rgb_center
                rgb_right = rgb_right

            outputs_left = []
            if self.left_backbone.model.pred_hog_wt:
                hog_outputs = self.left_backbone.model.pred_head(block_outputs_left, output_masks_left, False)
                outputs_left += hog_outputs

            outputs_center = []
            if self.center_backbone.model.pred_hog_wt:
                hog_outputs = self.center_backbone.model.pred_head(block_outputs_center, output_masks_center, False)
                outputs_center += hog_outputs

            outputs_right = []
            if self.right_backbone.model.pred_hog_wt:
                hog_outputs = self.right_backbone.model.pred_head(block_outputs_right, output_masks_right, False)
                outputs_right += hog_outputs
            
            return outputs_left, outputs_center, outputs_right, labels_left, labels_center, labels_right
    
    def forward(self, rgb_left=None, rgb_center=None, rgb_right=None, gloss=None):
        """
        Forward qua từng stage của Uniformer, sau đó áp dụng fusion Linformer.
        """
        gloss = None 
        # --- Stage 1 ---
        if self.maskFeat:
            outputs_left, outputs_center, outputs_right, labels_left, labels_center, labels_right = self.forward_maskfeat(rgb_left,rgb_center,rgb_right)
            return {'outputs_left': outputs_left, 'labels_left': labels_left,
                    'outputs_center': outputs_center, 'labels_center': labels_center,
                    'outputs_right': outputs_right, 'labels_right': labels_right}

        else:
            fusion_ft = self.forward_ft(rgb_left,rgb_center,rgb_right)
            if self.pseudo_context and gloss is not None:
                # Đảm bảo gloss là danh sách có độ dài bằng batch size
                B = fusion_ft.shape[0]
                assert len(gloss) == B, "Gloss must same size with batch size"

                if self.prompt_type == "story":
                    messages_list = [
                        [{"role": "user", "content": f"Create a story using {g} to highlight its semantic meaning. Answer using one sentence."}]
                        for g in gloss
                    ]
                elif self.prompt_type == "explain":
                    messages_list = [
                        [{"role": "user", "content": f"Explain the meaning of {g} without mention it. Answer using one sentence."}]
                        for g in gloss
                    ]
                elif self.prompt_type == "straightforward":
                    messages_list = [
                        [{"role": "user", "content": f"Repeat the sentence 'This is {g}'."}]
                        for g in gloss
                    ]
                else:
                    raise ValueError("prompt_type must 'story', 'explain', or 'straightforward'")

               
                outputs = self.pipe(messages_list, max_new_tokens=256)
                generated_texts = [output["generated_text"] for output in outputs] 

                text_embeddings = []
                for text in generated_texts:
                    input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
                    with torch.no_grad():
                        model_output = self.text_embedder(input_ids, output_hidden_states=True)
                    hidden_states = model_output.hidden_states[-1]  # [1, seq_len, D_text]
                    embedding = torch.mean(hidden_states[0], dim=0)  # [D_text]
                    text_embeddings.append(embedding)
                text_embeddings = torch.stack(text_embeddings)  # [B, D_text]

            logits = self.head(fusion_ft)
            return {'logits': logits}

class UsimKD(nn.Module):
    def __init__(self, num_classes=199, embed_size=512, pretraiend=True, pretrained_name=None, hierarchical_simkd=False, device=None, **kwargs):
        super().__init__()
        self.hierarchical_simkd = hierarchical_simkd
        self.teacher = UFThreeView(
            num_classes=num_classes,
            embed_size=embed_size,
            pretraiend=False,
            pretrained_left=None,
            pretrained_center=None,
            pretrained_right=None,
            co_attetion=True,
            maskFeat=False,
            device=device
        )
        ckpt_path = "checkpoints/UFThreeView/UFThreeView MultiVSL200 w DCA from MaxFlow MaskFeat 0.4/best_checkpoints_loss.pth"
        self.teacher.load_state_dict(torch.load(ckpt_path,map_location='cpu'))
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher_classifier = self.teacher.head
        
        self.student = UFOneView(
            num_classes=num_classes,
            maskFeat=False,
            pretraiend=pretraiend,
            pretrained_name=pretrained_name,
            device=device
        )
   
        s_n = embed_size  
        t_n = embed_size * 3  
 
        self.projection = nn.Sequential(
            nn.Linear(s_n,t_n*2),
            nn.LayerNorm(t_n*2),
            nn.GELU(),
            nn.Linear(t_n*2,t_n)
        )
        if self.hierarchical_simkd:
            self.projection_layers = nn.ModuleDict({
                'stage1': nn.Conv3d(3 * 64, 64, kernel_size=1),   # Teacher: 3 views * 64 -> Student: 64
                'stage2': nn.Conv3d(3 * 128, 128, kernel_size=1), # Teacher: 3 views * 128 -> Student: 128
                'stage3': nn.Conv3d(3 * 320, 320, kernel_size=1), # Teacher: 3 views * 320 -> Student: 320
                'stage4': nn.Conv3d(3 * 512, 512, kernel_size=1)  # Teacher: 3 views * 512 -> Student: 512
            })
            if device is not None:
                self.projection_layers.to(device)

        if device is not None:
            self.to(device)

    def forward(self, rgb_left=None, rgb_center=None, rgb_right=None, gloss=None):
        logits = None
        student_ft = None
        teacher_ft = None
        student_features = None
        projected_teacher_features = None

        if self.training:
            self.teacher.eval()
            if self.hierarchical_simkd:
                teacher_ft, teacher_features = self.teacher.forward_ft(rgb_left=rgb_left, rgb_center=rgb_center, rgb_right=rgb_right, gloss=gloss, return_features=True)
                student_ft, student_features = self.student.forward_ft(rgb_center, return_features=True)
                student_ft = self.projection(student_ft)

                teacher_ft = teacher_ft.detach()
                for stage in teacher_features:
                    teacher_features[stage]['left'] = teacher_features[stage]['left'].detach()
                    teacher_features[stage]['center'] = teacher_features[stage]['center'].detach()
                    teacher_features[stage]['right'] = teacher_features[stage]['right'].detach()

                # Thực hiện projection cho teacher_features
                projected_teacher_features = {}
                for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                    teacher_feature = teacher_features[stage]
                    teacher_concat = torch.cat([teacher_feature['left'], teacher_feature['center'], teacher_feature['right']], dim=1)  # [B, 3C, T, H, W]
                    
                    projected_teacher = self.projection_layers[stage](teacher_concat)  # [B, C, T, H, W]
                    projected_teacher_features[stage] = projected_teacher
            else:
                teacher_ft = self.teacher.forward_ft(rgb_left=rgb_left, rgb_center=rgb_center, rgb_right=rgb_right, gloss=gloss).detach()
                student_ft = self.student.forward_ft(rgb_center)
                student_ft = self.projection(student_ft)
        else:
            self.student.eval()
            student_ft = self.student.forward_ft(rgb_center)
            student_ft = self.projection(student_ft)
            logits = self.teacher_classifier(student_ft)

        return {
            'trans_feat_s': student_ft,
            'trans_feat_t': teacher_ft,
            'logits': logits,
            'student_features': student_features,
            'teacher_features': projected_teacher_features
        }
