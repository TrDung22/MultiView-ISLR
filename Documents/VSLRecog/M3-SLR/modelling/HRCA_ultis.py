import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models.video import s3d
import torchvision.models as models
import torchvision
from pytorch_lightning.utilities.migration import pl_legacy_patch


class FeatureExtractor(nn.Module):
    """Feature extractor for RGB clips, powered by a 2D CNN backbone."""

    def __init__(self, cnn='rn34', embed_size=512, freeze_layers=0):
        """Initialize the feature extractor with given CNN backbone and desired feature size."""
        super().__init__()

        if cnn == 'rn18':
            model = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif cnn == 'rn34':
            model = resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif cnn == 'rn50':
            model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif cnn == 's3d':
            model = s3d(weights='DEFAULT')
        else:
            raise ValueError(f'Unknown value for `cnn`: {cnn}')

        if cnn in ['rn18', 'rn34', 'rn50']:
            self.model = nn.Sequential(*list(model.children())[:-2])
        if cnn in ['s3d']:
            model.classifier = nn.Identity()
            self.model = model

        # Freeze layers if requested.
        for layer_index in range(freeze_layers):
            for param in self.model[layer_index].parameters(True):
                param.requires_grad = False

        # ResNet-18 & ResNet-34 output 512-dim features; ResNet-50 output 2048-dim.
        if embed_size != 512 and (cnn in ['rn18','rn34']):
            self.pointwise_conv = nn.Conv2d(512, embed_size, 1)
        else:
            self.pointwise_conv = nn.Identity()

        if cnn == 'rn50':
            self.pointwise_conv = nn.Conv2d(2048, embed_size, 1)

        self.avg_pool = F.adaptive_avg_pool2d

    def forward(self, rgb_clip):
        """Extract features from the RGB images."""
        b, t, c, h, w = rgb_clip.size()
        # Process all sequential data in parallel as a large mini-batch.
        rgb_clip = rgb_clip.view(b*t, c, h, w)

        features = self.model(rgb_clip)
        # features = self.pointwise_conv(features)

        # features = self.avg_pool(features, 1).squeeze()
        # features = features.view(b, t, -1)
        return features


class LayerNormalization(nn.Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.shape[1] == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu) / (sigma + self.eps)
        ln_out = ln_out * self.a_2 + self.b_2
        return ln_out


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        # q.size(): [nh*b x T x d_k]
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        Args:
            n_head:     số head
            d_model:    kích thước đầu vào (embedding size) = n_head * d_v
            d_k:        chiều cho mỗi head (Query/Key)
            d_v:        chiều cho mỗi head (Value)
            dropout:    dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # 3 lớp Linear để gộp cho Q, K, V
        self.W_Q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_head * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_model=d_model, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # Khởi tạo trọng số
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_V.weight)

    def forward(self, q, k, v):
        residual = q

        B_q, T_q, _ = q.size()
        B_k, T_k, _ = k.size()
        B_v, T_v, _ = v.size()

        # Projection
        q_proj = self.W_Q(q)  # (B_q, T_q, n_head*d_k)
        k_proj = self.W_K(k)  # (B_k, T_k, n_head*d_k)
        v_proj = self.W_V(v)  # (B_v, T_v, n_head*d_v)

        # Reshape Q
        q_proj = q_proj.view(B_q, T_q, self.n_head, self.d_k)
        q_proj = q_proj.permute(0, 2, 1, 3).contiguous()
        q_proj = q_proj.view(B_q * self.n_head, T_q, self.d_k)

        # Reshape K
        k_proj = k_proj.view(B_k, T_k, self.n_head, self.d_k)
        k_proj = k_proj.permute(0, 2, 1, 3).contiguous()
        k_proj = k_proj.view(B_k * self.n_head, T_k, self.d_k)

        # Reshape V
        v_proj = v_proj.view(B_v, T_v, self.n_head, self.d_v)
        v_proj = v_proj.permute(0, 2, 1, 3).contiguous()
        v_proj = v_proj.view(B_v * self.n_head, T_v, self.d_v)

        # Scaled Dot-Product
        outputs = self.attention(q_proj, k_proj, v_proj)
        # outputs shape: (B_q*n_head, T_q, d_v)

        # Gộp các head
        outputs = outputs.view(B_q, self.n_head, T_q, self.d_v)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        outputs = outputs.view(B_q, T_q, self.n_head * self.d_v)  # = d_model

        outputs = self.dropout(outputs)

        return outputs + residual


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_hid, d_inner_hid, dropout=0.1, layer_norm=True):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_hid) if layer_norm else nn.Identity()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(2,1)))
        output = self.w_2(output).transpose(1,2)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class DecoderBlock(nn.Module):
    """
    DecoderBlock refactor
    """
    def __init__(self, 
                 input_size,
                 n_head, 
                 d_k, 
                 d_v,
                 dropout=0.1, 
                 layer_norm=True,
                 ffn_inner=2048, lcfe=False):
        super(DecoderBlock, self).__init__()

        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=input_size,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )
        # Thêm FFN sau self-attention
        self.ffn = PositionwiseFeedForward(
            d_hid=input_size,
            d_inner_hid=ffn_inner,
            dropout=dropout,
            layer_norm=layer_norm
        )

        self.norm = LayerNormalization(input_size) if layer_norm else nn.Identity()
        self.lcfe = LCFE(input_size) if lcfe else nn.Identity()

    def forward(self, enc_input):
        # Self-attention thuần
        enc_output = self.slf_attn(enc_input, enc_input, enc_input)
        if isinstance(self.lcfe, nn.Identity):
            enc_output = self.norm(enc_output)
        else:
            lc_ft = self.lcfe(enc_input)
            enc_output = self.norm(enc_output + lc_ft)
        # FFN
        enc_output = self.ffn(enc_output)
        return enc_output


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention refactor:
      - Tách [CLS] từ x làm Q,
      - Tokens còn lại trong context làm K/V,
      - Sau đó ghép Q (đã updated) với phần còn lại của x.
    """
    def __init__(self, input_size, hidden_size, inner_hidden_size, n_heads, dropout, layer_norm=True, lcfe=False):
        super().__init__()
        self.cross_attn = MultiHeadAttention(
            n_head=n_heads,
            d_model=input_size,
            d_k=hidden_size // n_heads,
            d_v=hidden_size // n_heads,
            dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            d_hid=input_size,
            d_inner_hid=inner_hidden_size,
            dropout=dropout,
            layer_norm=layer_norm
        )
        self.norm = LayerNormalization(input_size) if layer_norm else nn.Identity()
        self.lcfe = LCFE(input_size) if lcfe else nn.Identity()

    def forward(self, x, context):
        """
        x: (B, T, D) - đã có CLS ở đầu,
        context: (B, T, D) - cũng đã có CLS hoặc không, nhưng ta chỉ xài context[:, 1:] thôi.
        """
        # Q là CLS token
        cls_q = x[:, :1, :]          # (B, 1, D)
        # K, V là phần tokens (bỏ CLS) từ context
        kv = context[:, 1:, :]       # (B, T-1, D) giả sử T >= 2

        # Tính cross-attention
        updated_cls = self.cross_attn(cls_q, kv, kv)  # (B, 1, D)

        # Ghép CLS đã update với các token còn lại của x
        out = torch.cat([updated_cls, x[:, 1:, :]], dim=1)  # (B, T, D)

        if isinstance(self.lcfe, nn.Identity):
            out = self.norm(out)
        else:
            lc_ft = self.lcfe(x)
            out = self.norm(out + lc_ft)
        # FFN trên toàn sequence
        out = self.feed_forward(out)

        return out

class SpatialAttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size, inner_hidden_size, n_heads, dropout=0.1, layer_norm=True):
        super().__init__()
        # Self-attention block (DecoderBlock)
        self.self_attn = DecoderBlock(
            input_size=input_size,
            n_head=n_heads,
            d_k=hidden_size // n_heads,
            d_v=hidden_size // n_heads,
            dropout=dropout,
            layer_norm=layer_norm,
            ffn_inner=inner_hidden_size,
            lcfe=False
        )
        # Cross-attention block
        self.cross_attn = CrossAttentionBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            inner_hidden_size=inner_hidden_size,
            n_heads=n_heads,
            dropout=dropout,
            layer_norm=layer_norm,
            lcfe=False
        )

    def forward(self, x, context):
        # 1) Self-attention
        x = self.self_attn(x)
        # 2) Cross-attention
        x = self.cross_attn(x, context)
        return x
    
class SpatialCrossAttention(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, sequence_size,
                 inner_hidden_factor=2, layer_norm=True, dropout=0.1):
        super().__init__()

        input_sizes = [hidden_size] * len(n_heads)
        input_sizes[0] = input_size  # block đầu tiên có thể khác
        hidden_sizes = [hidden_size] * len(n_heads)

        self.position_encoding = PositionEncoding(
            n_positions=sequence_size,
            hidden_size=hidden_size
        )

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'viewl': SpatialAttentionBlock(
                    inp_size, hid_size, int(hid_size * inner_hidden_factor),
                    n_head, dropout=dropout, layer_norm=layer_norm
                ),
                'viewcl': SpatialAttentionBlock(
                    inp_size, hid_size, int(hid_size * inner_hidden_factor),
                    n_head, dropout=dropout, layer_norm=layer_norm
                ),
                'viewcr': SpatialAttentionBlock(
                    inp_size, hid_size, int(hid_size * inner_hidden_factor),
                    n_head, dropout=dropout, layer_norm=layer_norm
                ),
                'viewr': SpatialAttentionBlock(
                    inp_size, hid_size, int(hid_size * inner_hidden_factor),
                    n_head, dropout=dropout, layer_norm=layer_norm
                )
            })
            for i, (inp_size, hid_size, n_head) in enumerate(zip(input_sizes, hidden_sizes, n_heads))
        ])

    def forward(self, xl, xc, xr, cls_token_encodings=False):
        # Thêm positional embedding (và CLS token nếu cần)
        xl = self.position_encoding(
            xl,
            is_view=True, view_id=0,
            use_cls=cls_token_encodings
        )
        xc = self.position_encoding(
            xc,
            is_view=True, view_id=1,
            use_cls=cls_token_encodings
        )
        xr = self.position_encoding(
            xr,
            is_view=True, view_id=2,
            use_cls=cls_token_encodings
        )

        for layer in self.layers:
            xl = layer['viewl'](xl, context=xc)
            xc = layer['viewcl'](xc, context=xl)
            xc = layer['viewcr'](xc, context=xr)
            xr = layer['viewr'](xr, context=xc)

        return xl, xc, xr

class TemporalAttentionBlock(nn.Module):
    """
    Block cho 1 stream:
      - Self-attention thuần trên x,
      - Cross-attention (tách [CLS] ở CrossAttentionBlock).
    """
    def __init__(self, input_size, hidden_size, inner_hidden_size, n_heads, dropout=0.1, layer_norm=True):
        super().__init__()
        # Self-attention block (DecoderBlock)
        self.self_attn = DecoderBlock(
            input_size=input_size,
            n_head=n_heads,
            d_k=hidden_size // n_heads,
            d_v=hidden_size // n_heads,
            dropout=dropout,
            layer_norm=layer_norm,
            ffn_inner=inner_hidden_size,
            lcfe=False
        )
        # Cross-attention block
        self.cross_attn = CrossAttentionBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            inner_hidden_size=inner_hidden_size,
            n_heads=n_heads,
            dropout=dropout,
            layer_norm=layer_norm,
            lcfe=False
        )

    def forward(self, x, context):
        # 1) Self-attention
        x = self.self_attn(x)
        # 2) Cross-attention
        x = self.cross_attn(x, context)
        return x

class TemporalCrossAttention(nn.Module):
    """
    Ví dụ module để cross attention giữa 2 stream.
    Ở đây ta chỉ gọi StreamAttentionBlock, 
    và [CLS] token được quản lý/nối từ PositionEncoding bên ngoài (nếu có).
    """
    def __init__(self, input_size, hidden_size, n_heads, sequence_size,
                 inner_hidden_factor=2, layer_norm=True, dropout=0.1):
        super().__init__()

        input_sizes = [hidden_size] * len(n_heads)
        input_sizes[0] = input_size  # block đầu tiên có thể khác
        hidden_sizes = [hidden_size] * len(n_heads)

        self.position_encoding = PositionEncoding(
            n_positions=sequence_size,
            hidden_size=hidden_size
        )

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'stream1': TemporalAttentionBlock(
                    inp_size, hid_size, hid_size * inner_hidden_factor,
                    n_head, dropout=dropout, layer_norm=layer_norm
                ),
                'stream2': TemporalAttentionBlock(
                    inp_size, hid_size, hid_size * inner_hidden_factor,
                    n_head, dropout=dropout, layer_norm=layer_norm
                )
            })
            for i, (inp_size, hid_size, n_head) in enumerate(zip(input_sizes, hidden_sizes, n_heads))
        ])

    def forward(self, x1, x2, cls_token_encodings=False):
        # Thêm positional embedding (và CLS token nếu cần)
        x1 = self.position_encoding(
            x1,
            is_stream=True, stream_id=0,
            use_cls=cls_token_encodings
        )
        x2 = self.position_encoding(
            x2,
            is_stream=True, stream_id=1,
            use_cls=cls_token_encodings
        )

        for layer in self.layers:
            x1 = layer['stream1'](x1, context=x2)
            x2 = layer['stream2'](x2, context=x1)

        return x1, x2

class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, num_classes)
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        return self.fc(self.dropout(x))


class PositionEncoding(nn.Module):
    """
    Thêm [CLS] token và positional encoding.
    """
    def __init__(self, n_positions, hidden_size, patch_size=1, n_streams=2, n_views=3):
        super().__init__()
        self.enc = nn.Embedding(n_positions, hidden_size)

        # Calculate the position encoding
        position_enc = np.array([
            [pos / np.power(10000, 2.0*(j//2)/hidden_size) for j in range(hidden_size)]
            if pos != 0 else np.zeros(hidden_size)
            for pos in range(n_positions)
        ])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
        self.enc.weight = nn.Parameter(
            torch.from_numpy(position_enc).float(),
            requires_grad=True
        )

        self.cls_tokens_stream = nn.Parameter(
            torch.zeros(n_streams, 1, hidden_size)
        )
        nn.init.normal_(self.cls_tokens_stream, std=0.02)

        self.cls_tokens_view = nn.Parameter(
            torch.zeros(n_views, 1, hidden_size)
        )
        nn.init.normal_(self.cls_tokens_view, std=0.02)

        self.patch_size = patch_size

    def forward(self, x,
                is_stream=False, stream_id=0,
                is_view=False, view_id=0,
                use_cls=False):
        if is_stream:
            B, T, D = x.size()
        elif is_view:
            K, C, H, W = x.size()
            P = self.patch_size

            # Split image into patches
            assert H % P == 0 and W % P == 0, "Height and Width must be divisible by patch size."
            N = (H // P) * (W // P)  # Number of patches
            x = x.view(K, C, H // P, P, W // P, P)  # Reshape into patches
            x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # Move patch size dimensions next to channels
            x = x.view(K, N, P * P * C)  # Flatten patches into vectors (B*T, N, P^2 * C)

        if use_cls:
            if is_stream:
                cls_token = self.cls_tokens_stream[stream_id:stream_id+1]
                cls_token = cls_token.expand(B, -1, -1)  # (B, 1, D)
                
            elif is_view:
                cls_token = self.cls_tokens_view[view_id:view_id+1]
                cls_token = cls_token.expand(K, -1, -1)  # (B,1,P^2*C)
            else:
                cls_token = torch.zeros(1, 1, D, device=x.device)

            x = torch.cat([cls_token, x], dim=1)

        size = x.size(1)  # T or T+1
        indices = torch.arange(size, device=x.device, dtype=torch.long)
        pe = self.enc(indices)  # (size, D)
        x = x + pe
        return x

class GlossAwareFusion(nn.Module):
    def __init__(self, visual_in_dim=512, proj_dim=300):
        super(GlossAwareFusion, self).__init__()
        # Project visual features from 512 to 300
        self.visual_proj = nn.Linear(visual_in_dim, proj_dim)
        
    def forward(self, visual_features, gloss_features):
        """
        visual_features: Tensor shape [B, T, 3, 512] từ 3 encoder (3 view)
        gloss_features: Tensor shape [B, 300] (từng sample gloss embedding)
        
        Returns:
            aggregated_features: Tensor shape [B, T, 300]
        """
        B, T, num_views, _ = visual_features.size()
        
        # Project visual features: [B, T, 3, 300]
        visual_proj = self.visual_proj(visual_features)
        
        # Normalize visual features along last dimension
        visual_norm = F.normalize(visual_proj, p=2, dim=-1)
        
        # Normalize gloss features (shape [B, 300])
        gloss_norm = F.normalize(gloss_features, p=2, dim=-1)
        # Unsqueeze gloss to shape [B, 1, 1, 300] để so sánh với từng frame, từng view
        gloss_norm = gloss_norm.unsqueeze(1).unsqueeze(2)
        
        # Tính cosine similarity giữa gloss và mỗi visual feature: kết quả có shape [B, T, 3]
        sim = (visual_norm * gloss_norm).sum(dim=-1)
        
        # Tính trọng số bằng softmax theo chiều view
        weights = F.softmax(sim, dim=2)  # [B, T, 3]
        # Expand weights sang [B, T, 3, 1] để nhân với visual features
        weights = weights.unsqueeze(-1)
        
        # Tính weighted sum trên chiều view
        aggregated_ft = (weights * visual_proj).sum(dim=2)  # [B, T, 300]
        
        return aggregated_ft
    
class VisualOnlyFusion(nn.Module):
    def __init__(self, visual_in_dim=512, proj_dim=300):
        super(VisualOnlyFusion, self).__init__()
        # Project từ 512 xuống 300 cho mỗi view
        self.visual_proj = nn.Linear(visual_in_dim, proj_dim)
        # Một lớp gating học được: nhận concat của 3 view đã được project và cho ra 3 logit
        self.gating_fc = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 3)
        )
        
    def forward(self, visual_features):
        """
        visual_features: Tensor [B, T, 3, 512]
        """
        B, T, num_views, _ = visual_features.size()
        # Project mỗi view: [B, T, 3, 300]
        visual_proj = self.visual_proj(visual_features)
        
        # Cũng có thể chuẩn hóa nếu cần
        visual_norm = F.normalize(visual_proj, p=2, dim=-1)
        
        # Tính gating logits cho mỗi frame:
        # Flatten chiều view: [B, T, 3*300]
        flat = visual_norm.view(B, T, -1)
        gating_logits = self.gating_fc(flat)  # [B, T, 3]
        
        # Tính trọng số qua softmax theo chiều view
        weights = F.softmax(gating_logits, dim=-1).unsqueeze(-1)  # [B, T, 3, 1]
        
        # Weighted sum trên chiều view
        aggregated_ft = (weights * visual_proj).sum(dim=2)  # [B, T, 300]
        return aggregated_ft
