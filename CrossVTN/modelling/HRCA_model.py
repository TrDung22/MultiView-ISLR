import torch
from torch import nn

from .HRCA_ultis import FeatureExtractor, LinearClassifier, TemporalCrossAttention, GlossAwareFusion, VisualOnlyFusion
# from .vtn_utils import SelfAttention
import torch.nn.functional as F
from pytorch_lightning.utilities.migration import pl_legacy_patch
import fasttext

class HRCA(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers_stream=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: HRCA")
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.feature_extractor_heatmap = FeatureExtractor(cnn, embed_size, freeze_layers)
        self.feature_extractor_rgb = FeatureExtractor(cnn, embed_size, freeze_layers)

        num_attn_features = embed_size*2
        # self.hmap_attn = SelfAttention(num_attn_features, num_attn_features,
        #                                             [num_heads] * num_layers_stream,
        #                                             self.sequence_length, layer_norm=True, dropout=dropout)
        # self.rgb_attn = SelfAttention(num_attn_features, num_attn_features,
        #                                             [num_heads] * num_layers_stream,
        #                                             self.sequence_length, layer_norm=True, dropout=dropout)
        self.cross_attention = TemporalCrossAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers_stream,
                                                    self.sequence_length+1, layer_norm=True, dropout=dropout)
        self.classifier = LinearClassifier(num_attn_features, num_classes, dropout)
        self.num_attn_features = num_attn_features
        self.dropout = dropout
        self.num_classes = num_classes
        self.relu = F.relu

    def reset_head(self, num_classes):
        self.classifier = LinearClassifier(self.num_attn_features, num_classes, self.dropout)
        print("Reset to ", num_classes)

    def forward_temporal(self, hmap_ft, rgb_ft):
        hmap_ft, rgb_ft = self.cross_attention(hmap_ft, rgb_ft, cls_token_encodings=True)
        cls_hmap = hmap_ft[:, 0, :]
        cls_rgb = rgb_ft[:, 0, :]
        hmap_logits = self.classifier(cls_hmap)
        rgb_logits = self.classifier(cls_rgb)
        y =  hmap_logits + rgb_logits
        return y, hmap_logits, rgb_logits

    def forward(self, heatmap=None, rgb=None, **kwargs):
        """Extract the image feature vectors."""
        # b, t, x, c, h, w = rgb.size()
        b, t, c, h, w = rgb.size()
        rgb_feature = self.feature_extractor_rgb(rgb.view(b, t, c, h, w)).view(b, t, -1)
        heatmap_feature = self.feature_extractor_heatmap(heatmap.view(b, t, c, h, w)).view(b, t, -1)
        # heatmap_feature = F.interpolate(heatmap_feature, scale_factor=2, mode='linear', align_corners=True)

        heatmap_feature, rgb_feature = self.cross_attention(heatmap_feature, rgb_feature, cls_token_encodings=True)
        cls_heatmap = heatmap_feature[:, 0, :]
        cls_rgb = rgb_feature[:, 0, :]

        # heatmap_feature = self.hmap_attn(heatmap_feature)
        # rgb_feature = self.rgb_attn(rgb_feature)
        # y = torch.cat([heatmap_feature, rgb_feature], dim=-1)
        # y = self.classifier(y)
        y_hmap = self.classifier(cls_heatmap)
        y_rgb = self.classifier(cls_rgb)
        y = y_hmap + y_rgb
        return {'logits': y}

class HRMSSCA(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers_stream=2, num_layers_view=1, embed_size=512, sequence_length=16, cnn='rn34', freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: HRMSSCA")
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.feature_extractor_left = FeatureExtractor(cnn, embed_size, freeze_layers)
        self.feature_extractor_center = FeatureExtractor(cnn, embed_size, freeze_layers)
        self.feature_extractor_right = FeatureExtractor(cnn, embed_size, freeze_layers)

        self.gloss_embbeder = fasttext.load_model("checkpoints/cc.vi.300.bin")

        num_attn_features = 300
        # self.MSS = SpatialCrossAttention(num_attn_features, num_attn_features,
        #                                             [num_heads] * num_layers_view,
        #                                             patchNum+1, layer_norm=True, dropout=dropout)

        self.temporal_decoder_left = SelfAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers_stream,
                                                    self.sequence_length, layer_norm=True, dropout=dropout)
        self.temporal_decoder_center = SelfAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers_stream,
                                                    self.sequence_length, layer_norm=True, dropout=dropout)
        self.temporal_decoder_right = SelfAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers_stream,
                                                    self.sequence_length, layer_norm=True, dropout=dropout)

        self.classifier = LinearClassifier(num_attn_features, num_classes, dropout)
        self.num_attn_features = num_attn_features
        self.dropout = dropout
        self.num_classes = num_classes

        self.avg_pool = F.adaptive_avg_pool2d
        self.gloss_aware_fusion = GlossAwareFusion(visual_in_dim=embed_size, proj_dim=300)
        self.visual_only_fusion = VisualOnlyFusion(visual_in_dim=embed_size, proj_dim=300)
        

    def forward_features(self, rgb_left=None, rgb_center=None, rgb_right=None, hmap_left=None, hmap_right=None, hmap_center=None):
    
        return None

    def forward(self, rgb_left=None, rgb_center=None, rgb_right=None, gloss=None, hmap_left=None, hmap_right=None, hmap_center=None):
        b, t, c, h, w = rgb_center.size()

        rgb_left_feature = self.feature_extractor_left(rgb_left.view(b, t, c, h, w))
        rgb_right_feature = self.feature_extractor_center(rgb_right.view(b, t, c, h, w))
        rgb_center_feature = self.feature_extractor_right(rgb_center.view(b, t, c, h, w))

        rgb_left_feature = self.avg_pool(rgb_left_feature, 1).squeeze()
        rgb_center_feature = self.avg_pool(rgb_center_feature, 1).squeeze()
        rgb_right_feature = self.avg_pool(rgb_right_feature, 1).squeeze()

        rgb_left_ft = rgb_left_feature.view(b,t,-1)
        rgb_center_ft = rgb_center_feature.view(b,t,-1)
        rgb_right_ft = rgb_right_feature.view(b,t,-1)

        visual_ft = torch.stack([rgb_left_ft, rgb_center_ft, rgb_right_ft], dim=2)

        if gloss is not None:
            gloss_numpy_list = [
            self.gloss_embbeder.get_word_vector(g_str)  # -> np.array([300], dtype=float32)
            for g_str in gloss
            ]
            # Chuyển tất cả sang torch.Tensor rồi stack [b, embedding_dim]
            gloss_ft = torch.stack([
                torch.from_numpy(vec_np)
                for vec_np in gloss_numpy_list
            ], dim=0)  # -> shape [b, 300]

            # Đưa gloss_ft lên cùng device với visual_ft (thường là GPU)
            gloss_ft = gloss_ft.to(visual_ft.device)

            gloss_visual_fusion_ft = self.gloss_aware_fusion(visual_ft,gloss_ft) #[B,T,300]
            visual_fusion_ft = self.visual_only_fusion(visual_ft)
            y = self.temporal_decoder_center(gloss_visual_fusion_ft)
        else:
            gloss_visual_fusion_ft = None
            visual_fusion_ft = self.visual_only_fusion(visual_ft)
            y = self.temporal_decoder_center(visual_fusion_ft)

        y = self.classifier(y.mean(1))
        return {'logits': y,
                'gloss_visual_fusion_ft': gloss_visual_fusion_ft,
                'visual_fusion_ft': visual_fusion_ft}
                # 'rgb_left_ft': rgb_left_feature,
                # 'rgb_center_ft': rgb_center_feature,
                # 'rgb_right_ft': rgb_right_feature}
