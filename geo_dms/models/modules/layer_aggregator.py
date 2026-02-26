# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerAggregator(nn.Module):
    def __init__(self, num_layers=4, channel_dim=768):
        super().__init__()
        self.num_layers = num_layers
        
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, features_list):
        """
        Args:
            features_list: List[Tensor], 每个 Tensor 形状为 [B, C, H, W]
        Returns:
            fused_feature: [B, C, H, W]
        """
        if len(features_list) != self.num_layers:
            features_list = features_list[-self.num_layers:]
            
        normalized_weights = F.softmax(self.layer_weights, dim=0)
        
        stacked_feats = torch.stack(features_list, dim=0)
        
        weights_reshaped = normalized_weights.view(-1, 1, 1, 1, 1)
        weighted_feats = stacked_feats * weights_reshaped
        
        fused_feature = torch.sum(weighted_feats, dim=0)
        
        return fused_feature, normalized_weights

    def get_layer_importance(self):
        return F.softmax(self.layer_weights, dim=0).detach().cpu().numpy()


class LayerAggregatorV2(nn.Module):
    def __init__(self, num_layers=4, channel_dim=384): 
        super().__init__()
        self.num_layers = num_layers
        
        self.pre_norm = nn.LayerNorm(channel_dim * num_layers)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_layers * channel_dim, channel_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, channel_dim),
            nn.ReLU(inplace=True)
        )
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(channel_dim, channel_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim // 4, channel_dim, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, features_list):
        if len(features_list) != self.num_layers:
            features_list = features_list[-self.num_layers:]
            
        concat_feats = torch.cat(features_list, dim=1) # [B, 4*C, H, W]
        
        if torch.isinf(concat_feats).any() or torch.isnan(concat_feats).any():
            concat_feats = torch.nan_to_num(concat_feats, nan=0.0, posinf=1e4, neginf=-1e4)

        B, C_all, H, W = concat_feats.shape
        concat_feats = concat_feats.flatten(2).transpose(1, 2) # [B, HW, C_all]
        concat_feats = self.pre_norm(concat_feats)
        concat_feats = concat_feats.transpose(1, 2).view(B, C_all, H, W)

        fused_feat = self.fusion_conv(concat_feats)
        
        weights = self.channel_att(fused_feat)
        
        final_feat = fused_feat * weights
        
        return final_feat, weights.mean() 

    def get_layer_importance(self):
        return [0.25] * self.num_layers

class LayerAggregatorV3(nn.Module):
    def __init__(self, num_layers=4, channel_dim=768):
        super().__init__()
        self.num_layers = num_layers
        self.channel_dim = channel_dim
        
        self.input_norm = nn.LayerNorm(channel_dim)
        
        self.layer_embed = nn.Parameter(torch.zeros(num_layers, channel_dim))
        nn.init.normal_(self.layer_embed, std=0.02)
        
        self.attn = nn.MultiheadAttention(embed_dim=channel_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(channel_dim)
        
        self.weight_gen = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // 4), # 降维 bottleneck
            nn.ReLU(),
            nn.Linear(channel_dim // 4, 1) 
        )

    def forward(self, features_list):
        if len(features_list) != self.num_layers:
            features_list = features_list[-self.num_layers:]
            
        B, C, H, W = features_list[0].shape
        
        stacked = torch.stack(features_list, dim=1) # [B, L, C, H, W]
        
        if torch.isinf(stacked).any() or torch.isnan(stacked).any():
             stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e4, neginf=-1e4)
             
        global_desc = stacked.mean(dim=[-2, -1]) 
        
        global_desc = self.input_norm(global_desc)
        
        global_desc = global_desc + self.layer_embed.unsqueeze(0)
        
        desc_norm = self.norm(global_desc)
        attn_out, _ = self.attn(desc_norm, desc_norm, desc_norm)
        global_desc = global_desc + attn_out
        
        logits = self.weight_gen(global_desc)
        weights = F.softmax(logits, dim=1).unsqueeze(-1).unsqueeze(-1) # [B, L, 1, 1, 1]
        
        fused_feat = (stacked * weights).sum(dim=1) # [B, C, H, W]
        
        return fused_feat, weights.mean(dim=[0, 2, 3, 4])

class LastLayerAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_list):
        return features_list[-1], None