# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .transformer import Attention, LayerNorm2d, MLP

class PoseGuidedFusionModule(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads=8, 
        num_scales=3,  
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        num_kpts=53
    ):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        
        self.norm_feat = nn.LayerNorm(embed_dim)

        self.downsamplers = nn.ModuleList()
        for _ in range(num_scales - 1):
            self.downsamplers.append(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2)
            )

        self.attn_layers = nn.ModuleList([
            Attention(
                embed_dims=embed_dim,
                num_heads=num_heads,
                query_dims=embed_dim,
                key_dims=embed_dim,
                value_dims=embed_dim,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate
            ) for _ in range(num_scales)
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

        self.norm_attn = nn.LayerNorm(embed_dim)
        self.ffn = MLP(
            input_dim=embed_dim, 
            hidden_dim=int(embed_dim * mlp_ratio), 
            output_dim=embed_dim, 
            num_layers=2
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # GeoAdapter
        input_dim = num_kpts * 2
        self.geo_adapter = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_scales * 3) 
        )
        nn.init.constant_(self.geo_adapter[-1].weight, 0)
        nn.init.constant_(self.geo_adapter[-1].bias, -2.0)

    def generate_adaptive_bias(self, face_kpts, H, W, scale_idx, adaptive_params):
        B = face_kpts.shape[0]
        device = face_kpts.device
        
        face_kpts = torch.clamp(face_kpts, 0.0, 1.0)
        
        params = adaptive_params[:, scale_idx, :]
        
        sigma = torch.sigmoid(params[:, 0]).view(B, 1, 1) * 0.9 + 0.1
        
        offset = torch.tanh(params[:, 1:]).view(B, 1, 2) * 0.2
        face_center = face_kpts.mean(dim=1, keepdim=True) + offset
        
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid = torch.stack([x / (W - 1), y / (H - 1)], dim=-1)
        grid = grid.reshape(1, H * W, 2)
        
        dist_sq = (face_center - grid).pow(2).sum(dim=-1)
        
        bias = - (dist_sq / (2 * sigma.squeeze(-1)**2))
        
        bias = torch.clamp(bias, min=-10000.0) 
        
        return bias.unsqueeze(1)

    def forward(self, query_token, image_features, pose_kpts):
        B = query_token.shape[0]
        fused_feat_sum = 0
        curr_feat = image_features
        
        if torch.isinf(curr_feat).any() or torch.isnan(curr_feat).any():
             curr_feat = torch.nan_to_num(curr_feat, nan=0.0, posinf=1e4, neginf=-1e4)
        self.vis_cache = {"scales": []}
        adaptive_params = self.geo_adapter(pose_kpts.flatten(1))
        adaptive_params = adaptive_params.view(B, self.num_scales, 3)
        
        for i in range(self.num_scales):
            if i > 0:
                curr_feat = self.downsamplers[i-1](curr_feat)
            
            B, C, H, W = curr_feat.shape
            
            feat_flat = curr_feat.flatten(2).transpose(1, 2)
            feat_embed = self.norm_feat(feat_flat)
            attn_bias = self.generate_adaptive_bias(pose_kpts, H, W, i, adaptive_params)
            params = adaptive_params[:, i, :]
            vis_sigma = torch.sigmoid(params[:, 0]) * 0.9 + 0.1
            

            self.vis_cache["scales"].append({
                "scale_idx": i,
                "attn_bias": attn_bias.detach().cpu(),  
                "sigma": vis_sigma.detach().cpu(),      
                "bias_scale": self.scale_weights[i].detach().cpu(),
                "feat_size": (H, W)
            })

            scale_out = self.attn_layers[i](
                q=query_token, 
                k=feat_embed, 
                v=feat_embed, 
                attn_mask=attn_bias
            )
            
            weights = F.softmax(self.scale_weights, dim=0)
            fused_feat_sum += scale_out * weights[i]

        x = self.norm_attn(query_token + fused_feat_sum)
        x = x + self.ffn(x)
        x = self.norm_ffn(x)
        
        return x


# V2: 多极姿态引导融合 (PG-MANO)
class PoseGuidedMultipoleFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_scales=3, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        
        self.norm_feat = nn.LayerNorm(embed_dim)
        self.downsamplers = nn.ModuleList()
        for _ in range(num_scales - 1):
            self.downsamplers.append(nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2))

        self.attn_layers = nn.ModuleList([
            Attention(
                embed_dims=embed_dim, 
                num_heads=num_heads, 
                query_dims=embed_dim, 
                key_dims=embed_dim, 
                value_dims=embed_dim, 
                attn_drop=attn_drop_rate, 
                proj_drop=drop_rate
            ) for _ in range(num_scales)
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.ffn = MLP(input_dim=embed_dim, hidden_dim=int(embed_dim * mlp_ratio), output_dim=embed_dim, num_layers=2)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        
        self.sigmas = [0.05, 0.15, 0.3] 

    def generate_multiscale_bias(self, kpts, H, W, device, scale_idx):
        B, N, _ = kpts.shape
        sigma = self.sigmas[min(scale_idx, len(self.sigmas)-1)]
        
        kpts = torch.clamp(kpts, 0.0, 1.0)
        
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid = torch.stack([x / (W - 1), y / (H - 1)], dim=-1)
        grid = grid.reshape(1, H * W, 2)
        
        dist_sq_all = (kpts.unsqueeze(2) - grid.unsqueeze(1)).pow(2).sum(dim=-1)
        
        min_dist_sq, _ = dist_sq_all.min(dim=1, keepdim=True)
        
        bias = - (min_dist_sq / (2 * sigma**2))
        
        bias = torch.clamp(bias, min=-10000.0)
        
        return bias.unsqueeze(1)

    def forward(self, query_token, image_features, pose_kpts):
    
        
        fused_feat_sum = 0
        curr_feat = image_features
        
        if torch.isinf(curr_feat).any():
             curr_feat = torch.nan_to_num(curr_feat, posinf=1e4, neginf=-1e4)
        
        for i in range(self.num_scales):
            if i > 0:
                curr_feat = self.downsamplers[i-1](curr_feat)
            
            B, C, H, W = curr_feat.shape
            
            feat_flat = curr_feat.flatten(2).transpose(1, 2)
            feat_embed = self.norm_feat(feat_flat)
            
            attn_bias = self.generate_multiscale_bias(pose_kpts, H, W, curr_feat.device, i)
            
            scale_out = self.attn_layers[i](
                q=query_token, 
                k=feat_embed, 
                v=feat_embed, 
                attn_mask=attn_bias
            )
            
            weights = F.softmax(self.scale_weights, dim=0)
            fused_feat_sum += scale_out * weights[i]

        x = self.norm_attn(query_token + fused_feat_sum)
        x = x + self.ffn(x)
        x = self.norm_ffn(x)
        return x

class PoseGuidedAdaptiveFusion(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads=8, 
        num_scales=4,  
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        num_kpts=53,
        backbone_dim=1280
    ):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        
        self.layer_adapters = nn.ModuleList()
        self.layer_adapters.append(nn.Sequential(
            nn.Conv2d(backbone_dim, embed_dim, 1), 
            nn.GroupNorm(1, embed_dim), nn.GELU()))
        self.layer_adapters.append(nn.Sequential(
            nn.Conv2d(backbone_dim, embed_dim, 3, stride=2, padding=1), 
            nn.GroupNorm(1, embed_dim), nn.GELU()))
        self.layer_adapters.append(nn.Sequential(
            nn.Conv2d(backbone_dim, embed_dim, 3, stride=4, padding=1), 
            nn.GroupNorm(1, embed_dim), nn.GELU()))
        self.layer_adapters.append(nn.Sequential(
            nn.Conv2d(backbone_dim, embed_dim, 3, stride=8, padding=1), 
            nn.GroupNorm(1, embed_dim), nn.GELU()))

        self.global_downsamplers = nn.ModuleList()
        self.global_downsamplers.append(nn.Identity()) # Scale 0
        self.global_downsamplers.append(nn.AvgPool2d(kernel_size=2, stride=2)) # Scale 1
        self.global_downsamplers.append(nn.AvgPool2d(kernel_size=4, stride=4)) # Scale 2
        self.global_downsamplers.append(nn.AvgPool2d(kernel_size=8, stride=8)) # Scale 3

        self.attn_layers = nn.ModuleList([
            Attention(
                embed_dims=embed_dim,
                num_heads=num_heads,
                query_dims=embed_dim,
                key_dims=embed_dim,
                value_dims=embed_dim,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate
            ) for _ in range(num_scales)
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.ffn = MLP(input_dim=embed_dim, hidden_dim=int(embed_dim * mlp_ratio), output_dim=embed_dim, num_layers=2)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.num_regions = 4 
        self.params_per_region = 4 
        self.geo_adapter = nn.Sequential(
            nn.Linear(num_kpts * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, num_scales * self.num_regions * self.params_per_region) 
        )
        nn.init.constant_(self.geo_adapter[-1].weight, 0)
        nn.init.constant_(self.geo_adapter[-1].bias, 0.0) 

    def generate_adaptive_bias(self, pose_kpts, H, W, scale_idx, adaptive_params):
        B = pose_kpts.shape[0]
        device = pose_kpts.device
        
        regions_pts = {
            'face':   pose_kpts[:, 0:5, :],
            'body':   pose_kpts[:, 5:11, :],
            'r_hand': pose_kpts[:, 11:34, :],
            'l_hand': pose_kpts[:, 34:55, :]
        }
        
        current_params = adaptive_params[:, scale_idx, :].view(B, self.num_regions, 4)
        
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid = torch.stack([x / (W - 1), y / (H - 1)], dim=-1) # [H, W, 2]
        grid = grid.view(1, H * W, 2)
        
        total_prob_map = torch.zeros((B, H * W), device=device)
        
        region_keys = ['face', 'body', 'r_hand', 'l_hand']
        independent_maps = {}
        
        for i, key in enumerate(region_keys):
            pts = regions_pts[key] # [B, N_pts, 2]
            pts = torch.clamp(pts, 0.0, 1.0)
            center = pts.mean(dim=1, keepdim=True) # [B, 1, 2]
            
            r_params = current_params[:, i, :] # [B, 4]
            
            sigma = torch.sigmoid(r_params[:, 0]).view(B, 1, 1) * 0.45 + 0.05
            
            weight = torch.sigmoid(r_params[:, 1]).view(B, 1, 1)
            
            offset = torch.tanh(r_params[:, 2:]).view(B, 1, 2) * 0.2
            
            final_center = center + offset
            
            dist_sq = (final_center - grid).pow(2).sum(dim=-1) # [B, HW]
            gaussian = torch.exp(-dist_sq / (2 * sigma.squeeze(-1)**2))
            
            total_prob_map = total_prob_map + weight.squeeze(-1) * gaussian
            prob_chunk = weight.squeeze(-1) * gaussian
            total_prob_map = total_prob_map + prob_chunk
            
            independent_maps[key] = prob_chunk.detach().cpu()

        attn_bias = torch.log(total_prob_map + 1e-6)
        
        attn_bias = torch.clamp(attn_bias, min=-20.0)
        
        return attn_bias.unsqueeze(1), independent_maps # [B, 1, HW]

    def forward(self, query_token, feature_list, global_feat, pose_kpts):
        B = query_token.shape[0]
        fused_feat_sum = 0
        self.vis_cache = {"scales": []}
        
        adaptive_params = self.geo_adapter(pose_kpts.flatten(1))
        adaptive_params = adaptive_params.view(B, self.num_scales, -1)
        
        src_indices = [0, 1, 2, 3] 

        for i in range(self.num_scales):
            
            raw_idx = src_indices[i]
            raw_feat = feature_list[raw_idx] 
            
            if raw_feat.dim() == 3:
                L_dim = raw_feat.shape[1]
                S = int(L_dim**0.5) 
                raw_feat = raw_feat.permute(0, 2, 1).reshape(B, -1, S, S)
                
            feat_geometry = self.layer_adapters[i](raw_feat)
            
            feat_semantic = self.global_downsamplers[i](global_feat)
            
            curr_feat = feat_geometry + feat_semantic 
            
            B, C, H, W = curr_feat.shape
            attn_bias, region_maps = self.generate_adaptive_bias(pose_kpts, H, W, i, adaptive_params)
            
            self.vis_cache["scales"].append({
                "scale_idx": i,
                "attn_bias": attn_bias.detach().cpu(),
                "region_maps": region_maps,
                "feat_size": (H, W)
            })
            
            feat_flat = curr_feat.flatten(2).transpose(1, 2) # [B, HW, C]
            scale_out = self.attn_layers[i](query_token, feat_flat, feat_flat, attn_mask=attn_bias)
            
            weights = F.softmax(self.scale_weights, dim=0)
            fused_feat_sum += scale_out * weights[i]

        x = self.norm_attn(query_token + fused_feat_sum)
        x = x + self.ffn(x)
        x = self.norm_ffn(x)
        
        return x

class IdentityFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_token, global_feat=None, feature_list=None, pose_kpts=None, **kwargs):
        if global_feat is None:
            global_feat = kwargs.get('image_features')
            
        gap = global_feat.mean(dim=[-2, -1], keepdim=True)
        
        gap = gap.flatten(2).transpose(1, 2)
        
        out = gap + query_token
        
        return out