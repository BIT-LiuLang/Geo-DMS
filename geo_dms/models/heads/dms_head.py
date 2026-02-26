# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

class GeometryAwareFeatureExtractor(nn.Module):
    def __init__(
        self, 
        input_dim=1024, 
        pose_dim=133, 
        depth=2, 
        dropout=0.1
    ):
        super().__init__()
        self.depth = depth  
        
        self.visual_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU()
        )
        
        if self.depth > 0:
            self.pose_group_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(pose_dim, input_dim), 
                    nn.LayerNorm(input_dim), 
                    nn.GELU()
                ) for _ in range(3) 
            ])
            
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
            self.modal_embed = nn.Embedding(5, input_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, 
                nhead=8, 
                dim_feedforward=input_dim * 4,
                dropout=dropout, 
                activation="gelu", 
                batch_first=True, 
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.layer_scale = nn.Parameter(torch.ones(input_dim) * 1e-4, requires_grad=True)
            
            self._init_weights()
        else:
            self.transformer = None
        
        self.final_norm = nn.LayerNorm(input_dim)

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dinov3_feat, pose_params):
        B = dinov3_feat.shape[0]
        
        visual_token = self.visual_proj(dinov3_feat) 
        
        if self.depth == 0:
            return self.final_norm(visual_token.squeeze(1))
        
        N_visual = visual_token.shape[1]
        
        pose_tokens = [proj(pose_params).unsqueeze(1) for proj in self.pose_group_proj]
        pose_tokens = torch.cat(pose_tokens, dim=1) 
        
        cls_token = self.cls_token.expand(B, -1, -1) 
        
        tokens = torch.cat([cls_token, visual_token, pose_tokens], dim=1)
        
        id_cls = torch.zeros((B, 1), dtype=torch.long, device=tokens.device)
        id_vis = torch.ones((B, N_visual), dtype=torch.long, device=tokens.device)
        
        num_pose = len(self.pose_group_proj)
        id_pose = torch.arange(2, 2 + num_pose, device=tokens.device).unsqueeze(0).expand(B, -1)
        
        modal_ids = torch.cat([id_cls, id_vis, id_pose], dim=1)
        
        tokens = tokens + self.modal_embed(modal_ids)
        
        shortcut = tokens
        trans_out = self.transformer(tokens)
        tokens = shortcut + self.layer_scale * trans_out
        
        return self.final_norm(tokens[:, 0])

class DMSMultiTaskHead(nn.Module):
    def __init__(self, feature_extractor_cfg, task_configs):
        super().__init__()
        
        self.shared_extractor = GeometryAwareFeatureExtractor(**feature_extractor_cfg)
        input_dim = feature_extractor_cfg.get('input_dim', 1024)
        dropout = feature_extractor_cfg.get('dropout', 0.1)
        
        self.heads = nn.ModuleDict()
        
        print(f"🏗️ Building DMS Multi-Task Head...")
        for task_name, config in task_configs.items():
            if isinstance(config, int):
                num_classes = config
                head_type = "simple"
            else:
                num_classes = config['num_classes']
                head_type = config.get('type', 'simple')
            
            if head_type == "simple":
                self.heads[task_name] = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, num_classes)
                )
                
            elif head_type == "complex":
                hidden_dim = input_dim // 2
                self.heads[task_name] = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout * 2),
                    nn.Linear(hidden_dim, num_classes)
                )
            
            print(f"  └─ Task: [{task_name.ljust(10)}] Type={head_type}, Classes={num_classes}")

    def forward(self, dinov3_feat, pose_params, **kwargs):
        shared_feat = self.shared_extractor(dinov3_feat, pose_params)
        
        outputs = {}
        for task_name, head in self.heads.items():
            logits = head(shared_feat)
            
            if self.training:
                logits = 30 * torch.tanh(logits / 30)
            
            outputs[task_name] = logits
            
        return outputs