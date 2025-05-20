import os
import sys
import torch
import torch.nn as nn
import numpy as np
from mmpose.evaluation.functional import keypoint_pck_accuracy
from easydict import EasyDict
from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
import cv2
import os.path as osp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch.nn.functional as F
from mmpose.apis import init_model


from posetimation import get_cfg, update_config 


class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4):
        super(Poseidon, self).__init__()
        
        self.device = device

        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        self.backbone = self.model.backbone

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE
   
        # Scongelamento parziale del backbone
        if cfg.MODEL.FREEZE_WEIGHTS:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for layer in self.backbone.layers[-12:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.backbone.ln1.parameters():
                param.requires_grad = True
                

        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.embed_dim = cfg.MODEL.EMBED_DIM # 384
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS # 17
        self.num_frames = cfg.WINDOWS_SIZE
        
        self.is_train = True if phase == 'train' else False

        # Layer normalization
        self.layer_norm = nn.LayerNorm([self.embed_dim, 24, 18])

        # Print learning parameters
        print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n\n")

        print(f"Poseidon parameters: {round(self.count_parameters() / 1e6, 1)} M\n\n")

        print(f"Poseidon backbone parameters: {round(self.count_backbone_parameters() / 1e6, 1)} M\n\n")

    def count_backbone_parameters(self):
        return sum(p.numel() for p in self.backbone.parameters())

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())


    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        x = self.backbone(x)[0]

        x = x.view(batch_size, num_frames, self.embed_dim, 24, 18)

        # Cross-Attention
        center_frame_idx = num_frames // 2
        center_frame = x[:, center_frame_idx]

        # Deconvolution layers
        x = self.deconv_layer(center_frame)

        # Final layer
        x = self.final_layer(x)
        
        return x

    def set_phase(self, phase):
        self.phase = phase
        self.is_train = True if phase == TRAIN_PHASE else False

    def get_phase(self):
        return self.phase

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        N = output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            norm_factor=np.ones((N, 2), dtype=np.float32))

        return avg_acc



