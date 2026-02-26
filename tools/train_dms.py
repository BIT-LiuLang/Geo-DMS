#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import pyrootutils
import os

os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from yacs.config import CfgNode as CN # 引入 CN 以便动态添加节点
torch.serialization.add_safe_globals([CN])
from geo_dms.models.meta_arch.sam3d_body import GEODMS
from geo_dms.utils.config import get_config 
from geo_dms.utils.checkpoint import load_state_dict 

from geo_dms.data.datasets.dms_datasets import get_dms_loader
from configs.dms_config import add_dms_config

def main():
    parser = argparse.ArgumentParser(description="Training Script for SAM3D-DMS Branch")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()


    base_config_path = "checkpoints/sam-3d-body-dinov3/model_config.yaml"
    
    ckpt_path = "checkpoints/sam-3d-body-dinov3/model.ckpt"
    mhr_path = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    
    if not os.path.exists(base_config_path): base_config_path = os.path.join(root, base_config_path)
    if not os.path.exists(ckpt_path): ckpt_path = os.path.join(root, ckpt_path)
    if not os.path.exists(mhr_path): mhr_path = os.path.join(root, mhr_path)
    
    print(f"Paths Configured:\n - Base Config: {base_config_path}\n - Checkpoint: {ckpt_path}\n - MHR Model: {mhr_path}")

    
    cfg = get_config(base_config_path)
    
    cfg.defrost()
    add_dms_config(cfg)

    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path

    if args.config_file:
        print(f"Merging experiment config from {args.config_file}...")
        cfg.merge_from_file(args.config_file)
    
    if args.opts:
        cfg.merge_from_list(args.opts)
        
    cfg.freeze()

    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    print("Initializing SAM3D-DMS Model...")
    model = GEODMS(cfg)

    if os.path.exists(ckpt_path):
        print(f"Loading pretrained pose weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        
        load_state_dict(model, state_dict, strict=False)
        print("✅ Pretrained weights loaded.")
    else:
        print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}.")

    BATCH_SIZE = cfg.SOLVER.get("BATCH_SIZE", 32)
    
    train_datasets = cfg.DATASETS.TRAIN_DMS
    val_datasets = cfg.DATASETS.VAL_DMS
    
    print(f"🎯 [DMS] Training Sets: {train_datasets}")
    
    train_loader = get_dms_loader(
        dataset_names=train_datasets, 
        batch_size=BATCH_SIZE, 
        split="train"
    )
    
    val_loader = get_dms_loader(
        dataset_names=val_datasets, 
        batch_size=BATCH_SIZE, 
        split="validation"
    )

    if args.config_file:
        experiment_name = os.path.splitext(os.path.basename(args.config_file))[0]
    else:
        experiment_name = "dms_default_run"

    wandb_logger = WandbLogger(
        project="SAM3D-DMS", 
        name=experiment_name,
        save_dir="logs/",
        offline=False,
        log_model=False
    )
    
    checkpoint_callback_acc = ModelCheckpoint(
        dirpath=f"logs/dms_training/{experiment_name}",
        filename="{val/acc_emotion_rafdb:.4f}", 
        save_top_k=1,
        monitor="val/acc_emotion_rafdb", 
        mode="max",
        save_last=True  
    )
    checkpoint_callback_acc = ModelCheckpoint(
        dirpath=f"logs/dms_training/{experiment_name}",
        filename="{val/acc_distraction:.4f}", 
        save_top_k=1,
        monitor="val/acc_distraction", 
        mode="max",
        save_last=False  
    )
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=f"logs/dms_training/{experiment_name}",
        filename="{val/loss:.4f}", 
        save_top_k=1,
        monitor="val/loss",
        mode="min",      
        save_last=False  
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.get("MAX_EPOCHS", 10),
        accelerator="gpu",
        devices=[0],
        precision="bf16-mixed",
        logger=wandb_logger,
        callbacks=[checkpoint_callback_acc, checkpoint_callback_loss, lr_monitor, StochasticWeightAveraging(swa_lrs=1e-5)],
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        gradient_clip_val=0.5, 
        gradient_clip_algorithm="norm", 
        strategy="auto"
    )

    print(f"🚀 Starting DMS Training: {experiment_name}")
    if args.resume:
        print(f"♻️  Resuming from checkpoint: {args.resume}")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()