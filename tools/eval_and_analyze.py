import torch
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
from yacs.config import CfgNode as CN
import pyrootutils

os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/hf_cache"

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)

from geo_dms.models.meta_arch.sam3d_body import GEODMS
from geo_dms.utils.config import get_config
from geo_dms.utils.checkpoint import load_state_dict
from geo_dms.data.datasets.dms_datasets import get_dms_loader
from configs.dms_config import add_dms_config

CLASS_NAMES_RAW = {
    "emotion": ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
    "drowsy": ["Alert", "Drowsy"],
    "distraction": [
        "Safe", "Text(R)", "Phone(R)", "Text(L)", "Phone(L)", 
        "Radio", "Drink", "Reach", "Makeup", "Talk"
    ]
}

# 评估用类别
CLASS_NAMES_EVAL = {
    "emotion": ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
    "drowsy": ["Alert", "Drowsy"],
    "distraction": [
        "Safe", "Text(R)", "Phone(R)", "Text(L)", "Phone(L)", 
        "Radio", "Drink", "Reach", "Makeup", "Talk"
    ]
}

# 数据集关键词映射 (用于从路径识别数据集来源)
SOURCE_KEYWORDS = {
    'rafdb': ['raf', 'RAF'],
    'fer2013': ['fer2013', 'FER'],
    'affectnet': ['affectnet', 'AffectNet'],
    'drowsy_hf': ['Driver-Drowsiness'],
    'state_farm': ['state-farm', 'state_farm']
}

def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

def identify_dataset(path):
    for ds_name, keywords in SOURCE_KEYWORDS.items():
        for k in keywords:
            if k in str(path):
                return ds_name
    return "unknown"

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📥 Config: {args.config}")
    cfg = get_config(args.config)
    cfg.defrost()
    add_dms_config(cfg)
    
    if not hasattr(cfg.MODEL, "DMS"): cfg.MODEL.DMS = CN()
    cfg.MODEL.DMS.ENABLE = True 
    cfg.MODEL.DMS.FUSION_TYPE = "adaptive"  
    cfg.MODEL.DMS.AGGREGATOR_TYPE = "Inter-layer"
    
    # 路径修复
    mhr_rel_path = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    real_mhr_path = mhr_rel_path if os.path.exists(mhr_rel_path) else os.path.join(root, mhr_rel_path)
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = real_mhr_path
    
    if hasattr(cfg, "TRAIN"): cfg.TRAIN.USE_FP16 = False 
    cfg.freeze()

    print("🤖 Initializing model...")
    model = GEODMS(cfg)
    model.to(device)
    model.eval()
    
    if args.ckpt:
        print(f"📥 Loading Weights: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        load_state_dict(model, state_dict, strict=False)

    val_datasets = args.datasets 
    print(f"🎯 Evaluating on: {val_datasets}")
    
    val_loader = get_dms_loader(
        dataset_names=val_datasets, 
        batch_size=args.batch_size, 
        split="test", 
        force_augment=False 
    )

    results = {
        "emotion": {},
        "drowsy": {},
        "distraction": {}
    }
    
    save_dir = "evaluation_results"
    os.makedirs(save_dir, exist_ok=True)
    
    print("🚀 Running Inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            
            if batch['img'].dim() == 4:
                batch['img'] = batch['img'].unsqueeze(1)
            model._initialize_batch(batch)

            output = model.forward_step(batch, decoder_type="body")
            logits_dict = output.get("dms_logits")
            
            if logits_dict is None: continue

            paths = batch.get('img_path', [])

            for task_name in results.keys():
                if task_name not in logits_dict: continue
                
                label_key = f"{task_name}_label"
                if label_key not in batch: continue
                
                logits = logits_dict[task_name] # [B, NumClasses]
                targets = batch[label_key].flatten().long() # [B]
                
                # 过滤无效标签
                valid_mask = targets != -100
                if not valid_mask.any(): continue
                
                valid_logits = logits[valid_mask]
                valid_targets = targets[valid_mask]
                
                valid_preds = torch.argmax(valid_logits, dim=1)
                
                curr_paths = [paths[i] for i in range(len(paths)) if valid_mask[i]]
                
                for i, path in enumerate(curr_paths):
                    ds_name = identify_dataset(path)
                    
                    if ds_name not in results[task_name]:
                        results[task_name][ds_name] = {"preds": [], "targets": [], "bad_cases": []}
                    
                    p = valid_preds[i].item()
                    t = valid_targets[i].item()
                    
                    results[task_name][ds_name]["preds"].append(p)
                    results[task_name][ds_name]["targets"].append(t)
                    
                    if p != t and len(results[task_name][ds_name]["bad_cases"]) < 20:
                        valid_indices = torch.where(valid_mask)[0]
                        original_idx = valid_indices[i]
                        img_tensor = batch["img"][original_idx][0]
                        
                        bad_case_info = {
                            "img": denormalize_image(img_tensor),
                            "pred": p,
                            "target": t,
                            "path": str(path)
                        }
                        results[task_name][ds_name]["bad_cases"].append(bad_case_info)

    print("\n" + "="*50)
    print("📊 FINAL EVALUATION REPORT")
    print("="*50)
    
    for task_name, datasets_data in results.items():
        if not datasets_data: continue
        
        print(f"\nTask: [{task_name.upper()}]")
        class_names = CLASS_NAMES_EVAL[task_name]
        
        all_preds = []
        all_targets = []
        
        for ds_name, data in datasets_data.items():
            if not data["preds"]: continue
            
            p = np.array(data["preds"])
            t = np.array(data["targets"])
            
            all_preds.extend(p)
            all_targets.extend(t)
            
            acc = accuracy_score(t, p)
            print(f"   └─ Dataset: {ds_name:<15} | Acc: {acc:.4f} | Samples: {len(t)}")
            
            cm = confusion_matrix(t, p)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'CM - {task_name} ({ds_name})')
            plt.ylabel('True'); plt.xlabel('Pred')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"cm_{task_name}_{ds_name}.png"))
            plt.close()
            
            bad_case_dir = os.path.join(save_dir, "bad_cases", task_name, ds_name)
            os.makedirs(bad_case_dir, exist_ok=True)
            for i, item in enumerate(data["bad_cases"]):
                img_bgr = cv2.cvtColor((item["img"]*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                p_name = class_names[item['pred']]
                t_name = class_names[item['target']]
                
                cv2.putText(img_bgr, f"GT: {t_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_bgr, f"PD: {p_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                safe_name = os.path.basename(item['path']).replace("/", "_")[-20:]
                cv2.imwrite(os.path.join(bad_case_dir, f"{i}_{safe_name}.jpg"), img_bgr)

        if all_preds:
            print(f"\n   >>> Overall Report for {task_name.upper()} <<<")
            print(classification_report(all_preds, all_targets, target_names=class_names, digits=4))
            
            cm_total = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_total, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Overall CM - {task_name}')
            plt.ylabel('True'); plt.xlabel('Pred')
            plt.savefig(os.path.join(save_dir, f"cm_TOTAL_{task_name}.png"))
            plt.close()
            
    print(f"\n✅ All results saved to: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoints/sam-3d-body-dinov3/model_config.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", default=["rafdb", "fer2013", "affectnet", "drowsy_hf", "state_farm"], 
                        help="List of datasets to use for validation")
    args = parser.parse_args()
    main(args)