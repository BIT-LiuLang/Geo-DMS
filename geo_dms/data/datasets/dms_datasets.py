# Copyright (c) Meta Platforms, Inc. and affiliates.
# DMS Dataset Loader - Supporting Multi-Source (HF, Kaggle, Local) with Correct Mappings & Sampling

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import numpy as np
import os
from PIL import Image
import pandas as pd
import cv2 
import random

# --- 引入项目特定的 Transforms ---
from torchvision.transforms import ToTensor, Normalize
from geo_dms.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper
)

# Optional Imports
try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

try:
    import kagglehub
except ImportError:
    kagglehub = None

# ==========================================
# 1. 标签映射定义
# ==========================================
MAPPING_STATEFARM = {f"c{i}": i for i in range(10)}

# RAF-DB (Kaggle: shuvoalok/raf-db-dataset)
MAPPING_RAFDB_KAGGLE = {
    1: 6, 2: 2, 3: 1, 4: 3, 5: 5, 6: 0, 7: 4
}

# ==========================================
# 2. 基础转换 & BBox 工具
# ==========================================
def get_base_transform(img_size=(256, 256)):
    return Compose([
        GetBBoxCenterScale(),
        TopdownAffine(input_size=img_size, use_udp=False),
        VisionTransformWrapper(ToTensor()),
        VisionTransformWrapper(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
    ])

def augment_bbox(bbox, img_h, img_w, shift_ratio=0.2, scale_ratio=0.15):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    # 0. 基础检查
    if w <= 1 or h <= 1:
        return np.array([0, 0, img_w, img_h], dtype=np.float32)

    cx = x1 + w * 0.5
    cy = y1 + h * 0.5

    # 1. 随机平移
    dx = np.random.uniform(-shift_ratio, shift_ratio) * w
    dy = np.random.uniform(-shift_ratio, shift_ratio) * h
    cx += dx
    cy += dy

    # 2. 随机缩放
    scale = np.random.uniform(1 - scale_ratio, 1 + scale_ratio)
    new_w = w * scale
    new_h = h * scale

    # 3. 重新计算
    new_x1 = cx - new_w * 0.5
    new_y1 = cy - new_h * 0.5
    new_x2 = cx + new_w * 0.5
    new_y2 = cy + new_h * 0.5

    # [关键恢复] 4. 严格的安全检查
    # 确保框还在图片范围内 (至少保留 30% 的内容在图内)
    new_x1 = np.clip(new_x1, -new_w * 0.7, img_w + new_w * 0.7)
    new_y1 = np.clip(new_y1, -new_h * 0.7, img_h + new_h * 0.7)
    
    # 5. 确保没有反转
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return bbox # 回退
        
    final_w = new_x2 - new_x1
    final_h = new_y2 - new_y1
    
    # 6. 最小尺寸硬限制 (防止 NaN)
    if final_w < 32 or final_h < 32:
        return bbox 
        
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

def deterministic_bbox(bbox, img_h, img_w, scale_ratio=0.15):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1: return np.array([0, 0, img_w, img_h], dtype=np.float32)

    cx, cy = x1 + w * 0.5, y1 + h * 0.5
    scale = 1 + scale_ratio  
    new_w, new_h = w * scale, h * scale

    new_x1 = np.clip(cx - new_w * 0.5, 0, img_w)
    new_y1 = np.clip(cy - new_h * 0.5, 0, img_h)
    new_x2 = np.clip(cx + new_w * 0.5, 0, img_w)
    new_y2 = np.clip(cy + new_h * 0.5, 0, img_h)
    
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

def refine_statefarm_bbox(bbox, img_h, img_w, expand_ratio=0.4):
    """
    StateFarm 专用 Refine 逻辑：
    1. 规则纠偏：把框强行拉回驾驶员一侧（右侧）
    2. 扩充：基于紧凑的检测框向外扩充，纳入上下文
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    if w <= 1 or h <= 1: 
        return np.array([0, 0, img_w, img_h], dtype=np.float32)

    cx = x1 + w * 0.5
    cy = y1 + h * 0.5

    # 规则 A: 左右纠偏 (拉回右侧驾驶员)
    if cx < img_w * 0.4:
        cx = img_w * 0.65
    
    # 规则 B: 上下纠偏 (上提给头部留空间)
    if cy > img_h * 0.5:
        cy -= img_h * 0.15
        h *= 1.3 
        
    # 规则 C: 宽度修正
    if w < img_w * 0.3:
        w = img_w * 0.5

    # 视野扩张
    scale_factor = 1 + expand_ratio
    final_w = w * scale_factor
    final_h = h * scale_factor

    new_x1 = cx - final_w * 0.5
    new_y1 = cy - final_h * 0.5
    new_x2 = cx + final_w * 0.5
    new_y2 = cy + final_h * 0.5
    
    new_x1 = np.clip(new_x1, 0, img_w)
    new_y1 = np.clip(new_y1, 0, img_h)
    new_x2 = np.clip(new_x2, 0, img_w)
    new_y2 = np.clip(new_y2, 0, img_h)
    
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

# ==========================================
# 3. 统一 DMS 数据集类
# ==========================================
class UnifiedDMSDataset(Dataset):
    def __init__(self, source_name, source_type, task_type, split="train", transform=None, force_augment=False, **kwargs):
        self.source_name = source_name
        self.source_type = source_type
        self.task_type = task_type
        self.split = "train" if split == "train" else "test"
        self.transform = transform if transform else get_base_transform()
        self.force_augment = force_augment
        self.items = []
        self.hf_data = None
        self.use_hf_indexing = False
        
        print(f"📦 [DMS Dataset] Loading {source_name} (Task: {task_type}, Split: {self.split})...")
        
        if source_type == "huggingface":
            self._load_huggingface(source_name)
        elif source_type == "kaggle":
            self._load_kaggle(source_name)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
            
        if not self.use_hf_indexing and self.split == "train":
            random.seed(42)
            random.shuffle(self.items)

        count = len(self.hf_data) if self.use_hf_indexing else len(self.items)
        print(f"✅ Loaded {count} samples from {source_name}")

    def _load_huggingface(self, name):
        if hf_load_dataset is None: raise ImportError("Please `pip install datasets`")
        self.use_hf_indexing = True
        
        def load_and_split_hf_dataset(dataset_name, val_ratio=0.1, fallback_split="train"):
            target_split = "train" if self.split == "train" else "test"
            try:
                ds = hf_load_dataset(dataset_name, split=target_split)
                print(f"   -> Found native split: {target_split}")
                return ds
            except Exception:
                print(f"   -> Native split '{target_split}' not found. Performing random split (seed=42).")
                try:
                    full_ds = hf_load_dataset(dataset_name, split=fallback_split)
                except:
                    full_ds = hf_load_dataset(dataset_name)[fallback_split]
                splitted = full_ds.train_test_split(test_size=val_ratio, seed=42)
                return splitted['train'] if self.split == "train" else splitted['test']

        if "affectnethq" in name:
            self.hf_data = load_and_split_hf_dataset(name, val_ratio=0.1)
            self.img_key = "image"; self.label_key = "label"
        elif "fer2013" in name:
            self.hf_data = load_and_split_hf_dataset(name, val_ratio=0.1)
            self.img_key = "jpg"; self.label_key = "cls"
        elif "Driver-Drowsiness-Dataset" in name:
            self.hf_data = load_and_split_hf_dataset(name, val_ratio=0.2)
            self.img_key = "image"; self.label_key = "label"
        else:
            raise ValueError(f"❌ Unknown HuggingFace dataset: {name}")

    def _load_kaggle(self, name):
        if kagglehub is None: raise ImportError("Please `pip install kagglehub pandas`")
        self.use_hf_indexing = False
        
        LOCAL_OVERRIDES = {
            "shuvoalok/raf-db-dataset": "/root/autodl-tmp/data/raf-db",
            "rightway11/state-farm-distracted-driver-detection": "/root/autodl-tmp/data/state-farm",
        }
        
        path = None
        if name in LOCAL_OVERRIDES and os.path.exists(LOCAL_OVERRIDES[name]):
            path = LOCAL_OVERRIDES[name]
            print(f"   -> 🚀 [Local Hit] {path}")
        else:
            print(f"   -> 🔍 Checking Kaggle Cache for {name}...")
            os.environ["KAGGLEHUB_CACHE"] = "/root/autodl-tmp/kaggle_cache"
            path = kagglehub.dataset_download(name)

        if "raf-db" in name:
            root_dir = path
            for r, d, f in os.walk(path):
                if "train" in d: root_dir = r; break
            
            target_subdir = "train" if self.split == "train" else "test"
            img_dir = os.path.join(root_dir, target_subdir)
            
            if not os.path.exists(img_dir):
                 img_dir = os.path.join(root_dir, "RAF-DB", target_subdir)
                 if not os.path.exists(img_dir): raise FileNotFoundError(f"RAF-DB {target_subdir} not found")

            for label_str in os.listdir(img_dir):
                if not label_str.isdigit(): continue
                raw_label = int(label_str)
                class_dir = os.path.join(img_dir, label_str)
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.items.append({"path": os.path.join(class_dir, fname), "label": raw_label})

        elif "state-farm" in name:
            csv_path = os.path.join(path, "driver_imgs_list.csv")
            df = pd.read_csv(csv_path)
            subjects = df['subject'].unique()
            np.random.seed(42)
            val_subjects = np.random.choice(subjects, size=int(len(subjects)*0.2), replace=False)
            
            df_target = df[~df['subject'].isin(val_subjects)] if self.split == "train" else df[df['subject'].isin(val_subjects)]
            img_root = os.path.join(path, "imgs", "train")
            
            for _, row in df_target.iterrows():
                full_path = os.path.join(img_root, row['classname'], row['img'])
                if os.path.exists(full_path):
                    self.items.append({"path": full_path, "label": MAPPING_STATEFARM.get(row['classname'], -1)})

    def __len__(self):
        return len(self.hf_data) if self.use_hf_indexing else len(self.items)

    def __getitem__(self, idx):
        # 1. Image Loading
        if self.use_hf_indexing:
            item = self.hf_data[idx]
            img = item[self.img_key]
            if img.mode != "RGB": img = img.convert("RGB")
            img = np.array(img)
            raw_label = item[self.label_key]
            current_path = f"HF_{self.source_name}_{idx}"
            mask_root = "/root/autodl-tmp/data/precomputed/" + self.source_name.split("/")[-1]
            mask_filename = f"{idx:08d}"
        else:
            item = self.items[idx]
            img_path = item['path']
            raw_label = item['label']
            
            # [关键防御 1] 捕获 OpenCV 读取失败
            img_cv2 = cv2.imread(img_path)
            if img_cv2 is None:
                # print(f"⚠️ Failed to read {img_path}, skipping...")
                return self.__getitem__((idx + 1) % len(self))
                
            img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            current_path = img_path
            mask_root = "/root/autodl-tmp/data/precomputed/" + self.source_name.split("/")[-1]
            mask_filename = os.path.splitext(os.path.basename(img_path))[0]

        # [关键防御 2] 复活：跳过极小图片/坏图
        H, W = img.shape[:2]
        if H < 32 or W < 32:
            # 递归读取下一张
            return self.__getitem__((idx + 1) % len(self))

        # 2. Mask & Meta Loading (Initial Attempt)
        mask_path = os.path.join(mask_root, mask_filename + ".png")
        meta_path = os.path.join(mask_root, mask_filename + ".npy")
        has_precomputed = os.path.exists(mask_path) and os.path.exists(meta_path)
        
        mask = None
        mask_score = 1.0
        cam_int = None
        raw_bbox = None # 存储“初始检测框”

        # -------------------------------------------------------------
        # Step A: 获取原始框 (From File OR From Rule)
        # -------------------------------------------------------------
        if has_precomputed:
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                meta = np.load(meta_path, allow_pickle=True).item()
                mask_score = float(meta.get("mask_score", 1.0))
                if "cam_int" in meta: cam_int = torch.from_numpy(meta["cam_int"]).float()
                
                # 如果有预计算的 bbox，直接用
                raw_bbox = meta.get("bbox", None)
            except:
                has_precomputed = False

        if not has_precomputed:
            # Fallback 逻辑
            H, W = img.shape[:2]
            mask = np.ones((H, W), dtype=np.uint8) * 255
            
            # [关键修复] StateFarm 虚拟检测框策略
            if self.task_type == "distraction" and "state-farm" in self.source_name:
                # 构造一个紧凑的右侧驾驶员框 (模拟检测器输出)
                # x: 40%~90%, y: 20%~80% -> 这是一个比较“窄”的框
                # 这样传给 refine 时，经过 expand (1.4x) 才会变成一个完美的 context crop
                raw_bbox = np.array([W*0.4, H*0.2, W*0.9, H*0.8], dtype=np.float32)
            else:
                # 其他任务 (如 RAF-DB) 默认为全图
                raw_bbox = np.array([0, 0, W, H], dtype=np.float32)

        # -------------------------------------------------------------
        # Step B: 应用 BBox Refine 策略 (统一处理)
        # -------------------------------------------------------------
        if raw_bbox is None: # 防御性编程
             H, W = img.shape[:2]
             raw_bbox = np.array([0, 0, W, H], dtype=np.float32)

        H, W = img.shape[:2]
        
        if self.task_type == "distraction":
            # 这里的 raw_bbox 要么是检测器出来的，要么是我们模拟的“紧凑框”
            # refine 会负责 expand 和 shift，保证训练输入的一致性
            bbox = refine_statefarm_bbox(raw_bbox, H, W, expand_ratio=0.4)
        
        elif self.task_type == "emotion":
            # Emotion 任务通常使用全图作为基础 (RAF-DB 已经是 crop 过的)
            bbox = np.array([0, 0, W, H], dtype=np.float32)
        
        else:
            bbox = raw_bbox # Drowsy 等任务直接用

        # -------------------------------------------------------------
        # Step C: Augmentation (Train vs Val)
        # -------------------------------------------------------------
        if self.split == "train" or self.force_augment:
            bbox = augment_bbox(bbox, H, W)
        else:
            bbox = deterministic_bbox(bbox, H, W)

        if cam_int is None:
            focal = (img.shape[0]**2 + img.shape[1]**2)**0.5
            cam_int = torch.tensor([[focal, 0, img.shape[1]/2], [0, focal, img.shape[0]/2], [0, 0, 1]], dtype=torch.float32)

        # 3. Transform
        res = self.transform({"img": img, "bbox": bbox, "bbox_format": "xyxy", "mask": mask})

        # 4. Label Logic
        labels = {"emotion_label": -100, "drowsy_label": -100, "distraction_label": -100}
        if self.task_type == "emotion":
            final_label = raw_label
            if "raf-db" in self.source_name.lower():
                final_label = MAPPING_RAFDB_KAGGLE.get(raw_label, -1)
            if 0 <= final_label <= 6:
                labels["emotion_label"] = final_label
        elif self.task_type == "distraction":
            labels["distraction_label"] = raw_label
        elif self.task_type == "drowsy":
            labels["drowsy_label"] = raw_label

        # 5. Pack Output
        person_valid = torch.tensor([1], dtype=torch.float32) 

        return {
            "img": res['img'],
            "bbox": torch.tensor(res['bbox']).float().unsqueeze(0),
            "cam_int": cam_int,
            "img_path": current_path,
            "person_valid": person_valid, 
            "has_body_info": torch.tensor([1.0 if self.task_type != "emotion" else 0.0]),
            "affine_trans": torch.from_numpy(res.get('affine_trans', np.eye(2, 3))).float().unsqueeze(0),
            "mask": (torch.from_numpy(res['mask']) > 127).float().unsqueeze(0).unsqueeze(0),
            "mask_score": torch.tensor([mask_score]),
            "img_size": torch.tensor([res['img'].shape[-2], res['img'].shape[-1]]).float().unsqueeze(0),
            "ori_img_size": torch.tensor([img.shape[0], img.shape[1]]).float().unsqueeze(0),
            "bbox_center": torch.from_numpy(res['bbox_center']).float().unsqueeze(0),
            "bbox_scale": torch.from_numpy(res['bbox_scale']).float().unsqueeze(0),
            **labels
        }

# ==========================================
# 4. Loader Builder
# ==========================================
def get_dms_loader(dataset_names, batch_size=32, split="train", force_augment=False):
    datasets = []
    transform = get_base_transform((256, 256))
    
    if "affectnet" in dataset_names:
        datasets.append(UnifiedDMSDataset("Piro17/affectnethq", "huggingface", "emotion", split, transform, force_augment=force_augment))
    if "fer2013" in dataset_names:
        datasets.append(UnifiedDMSDataset("clip-benchmark/wds_fer2013", "huggingface", "emotion", split, transform, force_augment=force_augment))
    if "drowsy_hf" in dataset_names:
        datasets.append(UnifiedDMSDataset("akahana/Driver-Drowsiness-Dataset", "huggingface", "drowsy", split, transform, force_augment=force_augment))
    
    if "rafdb" in dataset_names:
        datasets.append(UnifiedDMSDataset("shuvoalok/raf-db-dataset", "kaggle", "emotion", split, transform, force_augment=force_augment))
    if "state_farm" in dataset_names:
        datasets.append(UnifiedDMSDataset("rightway11/state-farm-distracted-driver-detection", "kaggle", "distraction", split, transform, force_augment=force_augment))

    if not datasets: 
        print(f"⚠️ Warning: No valid datasets found in {dataset_names}.")
        return None

    full_dataset = ConcatDataset(datasets)
    
    sampler = None
    shuffle = (split == "train")
    
    if split == "train":
        print("⚖️  [Sampler] Calculating weights for task-balanced sampling...")
        task_total_counts = {}
        for ds in datasets:
            t = ds.task_type
            task_total_counts[t] = task_total_counts.get(t, 0) + len(ds)
        
        num_tasks = len(task_total_counts)
        print(f"📊 Task Counts: {task_total_counts}")

        sample_weights = []
        for ds in datasets:
            t = ds.task_type
            n_ds = len(ds)
            if task_total_counts[t] > 0:
                w = 1.0 / (num_tasks * task_total_counts[t])
                sample_weights.extend([w] * n_ds)
            else:
                sample_weights.extend([0.0] * n_ds)

        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(full_dataset), 
            replacement=True
        )
        shuffle = False 

    return DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        sampler=sampler, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        drop_last=(split == "train")
    )