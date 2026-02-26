# Copyright (c) Meta Platforms, Inc. and affiliates.
# Geo-DMS Demo Script: 3D Pose + Multi-Task DMS Inference
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from tqdm import tqdm
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from geo_dms import SAM3DBodyEstimator
from geo_dms.models.meta_arch.sam3d_body import GEODMS
from geo_dms.utils.config import get_config
from geo_dms.utils.checkpoint import load_state_dict
from configs.dms_config import add_dms_config
from geo_dms.visualization.renderer import Renderer
from geo_dms.visualization.skeleton_visualizer import SkeletonVisualizer
from geo_dms.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)

DMS_LABELS = {
    "emotion": ["Neutral", "Fear", "Disgust", "Happy", "Neutral", "Anger", "Neutral"],
    "drowsy": ["Drowsy", "Non Drowsy"], 
    "distraction": [
        "Safe", "Text(R)", "Phone(R)", "Text(L)", "Phone(L)", 
        "Radio", "Drink", "Reach", "Makeup", "Talk"
    ]
}


def visualize_all_in_one(img_cv2, outputs, faces, alpha=0.6):
    """
    在一张图上同时叠加：原图 + 骨架(Keypoints) + 半透明Mesh
    """
    canvas = img_cv2.copy()

    if len(outputs) > 0:
        all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
        outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]
    else:
        outputs_sorted = []

    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        canvas = visualizer.draw_skeleton(canvas, keypoints_2d)

    if len(outputs_sorted) == 0:
        return canvas

    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    renderer = Renderer(focal_length=outputs_sorted[0]["focal_length"], faces=all_faces)

    blank_bg = np.zeros_like(img_cv2)
    mesh_render = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            blank_bg,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(0, 0, 0), 
        )
        * 255
    ).astype(np.uint8)

    mask = (mesh_render > 0).any(axis=2)
    if mask.any():
        blended = (
            canvas[mask].astype(np.float32) * (1 - alpha) + 
            mesh_render[mask].astype(np.float32) * alpha
        )
        canvas[mask] = blended.astype(np.uint8)

    return canvas

def get_face_box(keypoints, img_shape, padding=1.5):
    H, W = img_shape[:2]
    head_kpts = keypoints[:5, :2] 
    valid_mask = (head_kpts[:, 0] > 0) & (head_kpts[:, 1] > 0)
    
    if valid_mask.sum() < 2: return None
        
    valid_head_kpts = head_kpts[valid_mask]
    x_min, y_min = np.min(valid_head_kpts[:, 0]), np.min(valid_head_kpts[:, 1])
    x_max, y_max = np.max(valid_head_kpts[:, 0]), np.max(valid_head_kpts[:, 1])
    
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    w, h = x_max - x_min, y_max - y_min
    size = max(w, h) * padding
    
    x1 = max(0, int(cx - size / 2))
    y1 = max(0, int(cy - size / 2))
    x2 = min(W, int(cx + size / 2))
    y2 = min(H, int(cy + size / 2))
    
    return np.array([x1, y1, x2, y2])

def load_dms_model(config_path, ckpt_path, device):
    base = "checkpoints/sam-3d-body-dinov3/model_config.yaml"
    if not os.path.exists(base): 
        base = os.path.join(os.path.dirname(__file__), "checkpoints/sam-3d-body-dinov3/model_config.yaml")
    
    cfg = get_config(base)
    cfg.defrost()
    add_dms_config(cfg)
    cfg.merge_from_file(config_path)
    
    mhr_path = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    if not os.path.exists(mhr_path):
        mhr_path = os.path.join(os.environ.get("SAM3D_MHR_PATH", ""), "mhr_model.pt")
    
    if not hasattr(cfg.MODEL, "MHR_HEAD"):
        from yacs.config import CfgNode as CN
        cfg.MODEL.MHR_HEAD = CN()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    if hasattr(cfg, "TRAIN"): cfg.TRAIN.USE_FP16 = False
    cfg.freeze()
    
    print(f"🔹 Loading Geo-DMS model from {config_path}...")
    model = GEODMS(cfg)
    
    if os.path.exists(ckpt_path):
        print(f"   Loading weights: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        load_state_dict(model, state_dict, strict=False)
    else:
        print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}")

    model.to(device)
    model.eval()
    return model, cfg

def save_fusion_dashboard(img_result_bgr, results, save_path):
    img_rgb = cv2.cvtColor(img_result_bgr, cv2.COLOR_BGR2RGB)
    
    H, W, _ = img_rgb.shape
    
    BASE_WIDTH = 800.0
    scale = max(0.6, W / BASE_WIDTH)
    
    base_w = 240
    base_h = 200
    
    panel_w = int(base_w * scale)
    panel_h = int(base_h * scale)
    margin  = int(20 * scale)
    
    fs_title = max(10, int(16 * scale))
    fs_label = max(8,  int(12 * scale))
    fs_val   = max(9,  int(15 * scale))
    
    pad_y   = int(35 * scale)
    line_h  = int(40 * scale)
    val_off = int(85 * scale)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=600)
    ax = plt.Axes(fig, [0., 0., 1., 1.]) 
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(img_rgb)
    
    
    x0 = W - panel_w - margin
    y0 = margin
    
    rect = patches.Rectangle(
        (x0, y0), panel_w, panel_h, 
        linewidth=0, edgecolor='none', facecolor='black', alpha=0.6
    )
    ax.add_patch(rect)
    
    ax.text(x0 + panel_w / 2, y0 + pad_y, "Geo-DMS Analysis", 
            color='white', fontsize=fs_title, fontweight='bold', 
            family='sans-serif', ha='center')
    
    line_y = y0 + int(45 * scale)
    ax.plot([x0 + 10, x0 + panel_w - 10], [line_y, line_y], color='gray', linewidth=max(1, int(1*scale)))
    
    rgb_colors = {
        "distraction": (0.792, 0.404, 0.008), 
        "emotion":     (0.161, 0.431, 0.706), 
        "drowsy_bad":  (0.682, 0.125, 0.071),  
        "safe":        (0.039, 0.576, 0.588)   
    }
    
    rows = [
        ("Action:",  results.get("distraction", "N/A"), rgb_colors["distraction"]),
        ("Emotion:", results.get("emotion", "N/A"),     rgb_colors["emotion"]),
        ("State:",   results.get("drowsy", "N/A"),      rgb_colors["drowsy_bad"] if results.get("drowsy") == "Drowsy" else rgb_colors["safe"])
    ]
    
    curr_y = y0 + int(85 * scale)
    label_pad_x = int(20 * scale)
    
    for label, val, color in rows:
        final_color = rgb_colors["safe"] if val == "Safe" else color
        
        ax.text(x0 + label_pad_x, curr_y, label, 
                color=(1, 1, 1), fontsize=fs_label, family='sans-serif', fontweight='bold')
        
        ax.text(x0 + label_pad_x + val_off, curr_y, val,
                color=final_color, fontsize=fs_val, fontweight='bold', family='sans-serif')
        
        curr_y += line_h

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0)
    
    plt.close(fig)

def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output_dms_demo", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, model_cfg = load_dms_model(args.config, args.checkpoint, device)

    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    
    human_detector = None
    if args.detector_name:
        try:
            from tools.build_detector import HumanDetector
            human_detector = HumanDetector(name=args.detector_name, device=device, path=detector_path)
            print(f"✅ Human Detector ({args.detector_name}) loaded.")
        except ImportError:
            print("⚠️ HumanDetector not found.")
            
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=None,
    )

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images_list = sorted([
        image for ext in image_extensions
        for image in glob(os.path.join(args.image_folder, ext))
    ])
    
    print(f"🚀 Starting inference on {len(images_list)} images...")

    from geo_dms.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
    from torchvision.transforms import ToTensor, Normalize

    transform = Compose([
        GetBBoxCenterScale(),
        TopdownAffine(input_size=(256, 256), use_udp=False),
        VisionTransformWrapper(ToTensor()),
        VisionTransformWrapper(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
    ])

    for image_path in tqdm(images_list):
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None: continue
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        
        # 1. BBox Detection (Body)
        bbox = None
        if human_detector is not None:
            try:
                det_out = human_detector(img_rgb)
                if isinstance(det_out, tuple): bboxes = det_out[0]
                else: bboxes = det_out
                if isinstance(bboxes, list): bboxes = np.array(bboxes)
                if len(bboxes) > 0:
                    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                    bbox = bboxes[np.argmax(areas)]
            except Exception: pass
        if bbox is None: bbox = np.array([0, 0, W, H])

        # 2. Pass 1: Get Mesh & Pose
        mesh_outputs = estimator.process_one_image(
            image_path, bbox_thr=args.bbox_thresh, use_mask=False
        )
        
        final_vis_img = visualize_all_in_one(img_cv2, mesh_outputs, estimator.faces, alpha=0.6)
        
        # 3. DMS Inference (Pass 1 - Body)
        data_info_body = {"img": img_rgb, "bbox": bbox, "bbox_format": "xyxy", "mask": np.ones((H, W), dtype=np.uint8)*255}
        res_body = transform(data_info_body)
        
        def make_batch(res_item):
            img_tensor = res_item['img'].unsqueeze(0).to(device)
            img_tensor = img_tensor.unsqueeze(1) 
            batch = {
                "img": img_tensor,
                "has_body_info": torch.ones(1, 1).to(device),
                "cam_int": torch.eye(3).unsqueeze(0).to(device),
                "bbox_center": torch.from_numpy(res_item['bbox_center']).float().unsqueeze(0).unsqueeze(0).to(device),
                "bbox_scale": torch.from_numpy(res_item['bbox_scale']).float().unsqueeze(0).unsqueeze(0).to(device),
                "affine_trans": torch.from_numpy(res_item['affine_trans']).float().unsqueeze(0).unsqueeze(0).to(device),
                "img_path": [image_path],
                "mask": torch.ones(1, 1, 256, 256).to(device),
                "mask_score": torch.ones(1, 1).to(device),
                "ori_img_size": torch.tensor([H, W]).float().unsqueeze(0).unsqueeze(0).to(device),
                "img_size": torch.tensor([256, 256]).float().unsqueeze(0).unsqueeze(0).to(device),
            }
            return batch

        model._max_num_person = 1
        model._batch_size = 1
        dms_results = {}
        
        with torch.no_grad():
            batch_body = make_batch(res_body)
            out_body = model.forward_step(batch_body, decoder_type="body")
            
            if "dms_logits" in out_body and "distraction" in out_body["dms_logits"]:
                idx = torch.argmax(out_body["dms_logits"]["distraction"][0]).item()
                dms_results["distraction"] = DMS_LABELS["distraction"][idx]
            
            if "dms_logits" in out_body:
                if "emotion" in out_body["dms_logits"]:
                    idx = torch.argmax(out_body["dms_logits"]["emotion"][0]).item()
                    dms_results["emotion"] = DMS_LABELS["emotion"][idx]
                if "drowsy" in out_body["dms_logits"]:
                    idx = torch.argmax(out_body["dms_logits"]["drowsy"][0]).item()
                    dms_results["drowsy"] = DMS_LABELS["drowsy"][idx]

        # 4. DMS Inference (Pass 2 - Face, if available)
        if len(mesh_outputs) > 0:
            pred_kpts = mesh_outputs[0]["pred_keypoints_2d"]
            face_bbox = get_face_box(pred_kpts, (H, W), padding=1.5)
            
            if face_bbox is not None:
                fx1, fy1, fx2, fy2 = face_bbox.astype(int)
                cv2.rectangle(final_vis_img, (fx1, fy1), (fx2, fy2), (38, 34, 155), 2)
                
                data_info_face = {"img": img_rgb, "bbox": face_bbox, "bbox_format": "xyxy", "mask": np.ones((H, W), dtype=np.uint8)*255}
                res_face = transform(data_info_face)
                
                with torch.no_grad():
                    batch_face = make_batch(res_face)
                    out_face = model.forward_step(batch_face, decoder_type="body")
                    
                    if "dms_logits" in out_face:
                        logits = out_face["dms_logits"]
                        if "emotion" in logits:
                            idx = torch.argmax(logits["emotion"][0]).item()
                            dms_results["emotion"] = DMS_LABELS["emotion"][idx] 
                        if "drowsy" in logits:
                            idx = torch.argmax(logits["drowsy"][0]).item()
                            dms_results["drowsy"] = DMS_LABELS["drowsy"][idx]   

        out_name = os.path.join(output_folder, os.path.basename(image_path))
        
        save_fusion_dashboard(final_vis_img, dms_results, out_name)
        # ===================================================================

    print(f"✅ Demo finished. Results saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geo-DMS Demo Visualization")
    
    parser.add_argument("--image_folder", required=True, help="Input images folder")
    parser.add_argument("--output_folder", default="", help="Output folder")
    parser.add_argument("--config", default="dms_multitask_full_test.yaml", help="Path to DMS config yaml")
    parser.add_argument("--checkpoint", default="logs/dms_training/dms_multitask_full_test/val/acc_distraction=0.8698.ckpt", help="Path to model checkpoint")
    parser.add_argument("--detector_name", default="vitdet", help="Human detector name")
    parser.add_argument("--detector_path", default="", help="Detector weights path")
    parser.add_argument("--bbox_thresh", default=0.7, type=float)
    parser.add_argument("--segmentor_path", default="")
    parser.add_argument("--fov_path", default="")

    args = parser.parse_args()
    main(args)