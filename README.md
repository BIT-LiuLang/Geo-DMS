# Geo-DMS: Unified Multi-Task Driver Monitoring with 3D Geometric Priors

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18790509.svg)](https://doi.org/10.5281/zenodo.18790509) 

This repository holds the official implementation of the paper: **"Geo-DMS: Unified Multi-Task Driver Monitoring with 3D Geometric Priors"**.

## 📝 Abstract

Next-generation intelligent cockpits require a unified and high-precision understanding of driver states. Traditional driver monitoring systems, designed in a task-fragmented manner, struggle to resolve geometric ambiguities caused by occlusions. To bridge the gap between 2D visual representations and 3D physical structures, we propose Geo-DMS, a unified driver monitoring system that leverages DINOv3 as the visual backbone and SAM3D Body for explicit human geometric priors. Geo-DMS integrates multi-level features through an Inter-Layer Channel Aggregator and enforces structural constraints via a Pose-Guided Adaptive Fusion module. Unlike conventional classification-only approaches, Geo-DMS simultaneously predicts driver risk states (drowsiness, emotion, distraction) and metric-scale 3D human body meshes. Extensive experiments on five public datasets demonstrate competitive performance and strong robustness, achieving this with only 79.6M updated parameters (approximately 5.7% of the total model capacity), highlighting its efficiency and scalability.

## 💡 Method Overview

<div align="center">
  <img src="assets/pipeline.png" width="100%" alt="Pipeline Figure"/>
  <br/>
  <em>Figure 1: The overall framework of our proposed Geo-DMS.</em>
</div>

<br/>

The architecture of the Geo-DMS framework built upon the frozen SAM 3D Body framework (DINOv3 backbone & MHR decoder), the pipeline employs an Inter-Layer Channel Aggregator (ILCA) to unify global semantics ($F_{agg}$) and a Pose-Guided Adaptive Fusion (PGAF) module to inject geometric priors via a parallel strategy.

## 📦 Code & Checkpoints

The core engine of **Geo-DMS** is released. 

⏳ **Pre-trained Weights:** Currently undergoing final organization. 

## 🚀 Getting Started

### Installation & Deployment

For detailed environment setup and deployment instructions, please refer to our Installation Guide: [INSTALL.md](INSTALL.md).

### Model Training

To train the Geo-DMS model using your configured datasets, run the following command:

```bash
python tools/train_dms.py --config-file configs/experiments/dms_multitask.yaml
```

### Inference Demo

To run inference on a folder of images and visualize the multi-task results, use the demo script. Replace `xxxx.ckpt` with your actual checkpoint file name:

```bash
python tools/demo_dms_vis.py \
  --image_folder test_demo/demo \
  --config configs/experiments/dms_multitask.yaml \
  --checkpoint logs/dms_training/dms_multitask/val/xxxx.ckpt
```

## ✅ To-Do List

- [x] Release preprint (Research Square)
- [x] **Release core algorithm implementation**
- [x] Submit to IEEE ICCMS 2026
- [x] **Add installation and usage instructions**
- [ ] Release pretrained weights


## 📌 Citation

If you find this repository or our research helpful, please use the following BibTeX entry:

```bibtex
@inproceedings{liu2026geodms,
  title={Geo-DMS: Unified Multi-Task Driver Monitoring with 3D Geometric Priors},
  author={Liu, Lang and Jiang, Xiaobei and Wang, Zhengyu and Zhou, Junjun and Ma, Yixue},
  booktitle={ICCMS 2026: The 18th International Conference on Computer Modeling and Simulation},
  year={2026},
  note={Under Review}
}
```
