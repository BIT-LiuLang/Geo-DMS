# Geo-DMS: Unified Multi-Task Driver Monitoring with 3D Geometric Priors

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18790509.svg)](https://doi.org/10.5281/zenodo.18790509) 

This repository holds the official implementation of the paper: **"Geo-DMS: Unified Multi-Task Driver Monitoring with 3D Geometric Priors"**.

Currently, the paper is under review at the **2026 IEEE International Conference on Systems, Man, and Cybernetics (SMC 2026)**.

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

## 📢 Code & Dataset

To enhance the transparency and reproducibility of our research, the core implementation of Geo-DMS has been archived. 
* Note: We are currently organizing the complete pre-trained weights and will continually update this repository. Please star ("⭐️") this repository to receive notifications about future updates.

## ✅ To-Do List

- [x] Release preprint (Research Square)
- [x] **Release core algorithm implementation**
- [x] Submit to IEEE SMC 2026
- [ ] Release pretrained weights
- [ ] Add detailed installation and usage instructions

## 📌 Citation

If you find this repository or our research helpful, please use the following BibTeX entry:

```bibtex
@inproceedings{liu2026geodms,
  title={Geo-DMS: Unified Multi-Task Driver Monitoring with 3D Geometric Priors},
  author={Liu, Lang and Jiang, Xiaobei and Wang, Zhengyu and Zhou, Junjun and Ma, Yixue},
  booktitle={2026 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year={2026},
  note={Under Review}
}
