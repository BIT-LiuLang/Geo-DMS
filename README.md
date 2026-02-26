# Geo-DMS: Unified Multi-Task Driver Monitoring with 3D Geometric Priors

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18790509.svg)](https://doi.org/10.5281/zenodo.18790509) [![Research Square](https://img.shields.io/badge/Preprint-Research%20Square-blue)](https://www.researchsquare.com/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <br/>

This repository holds the official implementation of the paper: **"Geo-DMS: Unified Multi-Task Driver Monitoring with 3D Geometric Priors"**.

Currently, the paper is under review at *The Visual Computer* (TVC).

## 💡 Method Overview

The architecture of the Geo-DMS framework built upon the frozen SAM 3D Body framework (DINOv3 backbone & MHR decoder), the pipeline employs an Inter-Layer Channel Aggregator (ILCA) to unify global semantics ($F_{agg}$) and a Pose-Guided Adaptive Fusion (PGAF) module to inject geometric priors via a parallel strategy.

<div align="center">
  <img src="assets/pipeline.png" width="100%" alt="Pipeline Figure"/>
  <br/>
  <em>Figure 1: The overall framework of our proposed Geo-DMS.</em>
</div>

<br/>

## 📢 Code & Dataset

To enhance the transparency and reproducibility of our research, the core implementation of Geo-DMS has been archived. 
* Note: We are currently organizing the complete pre-trained weights and will continually update this repository. Please star ("⭐️") this repository to receive notifications about future updates.

## ✅ To-Do List

- [x] Paper submission
- [x] Release preprint (Research Square)
- [x] **Release core algorithm implementation**
- [ ] Release pretrained weights
- [ ] Add detailed installation and usage instructions

## 📌 Citation

As requested by the journal editor, if you find this repository or our research helpful, please consider citing our related manuscript currently under review at *The Visual Computer*:

```bibtex
@article{liu2026geodms,
  title={Geo-DMS: Unified Multi-Task Driver Monitoring with 3D Geometric Priors},
  author={Liu, Lang and Jiang, Xiaobei and Guo, Hongwei and Wang, Zhengyu},
  journal={The Visual Computer},
  year={2026},
  note={Under Review}
}
