# SD2-Mamba: Semantic-Density-Driven Mamba for Robust Domain Generalization Underwater Object Detection

## Introduction

This repository contains the code for our paper `SD2-Mamba: Semantic-Density-Driven Mamba for Robust Domain Generalization Underwater Object Detection` (ICASSP 2026)

Domain generalization for object detection aims to enhance detectors’ performance in unseen scenarios. Many existing methods for domain generalization in underwater object detection fail to
effectively separate foreground objects from noisy backgrounds, as they overlook semantic density cues crucial for distinguishing salient structures under severe visual degradation. In this paper,
we propose SD2-Mamba, a novel semantic density–driven state space model framework tailored for robust domain generalization in underwater object detection. First, we propose an Online Den
sity Peak Clustering (ODPC) module to estimate the importance of foreground regions, guiding the model’s attention towards the most relevant areas. Then, to mitigate the interference from noise
and background, the Density-aware Sequence Modulation (DSM) mechanism is proposed to adaptively adjust the sampling step and state gain of Mamba, preserving high-frequency structures in dense
regions while suppressing noisy background. Finally, we introduce Semantically Enhanced Instance Normalization (SE-IN) to recalibrate feature maps at both spatial and channel levels, enhancing
foreground saliency. Extensive experiments on the SUODAC2020 dataset demonstrate that SD2-Mamba achieves state-of-the-art performance, with an average mAP50 of 63.9%, while being highly
efficient with only 6.9M parameters and 147.1 FPS. It outperforms existing domain generalization methods in both accuracy and speed.

<img width="631" height="374" alt="image" src="https://github.com/user-attachments/assets/55cc4786-417e-4e08-8126-066c199c1e36" />

<img width="687" height="184" alt="image" src="https://github.com/user-attachments/assets/b80e6011-64c9-40b1-81bc-9e99b7622abc" />


## Getting started
### 1. Installation

SD2-Mamba is developed based on `torch==2.4.0` `pytorch-CUDA==12.4` and `CUDA-version==12.4`

### 2. Clone Project

```
git clone https://github.com/Ixiaohuihuihui/SD2-Mamba.git
```

### 3. Create and activate a conda environment

```
conda create -n SD2_Mamba -y python=3.11
conda activate SD2_Mamba
```

### 4. Install torch

```
pip install torch===2.4.0 torchvision torchaudio
```

### 5. Install Dependencies

```
pip install seaborn thop timm einops
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

### 6. Prepare S-UODAC2020 Dataset

Make sure your dataset structure as follows:
```
├── S-UODAC2020
│   ├── annotations
│   │   ├── instances_soure_train.json
│   │   └── instances_soure_val.json
|   |   └── instance_target.json
│   ├── images
│   │   ├── train
│   │   └── val
|   |   └── test
│   ├── labels
│   │   ├── train
│   │   └── val
|   |   └── test
```

### 7. Training

```
python train.py --data ultralytics/cfg/datasets/S-UODAC.yaml --config ultralytics/cfg/models/SD2-MAMBA/SD2-Mamba.yaml --amp  
```

## Acknowledgement

Our codebase is developed upon [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO.git). We thank the Mamba-YOLO authors for releasing their implementation and providing a solid foundation for our work. 
Mamba-YOLO builds on [Ultralytics](https://github.com/ultralytics/ultralytics) and the selective-scan from [VMamba](https://github.com/MzeroMiko/VMamba). 
