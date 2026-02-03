<div align=center>

# DP-SAPF: Saliency-Aware Parameter Fine-tuning of Public Models for Differentially Private Image Synthesis
</div>

This is the official implementation of paper ***DP-SAPF: Saliency-Aware Parameter Fine-tuning of Public Models for Differentially Private Image Synthesis***. This repository contains Pytorch training code and evaluation code. DP-SAPF is a Differetial Privacy (DP) image generation tool, which leverages the DP technique to generate synthetic data to replace the sensitive data, allowing organizations to share and utilize synthetic images without privacy concerns.


## 1. Contents
- DP-SAPF: Saliency-Aware Parameter Fine-tuning of Public Models for Differentially Private Image Synthesis
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
  - [3. Get Start](#3-get-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset and Files Preparation](#32-dataset-and-files-preparation)
    - [3.3 Training](#33-training)
    - [3.4 Evaluation](#34-evaluation)

## 2. Introduction

Differentially private (DP) image synthesis generates images that preserve the statistical characteristics of a sensitive dataset, enabling sensitive data analysis and usage while providing rigorous guarantees of privacy leakage. Existing methods typically fine-tune public models using DP Stochastic Gradient Descent (DP-SGD) on sensitive images to generate synthetic images. But full fine-tuning public models on sensitive datasets is computationally expensive and time-consuming, because current public models typically contain a large number of parameters. Existing methods propose using Low-Rank Adaptation (LoRA) to reduce the number of trainable parameters. However, we argue that exhaustive LoRA coverage across all public model layers is suboptimal in a DP setting, as it leads to excessive noise accumulation and is detrimental to training stability. 

To address this issue, we propose DP-SAPF, which uses a saliency-aware strategy to identify specific target parameters for LoRA training under DP. DP-SAPF is inspired by the fact that larger gradients signify higher saliency, indicating that these parameters are most critical for the DP learning. Specifically, we feed the sensitive images into public models, compute gradients, and add noise to the gradients to satisfy DP. Then, DP-SAPF identifies the most salient parameters, those exhibiting high gradient magnitudes on sensitive images, for DP fine-tuning.

## 3. Get Start
We provide an example for how to reproduce the results on CIFAR-10 in our paper.

### 3.1 Installation

To set up the environment of DP-SAPF, we use `conda` to manage our dependencies. 

Run the following commands to install the environments for training and evaluation:
 ```
bash install.sh
bash install_lora.sh
 ```

### 3.2 Dataset and Files Preparation

Preprocess dataset.
```
bash data_preparation.sh
```

### 3.3 Training
Run:
```
bash scripts/script-dp-sapf.sh
```

After training, the synthetic images will be saved in `exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen`.

### 3.4 Evaluation

Run:
```
conda activate dpimagebench
python eval.py -dn cifar10_32 -ep exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen
```
The FID and Acc on the testset will be saved into `exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/stdout.txt`.


## Todo list

### Writing

- Motivation

### Experiments

- Table 6 epsilon = \infty

- Hyper-parameter analysis: $\epsilon = \{0.2, 1.0, 5.0, 10, 15, 20\}$ for cifar10, celeba, SD-v1-5;  Selection ratio $ p = \{0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8\}$ for cifar10, celeba, SD-v1-5.

- Abalation studies in Table 5.

- visulization


## Acknowledgement
 
Part of code is borrowed from [DPImageBench](https://github.com/2019ChenGong/DPImageBench). We sincerely thank them for their contributions to the community.
