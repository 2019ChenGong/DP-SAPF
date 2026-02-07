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
    - [3.3 Training](#33-training)<div align=center>

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

Please run the following commands to install the environments for training and evaluation:
 ```
bash install.sh
bash install_lora.sh
 ```

### 3.2 Dataset and Files Preparation

Preprocess dataset.
```
bash data_preparation.sh
```

After running, we can find the folder `dataset`:

  ```plaintext
dataset/                                  
├── camelyon/       
├── celeba/ 
├── cifar10/ 
...
```

We list the studied datasets as follows in our paper, which include four sensitive datasets.
  | Usage |  Dataset  |
  | ------- | --------------------- |
  | Sensitive dataset | CIFAR-10, OCTMNIST, CelebA, Camelyon |


We list the studied public models as follows in our paper, which include four public models.
  | Usage |  Model Name  |
  | ------- | --------------------- |
  | Public Model | Stable-Diffusion-v1-5, Stable-Diffusion-2-1-base, Realistic-v6, Prompt2med |

The public models will be downloaded automatically when runing the training codes as follows.



### 3.3 Training
Please run:
```
bash scripts/script-dp-sapf.sh
```

After training, the synthetic images will be saved in `exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen`.

To evaluate DP-SAPF on other sensitive datasets, please edit dataset name and dataset folder in line 4-5 in `scripts/script-dp-sapf.sh`. The dataset folder can be found in `/dataset`.

For baselines PE, DP-Finetune, DP-LoRA, and DP-LDM, please run:

```
bash scripts/script-pe.sh
bash scripts/script-dp-finetune.sh
bash scripts/script-dp-lora.sh
bash scripts/script-dp-ldm.sh
```

Users can also edit `MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"` in each script for other public models. The following three public models are used in our experiments:
```
Manojb/stable-diffusion-2-1-base
Nihirc/Prompt2MedImage
SG161222/Realistic_Vision_V6.0_B1_noVAE
```


### 3.4 Evaluation

Please run:
```
conda activate dpimagebench
python eval.py -dn cifar10_32 -ep exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen
```
The FID and Acc on the testset will be saved into `exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/stdout.txt`.

For baselines, you just need to change `-ep` into the corresponding output directory like
```
conda activate dpimagebench
python eval.py -dn cifar10_32 -ep <output-dir>
```
    - [3.4 Evaluation](#34-evaluation)

## 2. Introduction

Differentially private (DP) image synthesis generates images that preserve the statistical characteristics of a sensitive dataset, enabling sensitive data analysis and usage while providing rigorous guarantees of privacy leakage. Existing methods typically fine-tune public models using DP Stochastic Gradient Descent (DP-SGD) on sensitive images to generate synthetic images. But full fine-tuning public models on sensitive datasets is computationally expensive and time-consuming, because current public models typically contain a large number of parameters. Existing methods propose using Low-Rank Adaptation (LoRA) to reduce the number of trainable parameters. However, we argue that exhaustive LoRA coverage across all public model layers is suboptimal in a DP setting, as it leads to excessive noise accumulation and is detrimental to training stability. 

To address this issue, we propose DP-SAPF, which uses a saliency-aware strategy to identify specific target parameters for LoRA training under DP. DP-SAPF is inspired by the fact that larger gradients signify higher saliency, indicating that these parameters are most critical for the DP learning. Specifically, we feed the sensitive images into public models, compute gradients, and add noise to the gradients to satisfy DP. Then, DP-SAPF identifies the most salient parameters, those exhibiting high gradient magnitudes on sensitive images, for DP fine-tuning.

## 3. Get Start
We provide an example for how to reproduce the results on CIFAR-10 in our paper.

### 3.1 Installation

To set up the environment of DP-SAPF, we use `conda` to manage our dependencies. 

Please run the following commands to install the environments for training and evaluation:
 ```
bash install.sh
bash install_lora.sh
 ```

### 3.2 Dataset and Files Preparation

Preprocess dataset.
```
bash data_preparation.sh
```

After running, we can find the folder `dataset`:

  ```plaintext
dataset/                                  
├── camelyon/       
├── celeba/ 
├── cifar10/ 
...
```

We list the studied datasets as follows in our paper, which include four sensitive datasets.
  | Usage |  Dataset  |
  | ------- | --------------------- |
  | Sensitive dataset | CIFAR-10, OCTMNIST, CelebA, Camelyon |


We list the studied public models as follows in our paper, which include four public models.
  | Usage |  Dataset  |
  | ------- | --------------------- |
  | Sensitive dataset | Stable-Diffusion-v1-5, Stable-Diffusion-2-1-base, Realistic-v6, Prompt2med |

The public models will be downloaded automatically when runing the training codes as follows.



### 3.3 Training
Please run:
```
bash scripts/script-dp-sapf.sh
```

After training, the synthetic images will be saved in `exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen`.

To evaluate DP-SAPF on other sensitive datasets, please edit dataset name and dataset folder in line 4-5 in `scripts/script-dp-sapf.sh`. The dataset folder can be found in `/dataset`.

Users can also edit `MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"` in line 14 for other public models.

### 3.4 Evaluation

Please run:
```
conda activate dpimagebench
python eval.py -dn cifar10_32 -ep exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/gen
```
The FID and Acc on the testset will be saved into `exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/stdout.txt`.
