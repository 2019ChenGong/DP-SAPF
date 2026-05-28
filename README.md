<div align=center>

# DP-SAPF: Saliency-Aware Parameter Fine-tuning of Public Models for Differentially Private Image Synthesis
</div>

This is the official implementation of DP-SAPF. DP-SAPF proposes a saliency-aware strategy to identify specific target parameters for LoRA training under differential privacy (DP). By selecting only the most salient parameter matrices — those with the highest gradient magnitudes on sensitive images — DP-SAPF reduces noise accumulation and avoids the training collapse that occurs when all attention layers are fine-tuned with DP-SGD.

Experiments on four sensitive image datasets show that DP-SAPF improves the utility and fidelity of synthetic images while requiring fewer computational resources than fine-tuning methods without parameter selection.

## 1. Contents
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
    <!-- - [2.1 Baselines](#21-baselines) -->
    - [2.1 Investigated Datasets](#22-investigated-datasets)
    - [2.2 Public Models](#23-public-models)
  - [3. Repo Contents](#3-repo-contents)
  - [4. Quick Start](#4-quick-start)
    - [4.1 Installation](#41-installation)
    - [4.2 Prepare Dataset](#42-prepare-dataset)
    - [4.3 Running](#43-running)
      <!-- - [4.3.1 Key Hyper-parameter Introductions](#431-key-hyper-parameter-introductions) -->
      - [4.3.1 How to Run (RQ1: Main Results)](#432-how-to-run-rq1-main-results)
      - [4.3.2 How to Run (RQ2: Strengths of Parameter Selection)](#433-how-to-run-rq2-strengths-of-parameter-selection)
      - [4.3.3 How to Run (RQ3: Hyper-parameter Analysis)](#434-how-to-run-rq3-hyper-parameter-analysis)
      - [4.3.4 How to Run (Discussions)](#435-how-to-run-discussions)
    - [4.4 Results](#44-results)
      - [4.4.1 Results Structure](#441-results-structure)
      - [4.4.2 Results Explanation](#442-results-explanation)
    <!-- - [4.5 Results Visualization](#45-results-visualization) -->
  - [Contacts](#contacts)
  <!-- - [Citation](#citation) -->
  - [Acknowledgement](#acknowledgement)

## 2. Introduction

<!-- ### 2.1 Baselines

We compare DP-SAPF against five DP image synthesis methods that leverage public models.

| Methods | Type | Link |
| ------- | ---- | ---- |
| PE | Fine-tuning-free | [\[NeurIPS 2023\] Differentially Private Synthetic Data via Foundation Model APIs](https://arxiv.org/abs/2305.09515) |
| AUG-PE | Fine-tuning-free | [\[ICML 2024\] Differentially Private Synthetic Data via APIs with Enhanced Mechanisms](https://arxiv.org/abs/2403.01749) |
| DP-LDM | Fine-tuning-based | [\[TODO\] Differentially Private Latent Diffusion Models](TODO) |
| DP-LoRA | Fine-tuning-based | [\[TODO\] DP fine-tuning with Low-Rank Adaptation](TODO) |
| DP-Finetune | Fine-tuning-based | [\[TODO\] Differentially Private Diffusion Models](TODO) | -->

### 2.1 Investigated Datasets

We perform experiments on four sensitive image datasets.

| Dataset | Training | Validation | Test | Resolution | Categories |
| ------- | -------- | ---------- | ---- | ---------- | ---------- |
| CIFAR-10 | 45,000 | 5,000 | 10,000 | 32×32 | 10 |
| OCTMNIST | 97,477 | 10,832 | 1,000 | 128×128 | 4 |
| CelebA | 162,770 | 19,867 | 19,962 | 256×256 | 2 (Gender) |
| Camelyon | 302,436 | 34,904 | 85,054 | 96×96 | 2 (Tumor) |

### 2.2 Public Models

We evaluate DP-SAPF on four widely-used public diffusion models.

| Public Model | Source | Resolution | Size | Year |
| ------------ | ------ | ---------- | ---- | ---- |
| [Stable-Diffusion-v1-5](https://huggingface.co/Manojb/stable-diffusion-2-1-base) | Stability AI | 512×512 | 1B | 2022 |
| [Stable-Diffusion-2-1-base](https://huggingface.co/Manojb/stable-diffusion-2-1-base) | Stability AI | 512×512 | 1B | 2022 |
| [Realistic-v6](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE) | Hugging Face | 896×896 | 1B | 2024 |
| [Prompt2med](https://huggingface.co/Nihirc/Prompt2MedImage) | Hugging Face | 512×512 | 1B | 2024 |

The public models will be downloaded automatically when running the training scripts.

## 3. Repo Contents

```plaintext
gap/
├── configs/            # Configuration files for DP image synthesis algorithms
├── data/               # Data preparation scripts
├── dataset/            # Datasets studied in the project
├── dm-lora/            # LoRA training for diffusion models
├── docker/             # Docker file
├── exp/                # Training outputs and evaluation results
├── evaluation/         # Evaluation module (utility and fidelity)
│   └── evaluator.py
├── models/             # DP image synthesis algorithm implementations
├── opacus/             # DP-SGD implementation
├── plot/               # Figures and plotting scripts
├── scripts/            # Training and baseline scripts
│   ├── script-dp-sapf.sh
│   ├── script-pe.sh
│   ├── script-augpe.sh
│   ├── script-dp-finetune.sh
│   ├── script-dp-lora.sh
│   └── script-dp-ldm.sh
├── utils/              # Helper functions
├── dm-lora/            # Main training code
├── eval.py             # Evaluation entry
├── cal_privacy.py      # Privacy budget (RDP cost ratio) calculation
└── requirements.txt
```

## 4. Quick Start

### 4.1 Installation

To set up the environment of DP-SAPF, we use `conda` to manage our dependencies.

```bash
git clone https://github.com/2019ChenGong/DP-SAPF.git
cd DP-SAPF
bash install.sh
bash install_lora.sh
bash install_dit.sh
```

### 4.2 Prepare Dataset

```bash
conda activate dpimagebench
bash data_preparation.sh
```

After running, we can find the folder `dataset`:

```plaintext
dataset/
├── camelyon/
├── celeba/
├── cifar10/
├── octmnist/
...
```

### 4.3 Running

#### 4.3.1 How to Run (RQ1: Main Results)

Users should first activate the conda environment:

```bash
conda activate dplora
```

Train DP-SAPF (example: CIFAR-10, 4 GPUs):

```bash
bash scripts/script-dp-sapf.sh
```

Note: For diffusers=0.21.0, you will get ImportError: cannot import name 'cached_download' from 'huggingface_hub' error. To solve it please remove the line from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info in dyanamic_models_utils.py script.

<!-- Train baselines:

```bash
bash scripts/script-pe.sh
bash scripts/script-augpe.sh
bash scripts/script-dp-finetune.sh
bash scripts/script-dp-lora.sh
bash scripts/script-dp-ldm.sh
``` -->

Users can edit `MODEL_NAME` in each script to switch public models:

```
stable-diffusion-v1-5/stable-diffusion-v1-5
Manojb/stable-diffusion-2-1-base
SG161222/Realistic_Vision_V6.0_B1_noVAE
Nihirc/Prompt2MedImage
```

Evaluate (change `-dn` and `-ep` for other datasets/methods):

```bash
python eval.py -dn cifar10_32 -ep <output-dir>
```

<!-- For **Figure (fig:synthetic_real)**, please refer to `plot/visualization.py`, `plot/visualization1.py`, and `plot/visualization2.py`. -->

#### 4.3.2 How to Run (RQ2: Strengths of Parameter Selection)

RQ2 uses `Stable-Diffusion-v1-5` as the public model and $\varepsilon=10.0$. We compare DP-SAPF against the following five variants:

- **Random**: randomly selects parameter matrices without saliency-aware mechanism.
- **Noisy**: replaces unselected parameters with random values to measure the contribution of the public model's generative capability.
- **w/o LoRA**: fine-tunes the selected parameter matrices directly with DP-SGD, without LoRA.
- **Layer-Level**: performs layer-wise selection instead of matrix-wise selection.
- **All Parameter**: applies saliency-aware selection to all fine-tunable parameter matrices (not only attention layers).

Run each variant:

```bash
bash scripts/script-dp-sapf-all.sh
bash scripts/script-dp-sapf-layer.sh
bash scripts/script-dp-sapf-noisy.sh
bash scripts/script-dp-sapf-nolora.sh
bash scripts/script-dp-sapf-random.sh
```

For **Figure (fig:rq2)**, please refer to `plot/ablation.py`.
For parameter selection visualization, please refer to `plot/vis_selection.py`.

#### 4.3.3 How to Run (RQ3: Hyper-parameter Analysis)

All RQ3 experiments use `Stable-Diffusion-v1-5`. Two sensitivity axes are evaluated:

**Selection ratio** $c \in \{0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7\}$ under $\varepsilon=10.0$:

```bash
bash scripts/script-dp-sapf-topk.sh
```

**Privacy budget** $\varepsilon \in \{0.2, 1.0, 5.0, 10.0, 15.0, 20.0\}$:

```bash
bash scripts/script-dp-sapf-eps.sh
```

**Noise scale** $\sigma_s \in \{5.0, 10.0, 20.0, 25.0\}$:

```bash
bash scripts/script-dp-sapf-sigma.sh
```

<!-- To compute RDP cost ratios (parameter-selection / DP-SGD):

```bash
python cal_privacy.py --method DP-SAPF --data_name celeba_male_256 -e 10.0 train.sigma_s=5.0
``` -->

For **Figure (fig:privacy_budget)**, please refer to `plot/plot_param_change.py` and `plot/plot_selection_ratio_combined.py`.

#### 4.3.3 How to Run (Discussions)

**Non-private setting** (*Table tab:no_dp* — DP-SAPF at $\varepsilon=\infty$):

```bash
bash scripts/script-dp-sapf-nodp.sh
```

**Without fine-tuning** (*Table tab:no_finetuning* — public model zero-shot vs. DP-SAPF fine-tuned):

Generate directly from the pretrained public model without any fine-tuning on sensitive data:

```bash
bash scripts/script-dp-dit.sh
```

**Transferability** (*Figure fig:dit* — DP-SAPF on DiT and alternative DP mechanisms EM / PTR):

For DiT:
```bash
bash scripts/script-dp-sapf-dit.sh
```

For EM and PTR:
```bash
bash scripts/script-dp-em.sh
bash scripts/script-dp-ptr.sh
```

For the transferability figure, please refer to `plot/plot_models_mechanism.py`.


### 4.4 Results

The results are recorded in `stdout.txt` inside each experiment folder.

#### 4.4.1 Results Structure

DP-SAPF produces a two-level output. The top-level folder `lora_{dataset}_{bs}_{steps}_eps{eps}` contains a subfolder for the parameter-selection stage and another for the DP fine-tuning and generation stage.

```plaintext
exp/
├── lora_cifar10_32_4096bs_1ksteps_eps10/              # OUTPUT_DIR
│   └── lora_k4q4v4o4_base_top0.3_fs5_finegrained_0.0005/  # DP fine-tuning + generation
│       ├── gen/
│       │   ├── syn_images.npy                         # synthetic images
│       │   ├── syn_labels.npy                         # synthetic labels
│       │   └── sample.png                             # sample grid
│       ├── pytorch_lora_weights.safetensors           # trained LoRA weights
│       ├── tuning_layers.json                         # selected parameter matrices
│       ├── samples/
│       │   └── validation_samples_*.png               # training progress samples
│       ├── log.txt
│       └── stdout.txt                                 # evaluation results
...
```

#### 4.4.2 Results Explanation

The following results can be found at the end of `stdout.txt`:

```
INFO - evaluator.py - The FID of synthetic images is XX
INFO - evaluator.py - The Inception Score of synthetic images is XX
INFO - evaluator.py - The Precision and Recall of synthetic images is XX and XX
INFO - evaluator.py - The FLD of synthetic images is XX
INFO - evaluator.py - The best acc of accuracy (adding noise to the results on the sensitive set of validation set) of synthetic images from resnet, wrn, and resnext are [XX, XX, XX].
INFO - evaluator.py - The average and std of accuracy of synthetic images are XX and XX
```

The synthetic images can be found at `./exp/lora_{dataset}_{bs}_{steps}_eps{eps}/lora_k4q4v4o4_{lower_name}_{lr}/gen/`.

<!-- ### 4.5 Results Visualization

We provide the plotting codes in the folder `plot/`.

- `visualization.py` / `visualization1.py` / `visualization2.py`: synthetic vs. real image grids (Figure fig:synthetic_real, RQ1).
- `ablation.py`: ablation figure comparing DP-SAPF vs. five variants (Figure fig:rq2, RQ2).
- `vis_selection.py`: visualization of selected parameter matrices across layers.
- `plot_param_change.py`: FID and Acc under varying privacy budget and selection ratio (Figure fig:privacy_budget, RQ3).
- `plot_selection_ratio_combined.py`: combined selection ratio analysis (RQ3).
- `plot_models_mechanism.py`: transferability to DiT and alternative DP mechanisms (Figure fig:dit, Discussion). -->

## Contacts

If you have any question about our work or this repository, please don't hesitate to contact us by email or open an issue.

- Chen Gong (ChenG_abc@outlook.com)

- Kecen Li (sunameizing@gmail.com)

<!-- ## Citation

```text
@article{TODO,
  title={DP-SAPF: Saliency-Aware Parameter Fine-tuning of Public Models for Differentially Private Image Synthesis},
  author={TODO},
  journal={TODO},
  year={2025}
}
``` -->

## Acknowledgement

Part of the code is borrowed from [DPImageBench](https://github.com/2019ChenGong/DPImageBench). We sincerely thank them for their contributions to the community.
