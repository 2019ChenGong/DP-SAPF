#!/bin/bash
set -e

eval "$(conda shell.bash hook)"

conda create -n dplora python=3.11 -y
conda activate dplora;
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126;
pip install -r requirements_lora.txt;
cd dm-lora/fast-differential-privacy; pip install -e .;