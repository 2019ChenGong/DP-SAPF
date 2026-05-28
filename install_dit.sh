#!/bin/bash
set -e

eval "$(conda shell.bash hook)"

conda create -n dplora_dit python=3.12 -y
conda activate dplora_dit;
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126;
git clone https://github.com/huggingface/diffusers;
cd diffusers;
pip install -e .;
cd ..;
pip install -r requirements_dit.txt;
cd dm-lora/fast-differential-privacy; pip install -e .;