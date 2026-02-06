#!/bin/bash
set -e

eval "$(conda shell.bash hook)"

conda create -n dplora python=3.11 -y
conda activate dplora;
pip install -r requirements_lora.txt;
cd dm-lora/fast-differential-privacy; pip install -e .;