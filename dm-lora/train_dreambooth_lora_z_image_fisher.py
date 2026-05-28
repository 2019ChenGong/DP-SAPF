#!/usr/bin/env python
# coding=utf-8
"""Fisher-information based transformer block selection for Z-Image LoRA fine-tuning.

Workflow
--------
1. Load ZImageTransformer2DModel (no LoRA adapters).
2. Run N forward/backward passes over the training images using the flow-matching loss.
3. Accumulate squared gradients (grad²) per transformer block index across
   all attention projection parameters (to_q, to_k, to_v).
4. Optionally add calibrated DP noise to the Fisher scores.
5. Select the top-K fraction of blocks by score.
6. Write a JSON file consumable by train_dreambooth_lora_z_image_dp.py via --lora_layers_json:
       {
         "layers_to_transform": [0, 3, 7, ...],
         "layers_pattern": "transformer_blocks",
         "target_modules": ["to_k", "to_q", "to_v"]
       }

Usage example
-------------
    python train_dreambooth_lora_z_image_fisher.py \
        --pretrained_model_name_or_path Tongyi-MAI/Z-Image \
        --instance_data_dir ./my_images \
        --instance_prompt "a photo of sks dog" \
        --output_dir ./fisher_out \
        --resolution 512 \
        --train_batch_size 1 \
        --fisher_num_batches 100 \
        --fisher_norm 1.0 \
        --fisher_sigma 5.0 \
        --top_k_lora 0.3 \
        --target_modules "to_k,to_q,to_v"

Then pass the result to the training script:
    python train_dreambooth_lora_z_image_dp.py \
        ... \
        --lora_layers_json ./fisher_out/zimage/fisher_layers.json \
        --eps 8.0
"""

import argparse
import copy
import json
import logging
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration, set_seed
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import Qwen2Tokenizer, Qwen3Model

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    free_memory,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_block_index(param_name: str):
    """Return (block_idx, proj_name) for any param in a numbered block layer containing a target projection."""
    match = re.search(r'\.(\d+)\.', param_name)
    if match is None:
        return None
    block_idx = int(match.group(1))
    for proj in ["to_q", "to_k", "to_v"]:
        if proj in param_name:
            return block_idx, proj
    return None


def detect_block_container(transformer, target_modules):
    """Auto-detect the nn.ModuleList name that holds the main transformer blocks."""
    container_max = defaultdict(int)
    for pname, _ in transformer.named_parameters():
        if not any(t in pname for t in target_modules):
            continue
        m = re.match(r'^(.*?)\.(\d+)\.', pname)
        if m:
            container = m.group(1)
            idx = int(m.group(2))
            container_max[container] = max(container_max[container], idx + 1)
    if not container_max:
        return "transformer_blocks"
    return max(container_max, key=container_max.get)


def get_sigmas_fn(noise_scheduler_copy, accelerator, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FolderImageDataset(Dataset):
    """Simple image folder dataset for Fisher estimation (folder of images only)."""

    VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root, prompt, size=512, center_crop=False, repeats=1):
        self.root = Path(root)
        self.prompt = prompt
        paths = sorted(p for p in self.root.iterdir() if p.suffix.lower() in self.VALID_EXT)
        if not paths:
            raise ValueError(f"No images found in {root}")

        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        self.pixel_values = []
        for path in paths:
            img = Image.open(path)
            img = exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = train_resize(img)
            if center_crop:
                img = train_crop(img)
            else:
                y1, x1, h, w = train_crop.get_params(img, (size, size))
                img = crop(img, y1, x1, h, w)
            self.pixel_values.append(to_tensor(img))

        self.pixel_values = self.pixel_values * repeats

    def __len__(self):
        return len(self.pixel_values)

    def __getitem__(self, index):
        return self.pixel_values[index], self.prompt


_BENCH_TRAIN_SIZES = {
    "mnist": 55000,
    "fmnist": 55000,
    "cifar": 45000,
    "eurosat": 21000,
    "celeba": 145064,
    "camelyon": 269538,
    "covidx": 67863,
    "octmnist": 97477,
}


class BenchImageDataset(Dataset):
    """Mirrors dataset.BenchDataset: per-label prompts via get_prompt(name).

    --instance_data_dir is the dataset name (e.g. "celeba_male_256") used to
    resolve per-class text prompts via get_prompt().
    """

    def __init__(self, bench_path, name, size=512, center_crop=False):
        from dataset_bench import ImageFolderDataset as _BenchIFD, get_prompt
        self._ds = _BenchIFD(bench_path, 3, use_labels=True)
        self.prompt_list = get_prompt(name)

        train_size = next((v for k, v in _BENCH_TRAIN_SIZES.items() if k in name), None)
        if train_size is not None and len(self._ds) > train_size:
            val_size = len(self._ds) - train_size
            torch.manual_seed(0)
            self._ds, _ = torch.utils.data.random_split(self._ds, [train_size, val_size])

        self._size = size
        self._center_crop = center_crop
        self._resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self._crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self._to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        image, label = self._ds[index]  # PIL.Image, int label
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self._resize(image)
        if self._center_crop:
            image = self._crop(image)
        else:
            y1, x1, h, w = transforms.RandomCrop.get_params(image, (self._size, self._size))
            image = crop(image, y1, x1, h, w)
        return self._to_tensor(image), self.prompt_list[int(label)]


def collate_fn(examples):
    pixel_values, prompts = zip(*examples)
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values, "prompts": list(prompts)}


# ---------------------------------------------------------------------------
# Fisher computation
# ---------------------------------------------------------------------------

def compute_fisher_for_zimage(
    transformer,
    vae,
    noise_scheduler_copy,
    train_dataloader,
    accelerator,
    args,
    num_batches,
    weight_dtype,
    prompt_embeds_cache,  # dict[prompt_str -> List[Tensor]] pre-computed per class
    vae_shift_factor,
    vae_scaling_factor,
    target_modules,
    random_selection=False,
):
    """Accumulate squared gradients per transformer block for the given target modules."""
    fisher = defaultdict(float)

    if random_selection:
        all_indices = set()
        for pname, _ in transformer.named_parameters():
            r = infer_block_index(pname)
            if r is not None:
                all_indices.add(r[0])
        for i in all_indices:
            fisher[i] = float(np.random.rand())
        return fisher

    transformer.requires_grad_(False)
    for name, param in transformer.named_parameters():
        result = infer_block_index(name)
        if result is None:
            continue
        if any(t in name for t in target_modules):
            param.requires_grad_(True)

    n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        logger.info(f"Fisher: {n_trainable:,} trainable parameters restricted to {target_modules}")

    transformer, train_dataloader = accelerator.prepare(transformer, train_dataloader)

    progress_bar = tqdm(range(num_batches), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Computing Fisher")

    for step, batch in enumerate(train_dataloader):
        if step >= num_batches:
            break

        pixel_values = batch["pixel_values"].to(
                            accelerator.device, non_blocking=True, dtype=vae.dtype
                        )

        with torch.no_grad():
            latent_dist = vae.encode(pixel_values).latent_dist
            latents = latent_dist.mode()
            latents = (latents - vae_shift_factor) * vae_scaling_factor
            latents = latents.to(dtype=weight_dtype)

        bsz_physical = latents.shape[0]

        # Build per-sample prompt embeddings from the pre-computed cache
        pe = [prompt_embeds_cache[p][0] for p in batch["prompts"]]

        if args.micro_batch_size > 1:
            latents = latents.repeat_interleave(args.micro_batch_size, dim=0)
            pe = [emb for emb in pe for _ in range(args.micro_batch_size)]

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        u = torch.rand(bsz, device=latents.device)
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long().cpu()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)
        timestep_normalized = (1000 - timesteps) / 1000

        sigmas = get_sigmas_fn(noise_scheduler_copy, accelerator, timesteps,
                               n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        # Z-Image 5D forward pass
        noisy_latents_5d = noisy_latents.unsqueeze(2)              # (B, C, H, W) -> (B, C, 1, H, W)
        noisy_latents_list = list(noisy_latents_5d.unbind(dim=0))  # List of (C, 1, H, W)

        transformer.zero_grad()
        model_pred_list = transformer(
            noisy_latents_list,
            timestep_normalized,
            pe,
            return_dict=False,
        )[0]
        model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)  # (B, C, H, W)
        model_pred = -model_pred  # Z-Image negates the prediction

        target = noise - latents
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)

        torch.nn.utils.clip_grad_norm_(
            [p for p in transformer.parameters() if p.requires_grad],
            max_norm=args.fisher_norm,
        )

        for name, param in transformer.named_parameters():
            if param.grad is None:
                continue
            result = infer_block_index(name)
            if result is None:
                continue
            block_idx, _ = result
            fisher[block_idx] += (param.grad ** 2).sum().item()

        transformer.zero_grad()
        progress_bar.update(1)

    # Gather across processes and optionally add DP noise
    final = {}
    for block_idx in sorted(fisher.keys()):
        local_val = torch.tensor(fisher[block_idx], device=accelerator.device)
        gathered = accelerator.gather(local_val)
        total = gathered.sum().cpu().item()
        if args.fisher_sigma is not None:
            total += float(np.random.randn()) * args.fisher_sigma * args.fisher_norm
        final[block_idx] = total
        if accelerator.is_main_process:
            logger.info(f"  block {block_idx:3d}: Fisher score = {total:.4e}")

    return final


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fisher block selection for Z-Image LoRA.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--instance_data_dir", type=str, default=None,
                        help="Folder containing training images. When --bench_path is set, this is "
                             "used as the dataset name (e.g. 'celeba_male_256') for get_prompt().")
    parser.add_argument("--bench_path", type=str, default=None,
                        help="Path to zip file or folder via dataset_bench.ImageFolderDataset. "
                             "Requires --instance_data_dir to specify the dataset name.")
    parser.add_argument("--instance_prompt", type=str, required=True,
                        help="Prompt used for all images (same as training).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Repeat each sample this many times (same as in dp training).")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--max_sequence_length", type=int, default=512)

    # Fisher
    parser.add_argument("--fisher_num_batches", type=int, default=200,
                        help="Number of batches to use for Fisher estimation.")
    parser.add_argument("--fisher_norm", type=float, default=0.01,
                        help="Gradient clip norm before Fisher accumulation.")
    parser.add_argument("--fisher_sigma", type=float, default=None,
                        help="DP noise std multiplied by fisher_norm added to each block score.")
    parser.add_argument("--top_k_lora", type=float, default=0.3,
                        help="Fraction of blocks to select (0 < top_k_lora <= 1).")
    parser.add_argument("--random_selection", action="store_true",
                        help="Assign random Fisher scores (ablation baseline).")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="to_k,to_q,to_v",
        help="Comma-separated attention projection names to include in Fisher computation and LoRA.",
    )
    parser.add_argument("--fisher_save_dir", type=str, default=None,
                        help="Directory to save fisher_layers.json. Defaults to --output_dir/zimage/.")
    parser.add_argument("--fisher_save_name", type=str, default="fisher_layers.json")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    plugin = GradientAccumulationPlugin(num_steps=args.gradient_accumulation_steps, sync_with_dataloader=False)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        gradient_accumulation_plugin=plugin,
    )

    output_dir = os.path.join(args.output_dir, "zimage")
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    target_modules = [t.strip() for t in args.target_modules.split(",")]

    # ---- Load models ----
    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    text_encoder = Qwen3Model.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant,
        # ignore_mismatched_sizes=True,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant
    )

    vae_shift_factor = vae.config.shift_factor
    vae_scaling_factor = vae.config.scaling_factor

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(dtype=weight_dtype, device=accelerator.device)
    transformer.to(dtype=weight_dtype, device=accelerator.device)
    text_encoder.to(dtype=weight_dtype, device=accelerator.device)

    # ---- Dataset (built first so we know all unique prompts) ----
    if args.bench_path is not None:
        if args.instance_data_dir is None:
            raise ValueError("--bench_path requires --instance_data_dir (dataset name for get_prompt)")
        train_dataset = BenchImageDataset(
            bench_path=args.bench_path,
            name=args.instance_data_dir,
            size=args.resolution,
            center_crop=args.center_crop,
        )
        unique_prompts = train_dataset.prompt_list
    elif args.instance_data_dir is not None:
        train_dataset = FolderImageDataset(
            root=args.instance_data_dir,
            prompt=args.instance_prompt,
            size=args.resolution,
            center_crop=args.center_crop,
        )
        unique_prompts = [args.instance_prompt]
    else:
        raise ValueError("Specify either --bench_path or --instance_data_dir")

    # ---- Encode all unique class prompts, then free the text encoder ----
    text_encoding_pipeline = ZImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=None,
        revision=args.revision,
    )
    text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)

    prompt_embeds_cache = {}
    with torch.no_grad():
        for prompt in unique_prompts:
            embeds, _ = text_encoding_pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=args.max_sequence_length,
            )
            if isinstance(embeds, (list, tuple)):
                embeds = [e.to(dtype=weight_dtype, device=accelerator.device) for e in embeds]
            else:
                embeds = [embeds.to(dtype=weight_dtype, device=accelerator.device)]
            prompt_embeds_cache[prompt] = embeds

    if accelerator.is_main_process:
        logger.info(f"Encoded {len(prompt_embeds_cache)} unique class prompt(s).")

    del text_encoding_pipeline, text_encoder, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Detect block container name before accelerator.prepare wraps the model
    layers_pattern = detect_block_container(transformer, target_modules)
    if accelerator.is_main_process:
        logger.info(f"Detected block container: '{layers_pattern}'")

    num_batches = max(1, args.fisher_num_batches // max(1, accelerator.num_processes))
    if accelerator.is_main_process:
        logger.info(
            f"Computing Fisher information: {num_batches} batches/process "
            f"(random_selection={args.random_selection})"
        )

    fisher_scores = compute_fisher_for_zimage(
        transformer=transformer,
        vae=vae,
        noise_scheduler_copy=noise_scheduler_copy,
        train_dataloader=train_dataloader,
        accelerator=accelerator,
        args=args,
        num_batches=num_batches,
        weight_dtype=weight_dtype,
        prompt_embeds_cache=prompt_embeds_cache,
        vae_shift_factor=vae_shift_factor,
        vae_scaling_factor=vae_scaling_factor,
        target_modules=target_modules,
        random_selection=args.random_selection,
    )

    if accelerator.is_main_process:
        sorted_blocks = sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)
        total_blocks = len(sorted_blocks)
        k = max(1, int(args.top_k_lora * total_blocks))
        selected_block_indices = sorted([idx for idx, _ in sorted_blocks[:k]])

        logger.info(
            f"\nSelected top-{args.top_k_lora*100:.1f}% = {k}/{total_blocks} blocks: "
            f"{selected_block_indices}"
        )

        save_dir = args.fisher_save_dir or output_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, args.fisher_save_name)

        config_out = {
            "layers_to_transform": selected_block_indices,
            "layers_pattern": layers_pattern,
            "target_modules": target_modules,
            "_meta": {
                "total_blocks": total_blocks,
                "top_k_lora": args.top_k_lora,
                "fisher_num_batches": args.fisher_num_batches,
                "fisher_norm": args.fisher_norm,
                "fisher_sigma": args.fisher_sigma,
                "random_selection": args.random_selection,
                "all_scores": {str(k): v for k, v in sorted_blocks},
            },
        }
        with open(save_path, "w") as f:
            json.dump(config_out, f, indent=2)

        logger.info(f"Saved Fisher layer selection to {save_path}")
        logger.info("Use with: --lora_layers_json " + save_path)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
