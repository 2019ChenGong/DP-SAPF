#!/usr/bin/env python
# coding=utf-8
# Z-Image DreamBooth LoRA + DP-SGD training script.
# Based on train_dreambooth_lora_z_image.py with fastDP privacy engine and
# Fisher-based layer selection (--lora_layers_json).

import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from opacus.accountants.utils import get_noise_multiplier
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
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
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module
from fastDP import PrivacyEngine_Distributed_extending

if is_wandb_available():
    import wandb

check_min_version("0.39.0.dev0")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Z-Image DreamBooth LoRA (DP-SGD) - {repo_id}

These are {repo_id} DreamBooth LoRA weights for {base_model} trained with differential privacy.

Trigger word: `{instance_prompt}`
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="apache-2.0",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = ["text-to-image", "diffusers-training", "diffusers", "lora", "z-image", "template:sd-lora"]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(pipeline, args, accelerator, pipeline_args, epoch, is_final_validation=False):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(samples_dir, f"step{epoch:06d}_{i}.png"))

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images


def struct_output(args, accelerator):
    """Create and return a structured experiment output directory."""
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    exp_name = f"zimage_lora_r{args.rank}_lr{args.learning_rate}"
    if args.eps is not None:
        exp_name += f"_eps{args.eps}"
    exp_dir = os.path.join(args.output_dir, exp_name)

    if accelerator.is_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)

        log_file = os.path.join(exp_dir, "log.txt")
        if hasattr(logger, "logger"):
            actual_logger = logger.logger
        else:
            actual_logger = logger
        log_file_abs = os.path.abspath(log_file)
        already_added = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == log_file_abs
            for h in actual_logger.handlers
        )
        if not already_added:
            actual_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file_abs, mode="a", encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            actual_logger.addHandler(fh)

    return exp_dir


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Z-Image DreamBooth LoRA + DP-SGD training.")

    # ---- model / data ----
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--instance_data_dir", type=str, default=None)
    parser.add_argument(
        "--bench_path",
        type=str,
        default=None,
        help="Path to zip file or folder via dataset_bench.ImageFolderDataset. "
             "Mutually exclusive with --instance_data_dir / --dataset_name.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--class_data_dir", type=str, default=None)
    parser.add_argument("--instance_prompt", type=str, default=None, required=False)
    parser.add_argument("--class_prompt", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)

    # ---- validation ----
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=1)

    # ---- LoRA ----
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help='Comma-separated target modules, e.g. "to_k,to_q,to_v,to_out.0". Ignored if --lora_layers_json is set.',
    )
    parser.add_argument(
        "--lora_layers_json",
        type=str,
        default=None,
        help=(
            "Path to JSON produced by train_dreambooth_lora_z_image_fisher.py. "
            'Expected format: {"layers_to_transform": [0,3,7,...], "target_modules": ["to_k","to_q","to_v"]}. '
            "Takes priority over --lora_layers."
        ),
    )

    # ---- prior preservation ----
    parser.add_argument("--with_prior_preservation", default=False, action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=100)

    # ---- output ----
    parser.add_argument("--output_dir", type=str, default="zimage-dreambooth-lora-dp")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", default=False, action="store_true")
    parser.add_argument("--random_flip", action="store_true")

    # ---- training ----
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # ---- loss ----
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)

    # ---- optimizer ----
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--prodigy_beta3", type=float, default=None)
    parser.add_argument("--prodigy_decouple", type=bool, default=True)
    parser.add_argument("--prodigy_use_bias_correction", type=bool, default=True)
    parser.add_argument("--prodigy_safeguard_warmup", type=bool, default=True)

    # ---- DP-SGD ----
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Privacy budget epsilon. If None, DP is disabled.",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
        help="Number of virtual samples per physical sample (repeat_interleave).",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=1500,
        help="Total training steps used for noise multiplier computation.",
    )
    parser.add_argument(
        "--fisher_batch_size",
        type=int,
        default=None,
        help="Batch size used during Fisher pre-computation (for account_history).",
    )
    parser.add_argument("--fisher_num", type=int, default=1)
    parser.add_argument("--fisher_sigma", type=float, default=None)

    # ---- hub / logging ----
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--cache_latents", action="store_true", default=False)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--upcast_before_saving", action="store_true", default=False)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--enable_npu_flash_attention", action="store_true")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.bench_path is not None and args.dataset_name is not None:
        raise ValueError("Cannot combine --bench_path with --dataset_name")
    if args.bench_path is not None and args.instance_data_dir is None:
        raise ValueError("--bench_path requires --instance_data_dir (dataset name for get_prompt(), e.g. 'celeba_male_256')")
    if args.bench_path is None and args.instance_data_dir is None and args.dataset_name is None:
        raise ValueError("Specify one of --bench_path + --instance_data_dir, --instance_data_dir, or --dataset_name")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify --class_data_dir with --with_prior_preservation.")
        if args.class_prompt is None:
            raise ValueError("You must specify --class_prompt with --with_prior_preservation.")
    else:
        if args.class_data_dir is not None:
            warnings.warn("--class_data_dir has no effect without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("--class_prompt has no effect without --with_prior_preservation.")

    return args


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        args=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self._args = args

        if args is not None and args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Install the datasets library: `pip install datasets`.")
            dataset = load_dataset(
                args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir
            )
            column_names = dataset["train"].column_names
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(f"--image_column '{image_column}' not found in dataset columns: {column_names}")
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(f"--caption_column '{args.caption_column}' not found in dataset columns.")
                custom_instance_prompts = dataset["train"][args.caption_column]
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exist.")
            instance_images = [
                Image.open(p) for p in sorted(Path(instance_data_root).iterdir())
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            ]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        self.pixel_values = []
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if args is not None and args.random_flip and random.random() < 0.5:
                image = train_flip(image)
            if center_crop:
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (size, size))
                image = crop(image, y1, x1, h, w)
            self.pixel_values.append(train_transforms(image))

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = min(len(self.class_images_path), class_num) if class_num else len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        example["instance_images"] = self.pixel_values[index % self.num_instance_images]

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            example["instance_prompt"] = caption if caption else self.instance_prompt
        else:
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [ex["instance_images"] for ex in examples]
    prompts = [ex["instance_prompt"] for ex in examples]

    if with_prior_preservation:
        pixel_values += [ex["class_images"] for ex in examples]
        prompts += [ex["class_prompt"] for ex in examples]

    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values, "prompts": prompts}


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


class BenchDatasetWrapper(Dataset):
    """Mirrors dataset.BenchDataset with per-label prompts via get_prompt(name).

    --instance_data_dir is treated as the dataset name (e.g. "celeba_male_256")
    passed to get_prompt() to resolve per-class text prompts.
    custom_instance_prompts=True signals the training loop to encode text per batch.
    """

    def __init__(self, bench_path, name, size=512, center_crop=False):
        from dataset_bench import ImageFolderDataset as _BenchIFD, get_prompt
        self._ds = _BenchIFD(bench_path, 3, use_labels=True)
        self.prompt_list = get_prompt(name)
        self.custom_instance_prompts = True
        self.num_instance_images = len(self._ds)

        train_size = next((v for k, v in _BENCH_TRAIN_SIZES.items() if k in name), None)
        if train_size is not None and len(self._ds) > train_size:
            val_size = len(self._ds) - train_size
            torch.manual_seed(0)
            self._ds, _ = torch.utils.data.random_split(self._ds, [train_size, val_size])

        self._length = len(self._ds)
        self._size = size
        self._center_crop = center_crop
        self._resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self._crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self._to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return self._length

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
        return {
            "instance_images": self._to_tensor(image),
            "instance_prompt": self.prompt_list[int(label)],
        }


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {"prompt": self.prompt, "index": index}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError("Do not use --hub_token with --report_to=wandb. Use `hf auth login` instead.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    plugin = GradientAccumulationPlugin(
        num_steps=args.gradient_accumulation_steps, sync_with_dataloader=False
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        gradient_accumulation_plugin=plugin,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb: `pip install wandb`.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Structured output directory
    args.output_dir = struct_output(args, accelerator)

    # Prior preservation: generate class images if needed
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            pipeline = ZImagePipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process):
                images = pipeline(example["prompt"]).images
                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)
            del pipeline
            free_memory()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True
            ).repo_id

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

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError("NPU flash attention requires torch_npu extensions.")

    text_encoding_pipeline = ZImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # ---- LoRA config ----
    layers_to_transform = None
    layers_pattern = None

    if args.lora_layers_json is not None:
        with open(args.lora_layers_json, "r") as f:
            fisher_config = json.load(f)
        target_modules = fisher_config.get("target_modules", ["to_k", "to_q", "to_v"])
        layers_to_transform = fisher_config.get("layers_to_transform", None)
        layers_pattern = fisher_config.get("layers_pattern", "transformer_blocks")
        if accelerator.is_main_process:
            logger.info(f"Fisher layer selection: blocks={layers_to_transform}, modules={target_modules}")

        if layers_to_transform is not None and layers_pattern is not None:
            # PEFT's layers_to_transform regex (.*\.{pattern}\.(\d+)\.) requires a dot
            # before the block container name, which fails when the container is top-level.
            # Expand to explicit module path names instead.
            explicit_targets = set()
            for name, _ in transformer.named_modules():
                if not any(name == t or name.endswith(f".{t}") for t in target_modules):
                    continue
                for block_idx in layers_to_transform:
                    prefix = f"{layers_pattern}.{block_idx}."
                    if name.startswith(prefix) or f".{layers_pattern}.{block_idx}." in name:
                        explicit_targets.add(name)
                        break
            if explicit_targets:
                target_modules = sorted(explicit_targets)
                layers_to_transform = None
                layers_pattern = None
                if accelerator.is_main_process:
                    logger.info(f"Expanded to {len(target_modules)} explicit LoRA target modules")
    elif args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    lora_config_kwargs = dict(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    if layers_to_transform is not None:
        lora_config_kwargs["layers_to_transform"] = layers_to_transform
        lora_config_kwargs["layers_pattern"] = layers_pattern

    transformer_lora_config = LoraConfig(**lora_config_kwargs)
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ---- Checkpoint save/load hooks ----
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")
                weights.pop()
            ZImagePipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        lora_state_dict = ZImagePipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected:
                logger.warning(f"Unexpected keys when loading LoRA: {unexpected}")
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Adjust per-device batch size for DP ----
    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps / accelerator.num_processes
    )
    args.train_batch_size = max(1, args.train_batch_size)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if accelerator.is_main_process:
        logger.info(f"Trainable LoRA parameters: {sum(p.numel() for p in transformer_lora_parameters):,}")

    # ---- Optimizer ----
    params_to_optimize = [{"params": transformer_lora_parameters, "lr": args.learning_rate}]

    if args.optimizer.lower() not in ("prodigy", "adamw"):
        logger.warning(f"Unsupported optimizer: {args.optimizer}. Defaulting to AdamW.")
        args.optimizer = "adamw"

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("Install bitsandbytes: `pip install bitsandbytes`.")
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("Install prodigyopt: `pip install prodigyopt`.")
        if args.learning_rate <= 0.1:
            logger.warning("Prodigy works better with learning_rate around 1.0.")
        optimizer = prodigyopt.Prodigy(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # ---- Dataset / Dataloader ----
    if args.bench_path is not None:
        train_dataset = BenchDatasetWrapper(
            bench_path=args.bench_path,
            name=args.instance_data_dir,    # dataset name, e.g. "celeba_male_256"
            size=args.resolution,
            center_crop=args.center_crop,
        )
    else:
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            class_prompt=args.class_prompt,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_num=args.num_class_images,
            size=args.resolution,
            repeats=args.repeats,
            center_crop=args.center_crop,
            args=args,
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # ---- Text embeddings ----
    def compute_text_embeddings(prompt, pipeline):
        pipeline = pipeline.to(accelerator.device)
        with torch.no_grad():
            prompt_embeds, _ = pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=args.max_sequence_length,
            )
        if args.offload:
            pipeline = pipeline.to("cpu")
        # Normalize to a list of tensors (one per prompt in the batch)
        if not isinstance(prompt_embeds, (list, tuple)):
            prompt_embeds = list(prompt_embeds)  # iterate over batch dim
        return prompt_embeds

    if not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states = compute_text_embeddings(args.instance_prompt, text_encoding_pipeline)

    if args.with_prior_preservation:
        class_prompt_hidden_states = compute_text_embeddings(args.class_prompt, text_encoding_pipeline)

    if not train_dataset.custom_instance_prompts:
        del text_encoder, tokenizer
        free_memory()
    else:
        # Pre-encode all unique class prompts once, then free the text encoder.
        if hasattr(train_dataset, "prompt_list"):
            unique_prompts = train_dataset.prompt_list
        else:
            unique_prompts = list(set(train_dataset.custom_instance_prompts))
        prompt_embeds_dict = {}
        for prompt in unique_prompts:
            emb = compute_text_embeddings(prompt, text_encoding_pipeline)
            prompt_embeds_dict[prompt] = emb
        if accelerator.is_main_process:
            logger.info(f"Pre-encoded {len(prompt_embeds_dict)} unique prompt(s), freeing text encoder.")
        text_encoding_pipeline = text_encoding_pipeline.to("cpu")
        del text_encoder, tokenizer
        free_memory()

    # For the static single-prompt case, build the base embedding list
    if not train_dataset.custom_instance_prompts:
        # instance_prompt_hidden_states is a list of 1 tensor
        base_prompt_embeds = instance_prompt_hidden_states
        if args.with_prior_preservation:
            class_base = class_prompt_hidden_states

    # ---- Latent caching ----
    if args.cache_latents:
        latents_cache = []
        vae = vae.to(accelerator.device)
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                latent_dist = vae.encode(batch["pixel_values"]).latent_dist
                latents = latent_dist.mode()
                latents = (latents - vae_shift_factor) * vae_scaling_factor
                latents_cache.append(latents)
        if args.validation_prompt is None:
            del vae
            free_memory()

    # ---- LR Scheduler ----
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ---- Privacy Engine ----
    privacy_engine = None
    if args.eps is not None:
        sample_size = len(train_dataset)
        delta = 1.0 / (sample_size * np.log(sample_size))
        privacy_engine = PrivacyEngine_Distributed_extending(
            transformer,
            batch_size=total_batch_size,
            grad_accum_steps=args.gradient_accumulation_steps,
            per_device_physical_batch_size=args.train_batch_size,
            sample_size=sample_size,
            num_steps=args.total_steps,
            target_epsilon=args.eps,
            target_delta=delta,
            clipping_fn="automatic",
            clipping_mode="MixOpt",
            origin_params=None,
            clipping_style="all-layer",
            num_GPUs=accelerator.num_processes,
            torch_seed_is_fixed=True,
            micro_batch_size=args.micro_batch_size,
        )
        if accelerator.num_processes == 1:
            privacy_engine.attach(optimizer)

        account_history = None
        if args.fisher_batch_size is not None and args.fisher_sigma is not None:
            account_history = [
                (args.fisher_sigma, min(args.fisher_batch_size / sample_size, 1.0), args.fisher_num)
            ]
        privacy_engine.noise_multiplier = get_noise_multiplier(
            target_epsilon=args.eps,
            target_delta=delta,
            sample_rate=total_batch_size / sample_size,
            steps=args.total_steps,
            accountant="prv",
            account_history=account_history,
        )
        if accelerator.is_main_process:
            logger.info(
                f"DP noise multiplier: {privacy_engine.noise_multiplier:.4f}  "
                f"(effective: {privacy_engine.effective_noise_multiplier:.4f})"
            )

    # ---- Accelerator prepare ----
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-zimage-lora-dp", config=vars(args))

    # ---- Training ----
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  micro_batch_size = {args.micro_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  DP enabled = {args.eps is not None}" + (f" (eps={args.eps})" if args.eps else ""))

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_trainable_params():
        return [p for p in transformer.parameters() if p.requires_grad]

    if accelerator.is_main_process:
        if args.validation_prompt is not None:
            validation_pipeline = ZImagePipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=unwrap_model(transformer),
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            pipeline_args = {
                "prompt": args.validation_prompt,
                "height": args.resolution,
                "width": args.resolution,
            }
            log_validation(pipeline=validation_pipeline, args=args, accelerator=accelerator,
                            pipeline_args=pipeline_args, epoch=-1)
            free_memory()
            del validation_pipeline

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                prompts = batch["prompts"]

                if train_dataset.custom_instance_prompts:
                    prompt_embeds_step = [prompt_embeds_dict[p][0] for p in prompts]
                else:
                    # Replicate the single prompt embedding for each sample in the batch
                    bsz_physical = len(prompts)
                    prompt_embeds_step = [base_prompt_embeds[0]] * bsz_physical
                    if args.with_prior_preservation:
                        prompt_embeds_step = prompt_embeds_step + [class_base[0]] * bsz_physical

                # Encode images to latents
                if args.cache_latents:
                    model_input = latents_cache[step]
                else:
                    vae_dev = vae.to(accelerator.device)
                    pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=vae_dev.dtype)
                    latent_dist = vae_dev.encode(pixel_values).latent_dist
                    model_input = latent_dist.mode()
                    model_input = (model_input - vae_shift_factor) * vae_scaling_factor
                    if args.offload:
                        vae.to("cpu")

                model_input = model_input.to(dtype=weight_dtype)

                # Amplify samples for per-sample gradient estimation
                if args.micro_batch_size > 1:
                    model_input = model_input.repeat_interleave(args.micro_batch_size, dim=0)
                    # Interleave prompt embeds to match repeat_interleave ordering
                    prompt_embeds_mb = [pe for pe in prompt_embeds_step for _ in range(args.micro_batch_size)]
                else:
                    prompt_embeds_mb = prompt_embeds_step

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                timestep_normalized = (1000 - timesteps) / 1000

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Z-Image 5D forward pass
                noisy_model_input_5d = noisy_model_input.unsqueeze(2)              # (B, C, H, W) -> (B, C, 1, H, W)
                noisy_model_input_list = list(noisy_model_input_5d.unbind(dim=0))  # List of (C, 1, H, W)

                model_pred_list = transformer(
                    noisy_model_input_list,
                    timestep_normalized,
                    prompt_embeds_mb,
                    return_dict=False,
                )[0]
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)  # (B, C, H, W)
                model_pred = -model_pred  # Z-Image negates the prediction

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                target = noise - model_input

                if args.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    ).mean()

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                        target.shape[0], -1
                    ),
                    1,
                ).mean()

                if args.with_prior_preservation:
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if privacy_engine is not None and accelerator.is_main_process:
                        cos_sims = []
                        for param in get_trainable_params():
                            if param.grad is not None:
                                dp_noise = torch.randn_like(param.grad) * privacy_engine.noise_multiplier
                                param.grad.add_(dp_noise)
                                cos_sim = torch.nn.functional.cosine_similarity(
                                    param.grad.flatten(),
                                    (param.grad - dp_noise).flatten(),
                                    dim=0,
                                )
                                cos_sims.append(cos_sim.item())
                        if cos_sims:
                            logger.info(f"step {global_step} | grad-noise cosine sim: {sum(cos_sims)/len(cos_sims):.4f}")
                    else:
                        accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for ckpt in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, ckpt))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                validation_pipeline = ZImagePipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline_args = {
                    "prompt": args.validation_prompt,
                    "height": args.resolution,
                    "width": args.resolution,
                }
                log_validation(pipeline=validation_pipeline, args=args, accelerator=accelerator,
                               pipeline_args=pipeline_args, epoch=epoch)
                free_memory()
                del validation_pipeline

    # ---- Save final LoRA weights ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save = {"transformer": transformer}

        ZImagePipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        # Final inference
        pipeline = ZImagePipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.load_lora_weights(args.output_dir)

        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {
                "prompt": args.validation_prompt,
                "height": args.resolution,
                "width": args.resolution,
            }
            images = log_validation(
                pipeline=pipeline, args=args, accelerator=accelerator,
                pipeline_args=pipeline_args, epoch=epoch, is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
