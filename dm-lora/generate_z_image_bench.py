"""Multi-GPU image generation with ZImagePipeline + DP-LoRA weights.

Pre-computes all prompt embeddings on CPU (text encoder only, single pass)
before spawning per-GPU workers, so each worker skips the Qwen3 text encoder
entirely and only holds the VAE + transformer in GPU memory.
"""

import argparse
import os
import shutil
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision
from diffusers import ZImagePipeline
from PIL import Image
from torch.multiprocessing import spawn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dataset_bench import get_prompt

warnings.filterwarnings("ignore", message="The following part of your input was truncated")


# ---------------------------------------------------------------------------
# Step 1: Pre-compute all prompt embeddings (text encoder only, CPU)
# ---------------------------------------------------------------------------

def precompute_embeddings(
    model_id: str,
    unique_prompts: List[str],
    max_sequence_length: int = 512,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Return (embed_map, neg_embed) where embed_map is {prompt: tensor (seq_len, hidden)}
    and neg_embed is the unconditional embedding for "" (seq_len, hidden).

    Loads only the tokenizer + text encoder; VAE, transformer, and scheduler
    are skipped by passing None, mirroring the training script's
    text_encoding_pipeline pattern.
    """
    print("Pre-computing prompt embeddings (text-encoder only, CPU)…")
    text_pipe = ZImagePipeline.from_pretrained(
        model_id,
        vae=None,
        transformer=None,
        scheduler=None,
        torch_dtype=torch.bfloat16,
    )
    text_pipe.text_encoder = text_pipe.text_encoder.to("cpu")

    def _encode(prompt: str) -> torch.Tensor:
        embeds, _ = text_pipe.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
        )
        emb = embeds[0] if isinstance(embeds, (list, tuple)) else embeds[0:1].squeeze(0)
        return emb.cpu()

    embed_map: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for prompt in tqdm(unique_prompts, desc="Encoding prompts"):
            embed_map[prompt] = _encode(prompt)
        neg_embed = _encode("")

    del text_pipe
    torch.cuda.empty_cache()
    print(f"Done. {len(embed_map)} unique prompt embedding(s) cached.")
    return embed_map, neg_embed


# ---------------------------------------------------------------------------
# Step 2: Per-GPU generation worker (no text encoder loaded)
# ---------------------------------------------------------------------------

def generate_batch(rank: int, world_size: int, args: argparse.Namespace) -> None:
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dtype = torch.bfloat16

    image_dir = os.path.join(args.output_dir, "gen")

    all_prompts = args.prompts * args.repeat
    local_prompts = all_prompts[rank::world_size]
    local_indices = list(range(rank, len(all_prompts), world_size))
    print(f"[GPU {rank}] Handling {len(local_prompts)} prompts")

    # Load VAE + transformer only; text encoder skipped (embeddings pre-computed).
    pipe = ZImagePipeline.from_pretrained(
        args.model_id,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    # pipe.enable_vae_slicing()          # lower peak VRAM during decode
    pipe.enable_attention_slicing()    # lower peak VRAM during transformer attention

    if args.checkpoint_path is not None:
        print(f"[GPU {rank}] Loading LoRA from {args.checkpoint_path}")
        pipe.load_lora_weights(args.checkpoint_path)

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
    gen_h, gen_w = args.gen_size

    with torch.no_grad():
        for i in tqdm(range(0, len(local_prompts), args.batch_size), disable=(rank != 0)):
            batch_info    = local_prompts[i : i + args.batch_size]
            batch_indices = local_indices[i : i + args.batch_size]
            batch_labels  = [item[1] for item in batch_info]
            batch_texts   = [item[0] for item in batch_info]

            # Build per-sample embedding lists (pipeline uses list + for CFG concatenation)
            prompt_embeds = [args.embed_map[t].to(device=device, dtype=dtype) for t in batch_texts]
            neg = args.neg_embed.to(device=device, dtype=dtype)
            negative_prompt_embeds = [neg] * len(batch_texts)

            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                height=gen_h,
                width=gen_w,
                num_inference_steps=25,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_type="pil",
            ).images

            for idx, label, img in zip(batch_indices, batch_labels, images):
                img_resized = img.resize(args.target_size, Image.LANCZOS)
                img_resized.save(os.path.join(image_dir, f"{label:06d}", f"{idx:06d}.png"))

    print(f"[GPU {rank}] Completed generation.")


# ---------------------------------------------------------------------------
# Orchestrator: pre-compute → spawn → aggregate
# ---------------------------------------------------------------------------

def generator_from_prompt_list_multigpu(
    model_id: str,
    output_dir: str,
    prompts: List[Tuple[str, int]],
    checkpoint_path: str = None,
    repeat: int = 1,
    target_size: Union[int, Tuple[int, int]] = 32,
    gen_size: Union[int, Tuple[int, int]] = 256,
    seed: int = 0,
    guidance_scale: float = 7.0,
    batch_size: int = 4,
    num_classes: int = 10,
    max_sequence_length: int = 512,
) -> None:
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No CUDA GPUs found.")
    print(f"Using {world_size} GPU(s) for generation.")

    # Pre-compute all embeddings once in the parent process.
    unique_texts = list(dict.fromkeys(p[0] for p in prompts))
    embed_map, neg_embed = precompute_embeddings(model_id, unique_texts, max_sequence_length)

    args = argparse.Namespace(
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        prompts=prompts,
        embed_map=embed_map,
        neg_embed=neg_embed,
        target_size=target_size if isinstance(target_size, tuple) else (target_size, target_size),
        gen_size=gen_size if isinstance(gen_size, tuple) else (gen_size, gen_size),
        batch_size=batch_size,
        seed=seed,
        guidance_scale=guidance_scale,
        repeat=repeat,
        num_classes=num_classes,
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "gen"), exist_ok=True)
    for cls in range(num_classes):
        os.makedirs(os.path.join(output_dir, "gen", f"{cls:06d}"), exist_ok=True)

    if world_size == 1:
        generate_batch(0, 1, args)
    else:
        spawn(generate_batch, args=(world_size, args), nprocs=world_size, join=True)

    # Aggregate per-class image dirs → .npy arrays
    dataset = ImageFolder(os.path.join(output_dir, "gen"), transform=transforms.ToTensor())
    loader  = DataLoader(dataset, batch_size=len(dataset))
    for x, y in loader:
        x, y = x.numpy(), y.numpy()
        np.save(os.path.join(output_dir, "gen", "syn_images.npy"), x)
        np.save(os.path.join(output_dir, "gen", "syn_labels.npy"), y)
        break

    # Sample grid (up to 8 per class)
    show = np.concatenate([x[y == cls][:8] for cls in range(num_classes)])
    torchvision.utils.save_image(
        torch.from_numpy(show),
        os.path.join(output_dir, "gen", "sample.png"),
        padding=1,
        nrow=8,
    )

    # Clean up per-class image dirs
    for cls in range(num_classes):
        shutil.rmtree(os.path.join(output_dir, "gen", f"{cls:06d}"))

    print(f"Generation completed. Results saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate samples with ZImagePipeline + DP-LoRA weights")
    parser.add_argument("--data_name",           type=str,   default="cifar10_32")
    parser.add_argument("--batch_size",          type=int,   default=8)
    parser.add_argument("--target_size",         type=int,   default=32)
    parser.add_argument("--gen_size",            type=int,   default=256)
    parser.add_argument("--repeat",              type=int,   default=1)
    parser.add_argument("--num",                 type=int,   default=1,
                        help="Total samples to generate (rounded down to multiple of num_classes).")
    parser.add_argument("--output_dir",          type=str,   default="")
    parser.add_argument("--checkpoint_path",     type=str,   default=None,
                        help="Directory containing pytorch_lora_weights.safetensors. "
                             "Defaults to --output_dir if not set.")
    parser.add_argument("--model_id",            type=str,   default=None)
    parser.add_argument("--guidance_scale",      type=float, default=7.0)
    parser.add_argument("--max_sequence_length", type=int,   default=512)
    args = parser.parse_args()

    prompt_list = get_prompt(args.data_name)
    num_classes = len(prompt_list)
    prompts = [[prompt, idx] for idx, prompt in enumerate(prompt_list)]
    prompts = prompts * (args.num // len(prompt_list))

    generator_from_prompt_list_multigpu(
        model_id=args.model_id,
        checkpoint_path=args.checkpoint_path or args.output_dir,
        output_dir=args.output_dir,
        prompts=prompts,
        batch_size=args.batch_size,
        target_size=(args.target_size, args.target_size),
        gen_size=(args.gen_size, args.gen_size),
        repeat=args.repeat,
        seed=None,
        guidance_scale=args.guidance_scale,
        num_classes=num_classes,
        max_sequence_length=args.max_sequence_length,
    )
