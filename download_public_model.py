from diffusers import StableDiffusionImg2ImgPipeline
import torch

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
model_id = "Manojb/stable-diffusion-2-1-base"
model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
model_id = "Nihirc/Prompt2MedImage"
model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)