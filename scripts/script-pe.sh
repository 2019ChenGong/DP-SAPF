conda activate dpimagebench
cd gap

eps=10
MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5" # Model card

python run.py setup.n_gpus_per_node=3 model.api=stable_diffusion model.api_params.random_sampling_checkpoint=$MODEL_NAME  model.api_params.variation_checkpoint=$MODEL_NAME train.variation_degree_schedule=[1.0,0.9,0.8,0.7,0.6,0.5] model.api_params.random_sampling_batch_size=30 model.api_params.variation_batch_size=30 train.image_size=512x512 -m PE -dn cifar10_32 -e $eps -ed base_numsample24k_10steps train.combine_variation_extraction=true model.api_params.random_sampling_num_inference_steps=10 model.api_params.variation_num_inference_steps=10

conda deactivate