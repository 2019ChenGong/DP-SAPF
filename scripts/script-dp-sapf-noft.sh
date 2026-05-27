conda activate dplora
cd gap/dm-lora/

subjects="cifar10_32" # Subject Name
data_path="../dataset/cifar10/train_32.zip"
model_resolution=256
sensitive_resolution=32
lower_name="base_noft"
MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5" # Model card


python generate_sd_bench_ori.py --batch_size 15 --data_name $subjects --output_dir "../exp/"$lower_name"_${subjects}_noft" --num 60000 --target_size $sensitive_resolution --model_id $MODEL_NAME --gen_size $model_resolution

conda deactivate