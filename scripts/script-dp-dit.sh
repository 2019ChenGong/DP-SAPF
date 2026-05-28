eval "$(conda shell.bash hook)"
conda activate dplora
cd dm-lora

subjects="celeba_male_256" # Subject Name
data_path="../dataset/celeba/train_256_Male.zip"
sensitive_resolution=256
INSTANCE_PROMPT="An image of a female face"
MODEL_NAME="Tongyi-MAI/Z-Image"

eps=10.0
lr=0.001
rank=4
lora_alpha=8
steps=1000
batch_size=512
grad_accum=64
resolution=256

# Fisher settings
fisher_num_batches=50000
fisher_sigma=5.0
top_k_lora=0.3

OUTPUT_DIR="../exp/zimage_${subjects}_${batch_size}bs_${steps}steps_eps${eps}_top0.3"
FISHER_JSON="${OUTPUT_DIR}/fisher_layers.json"

# -----------------------------------------------------------------------
# Step 1: Fisher layer selection
# -----------------------------------------------------------------------
accelerate launch train_dreambooth_lora_z_image_fisher.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --bench_path=$data_path \
    --instance_data_dir=$subjects \
    --instance_prompt="$INSTANCE_PROMPT" \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=bf16 \
    --resolution=$resolution \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --fisher_num_batches=$fisher_num_batches \
    --fisher_sigma=$fisher_sigma \
    --top_k_lora=$top_k_lora \
    --target_modules="to_k,to_q,to_v" \
    --seed=0

# -----------------------------------------------------------------------
# Step 2: DP-SGD LoRA training
# -----------------------------------------------------------------------
accelerate launch train_dreambooth_lora_z_image_dp.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --bench_path=$data_path \
    --instance_data_dir=$subjects \
    --validation_prompt="$INSTANCE_PROMPT" \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="bf16" \
    --resolution=$resolution \
    --train_batch_size=$batch_size \
    --gradient_accumulation_steps=$grad_accum \
    --learning_rate=$lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=$steps \
    --total_steps=$steps \
    --rank=$rank \
    --lora_alpha=$lora_alpha \
    --eps=$eps \
    --micro_batch_size=1 \
    --fisher_batch_size=$fisher_num_batches \
    --fisher_sigma=$fisher_sigma \
    --lora_layers_json=$FISHER_JSON \
    --checkpointing_steps=200 \
    --dataloader_num_workers=0 \
    --seed=0

python generate_z_image_bench.py --batch_size 8 --data_name $subjects --output_dir $OUTPUT_DIR"/zimage_lora_r${rank}_lr${lr}_eps${eps}" --num 60000 --target_size $sensitive_resolution --model_id $MODEL_NAME --gen_size $resolution

conda deactivate