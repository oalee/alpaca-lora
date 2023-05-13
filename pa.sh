WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 --master_port=3192 finetune.py \
    --base_model='decapoda-research/llama-13b-hf' \
    --data_path './alpaca_fa.json' \
    --num_epochs=2 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=8 \
    --micro_batch_size=16
