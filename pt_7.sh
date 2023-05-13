CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --data_path '/data/datasets/ptc_ds.json' \
    --num_epochs=2 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='/data/weights/ptc_7b_lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=32 \
    --val_set_size=20   

#--resume_from_checkpoint='/data/weights/ptc_13b_lora-alpaca' \
#    --val_set_size=20

