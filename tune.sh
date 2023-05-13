python finetune.py \
    --base_model='HiTZ/alpaca-lora-13b-en-pt-es-ca-eu-gl-at' \
    --data_path './alpaca_fa.json' \
    --num_epochs=2 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=8 \
    --micro_batch_size=16
