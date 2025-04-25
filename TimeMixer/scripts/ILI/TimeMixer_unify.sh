

# Illness dataset - TimeMixer benchmark for 4 prediction lengths

# GPU配置（可选）
# export CUDA_VISIBLE_DEVICES=1

# 通用超参数
model_name=TimeMixer
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=32
train_epochs=20
patience=10

# 循环跑不同的pred_len
for pred_len in 24 
do
  echo "Running prediction length: $pred_len"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili_60_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 60 \
    --label_len 0 \
    --pred_len $pred_len \
    --e_layers 5 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --n_heads 4 \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window 
done