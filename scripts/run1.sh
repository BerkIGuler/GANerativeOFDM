python train.py \
    --train_path dataset/train \
    --val_path dataset/val \
    --output_dir experiments/run1 \
    --batch_size 512 \
    --lr 0.0001 \
    --lambda_l1 10 \
    --dropout 0.4 \
    --num_epochs 40 \
    --save_freq 10 \
    --device cuda:1 \
    --d_loss_coef 0.5

python train.py \
    --train_path dataset/train \
    --val_path dataset/val \
    --output_dir experiments/run1 \
    --batch_size 512 \
    --lr 0.0001 \
    --lambda_l1 50 \
    --dropout 0.4 \
    --num_epochs 40 \
    --save_freq 10 \
    --device cuda:1 \
    --d_loss_coef 0.5

python train.py \
    --train_path dataset/train \
    --val_path dataset/val \
    --output_dir experiments/run1 \
    --batch_size 512 \
    --lr 0.0001 \
    --lambda_l1 200 \
    --dropout 0.4 \
    --num_epochs 40 \
    --save_freq 10 \
    --device cuda:1 \
    --d_loss_coef 0.5

python train.py \
    --train_path dataset/train \
    --val_path dataset/val \
    --output_dir experiments/run1 \
    --batch_size 512 \
    --lr 0.0001 \
    --lambda_l1 500 \
    --dropout 0.4 \
    --num_epochs 40 \
    --save_freq 10 \
    --device cuda:1 \
    --d_loss_coef 0.5