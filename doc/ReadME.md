# Conditional GAN Implementation Based on [Pix2Pix](https://arxiv.org/abs/1611.07004)
* The aim here is to generate robust OFDM channels conditioned on channel at pilot positions.

# Training 
### Basic usage with default parameters
`python train.py`

### Customize training parameters
`python train.py --batch_size 64 --lr 0.0001 --num_epochs 300`

### Use different paths
`python train.py --train_path /path/to/train --val_path /path/to/val --output_dir /path/to/output`

### Resume training from checkpoint
`python train.py --resume outputs/checkpoint_epoch_50.pth`

### Enable debug mode
`python train.py --debug`

### Full custom configuration
`
python train.py \
    --train_path data/train \
    --val_path data/val \
    --output_dir experiments/run1 \
    --batch_size 64 \
    --lr 0.0001 \
    --lambda_l1 50 \
    --dropout 0.5 \
    --num_epochs 300 \
    --save_freq 5 \
    --device cuda
`
# Sampling

`
python sample.py \
    --checkpoint outputs/best_model.pth \
    --data_path dataset/test \
    --batch_size 128 \
    --output_dir results/sampling_run1 \
    --device cuda
`

# Channel Statistics
`python analyze_ofdm.py \
    --data_dir /dataset/train \
    --pilot_dims 18 2 \
    --num_batches 100 \
    --batch_size 256 \
    --output_dir channel_analysis_results \
    --device cuda:0`

# Generate Paper Results
`
python generate_paper_results.py \
    --checkpoint experiments/run11/best_model/model_checkpoint.pth \
    --test_path dataset/test/SNR_test_set \
    --test_noisy_path dataset/test_noisy/SNR_test_set \
    --sample_path dataset/sample \
    --output_dir paper_results
`