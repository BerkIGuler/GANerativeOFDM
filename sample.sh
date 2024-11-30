#python sample.py \
#    --checkpoint experiments/run21/best_model/model_checkpoint.pth \
#    --data_path dataset/test_noisy/SNR_test_set \
#    --batch_size 128 \
#    --output_dir results/run21_test_noisy \
#    --device cuda:0
#
#python sample.py \
#    --checkpoint experiments/run22/best_model/model_checkpoint.pth \
#    --data_path dataset/test_noisy/SNR_test_set \
#    --batch_size 128 \
#    --output_dir results/run22_test_noisy \
#    --device cuda:0
#
#python sample.py \
#    --checkpoint experiments/run23/best_model/model_checkpoint.pth \
#    --data_path dataset/test_noisy/SNR_test_set \
#    --batch_size 128 \
#    --output_dir results/run23_test_noisy \
#    --device cuda:0

python sample.py \
    --checkpoint experiments/run11/epoch_10/model_checkpoint.pth \
    --data_path dataset/test_noisy/SNR_test_set \
    --batch_size 128 \
    --output_dir results/run11_test_noisy_ep10 \
    --device cuda:0

python sample.py \
    --checkpoint experiments/run11/epoch_10/model_checkpoint.pth \
    --data_path dataset/test/SNR_test_set \
    --batch_size 128 \
    --output_dir results/run11_test_ep10 \
    --device cuda:0