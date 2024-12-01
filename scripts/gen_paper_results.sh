python generate_paper_results.py \
    --checkpoint experiments/run11/best_model/model_checkpoint.pth \
    --test_path dataset/test/SNR_test_set \
    --test_noisy_path dataset/test_noisy/SNR_test_set \
    --sample_path dataset/sample \
    --output_dir paper_results