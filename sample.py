import argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from generator import UNet
from dataloader import MatDataset


def create_parser():
    parser = argparse.ArgumentParser(description='Sample from trained Conditional GAN')

    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--pilot_dims', nargs=2, type=int, default=[18, 2],
                        help='Dimensions of pilot signal (height width)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for sampling')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='samples',
                        help='Directory to save samples and results')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for sampling')

    return parser


def calculate_mse_db(generator, dataloader, device):
    """Calculate 10*log10(MSE) for the entire dataset"""
    generator.eval()
    mse_list = []

    with torch.no_grad():
        for inputs, targets, _ in tqdm(dataloader, desc='Evaluating MSE'):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Generate samples
            outputs = generator(inputs)

            # Calculate MSE
            mse = torch.nn.functional.mse_loss(outputs, targets, reduction='none')
            mse = mse.mean(dim=(1, 2, 3))  # Average over channels and dimensions

            # Convert to dB scale
            mse_db = 10 * torch.log10(mse)

            # Store results
            mse_list.extend(mse_db.cpu().numpy())

    return np.array(mse_list)


def plot_mse_histogram(mse_values, output_path):
    """Plot histogram of MSE values"""
    plt.figure(figsize=(10, 6))
    plt.hist(mse_values, bins=50, edgecolor='black')
    plt.xlabel('MSE (dB)')
    plt.ylabel('Count')
    plt.title('Distribution of MSE Values')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def main():
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)

    # Load model
    generator = UNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    # Create dataloader
    dataset = MatDataset(data_dir=args.data_path, pilot_dims=tuple(args.pilot_dims))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)  # No need to shuffle for evaluation

    # Calculate MSE in dB scale for all samples
    mse_db_values = calculate_mse_db(generator, dataloader, device)

    # Calculate statistics
    stats = {
        'num_samples': len(mse_db_values),
        'mean_mse_db': float(np.mean(mse_db_values)),
        'median_mse_db': float(np.median(mse_db_values)),
        'std_mse_db': float(np.std(mse_db_values)),
        'min_mse_db': float(np.min(mse_db_values)),
        'max_mse_db': float(np.max(mse_db_values)),
        'percentile_5': float(np.percentile(mse_db_values, 5)),
        'percentile_95': float(np.percentile(mse_db_values, 95))
    }

    # Save results
    results_file = output_dir / 'sampling_results.json'
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=4)

    # Plot histogram
    plot_path = output_dir / 'mse_distribution.png'
    plot_mse_histogram(mse_db_values, plot_path)

    # Print results
    print("\nSampling Results:")
    print(f"Number of test samples: {stats['num_samples']}")
    print("\nMSE Statistics (dB):")
    for key, value in stats.items():
        if key != 'num_samples':
            print(f"{key}: {value:.2f}")
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()