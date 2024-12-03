import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from generator import UNet
from dataloader import get_test_dataloaders


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
                        help='Device to use for sampling')

    return parser


def calculate_nmse_db(generator, dataloader, device):
    """Calculate 10*log10(NMSE) for the entire dataset"""
    generator.eval()
    nmse_list = []

    with torch.no_grad():
        for inputs, targets, _ in tqdm(dataloader, desc='Evaluating NMSE'):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Generate samples
            outputs = generator(inputs)

            # Calculate squared error
            squared_error = torch.nn.functional.mse_loss(outputs, targets, reduction='none')

            # Calculate power of the target signal
            target_power = torch.mean(targets ** 2, dim=(1, 2, 3))

            # Calculate NMSE (normalize MSE by target power)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            nmse = torch.mean(squared_error, dim=(1, 2, 3)) / (target_power + epsilon)

            # Convert to dB scale
            nmse_db = 10 * torch.log10(nmse)

            # Store results
            nmse_list.extend(nmse_db.cpu().numpy())

    return np.array(nmse_list)


def calculate_statistics(nmse_db_values):
    """Calculate statistics for NMSE values"""
    return {
        'num_samples': len(nmse_db_values),
        'mean_nmse_db': float(np.mean(nmse_db_values)),
        'median_nmse_db': float(np.median(nmse_db_values)),
        'std_nmse_db': float(np.std(nmse_db_values)),
        'min_nmse_db': float(np.min(nmse_db_values)),
        'max_nmse_db': float(np.max(nmse_db_values)),
        'percentile_5': float(np.percentile(nmse_db_values, 5)),
        'percentile_95': float(np.percentile(nmse_db_values, 95))
    }


def plot_nmse_histogram(nmse_values, output_path, dataset_name):
    """Plot histogram of NMSE values"""
    plt.figure(figsize=(10, 6))
    plt.hist(nmse_values, bins=50, edgecolor='black')
    plt.xlabel('NMSE (dB)')
    plt.ylabel('Count')
    plt.title(f'Distribution of NMSE Values - {dataset_name}')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def plot_combined_histogram(all_nmse_values, dataset_names, output_path):
    """Plot combined histogram of NMSE values from all datasets"""
    plt.figure(figsize=(12, 8))

    for nmse_values, name in zip(all_nmse_values, dataset_names):
        plt.hist(nmse_values, bins=50, alpha=0.5, label=name)

    plt.xlabel('NMSE (dB)')
    plt.ylabel('Count')
    plt.title('Distribution of NMSE Values - All Datasets')
    plt.legend()
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

    # Load model with error handling
    generator = UNet().to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Check if it's a full training checkpoint or just model weights
        if isinstance(checkpoint, dict):
            if 'generator_state_dict' in checkpoint:
                # Full training checkpoint
                generator.load_state_dict(checkpoint['generator_state_dict'])
            else:
                # Direct state dict
                generator.load_state_dict(checkpoint)
        else:
            raise ValueError("Checkpoint format not recognized")

        generator.eval()
        print(f"Successfully loaded checkpoint from {args.checkpoint}")

    except Exception as e:
        print(f"Error loading checkpoint from {args.checkpoint}: {str(e)}")
        print("Please verify that the checkpoint file exists and is not corrupted.")
        raise SystemExit(1)

    # Rest of the code remains the same...

    # Create dataloaders for each subfolder
    test_dataloaders = get_test_dataloaders(args.data_path, vars(args))

    # Dictionary to store results for all datasets
    all_results = {}
    all_nmse_values = []
    dataset_names = []

    # Process each dataset
    for dataset_name, dataloader in test_dataloaders:
        print(f"\nProcessing dataset: {dataset_name}")

        # Calculate NMSE in dB scale for all samples
        nmse_db_values = calculate_nmse_db(generator, dataloader, device)
        all_nmse_values.append(nmse_db_values)
        dataset_names.append(dataset_name)

        # Calculate statistics for this dataset
        stats = calculate_statistics(nmse_db_values)
        all_results[dataset_name] = stats

        # Plot individual histogram
        plot_path = output_dir / f'nmse_distribution_{dataset_name}.png'
        plot_nmse_histogram(nmse_db_values, plot_path, dataset_name)

        # Print results for this dataset
        print(f"\nResults for {dataset_name}:")
        print(f"Number of test samples: {stats['num_samples']}")
        print("\nNMSE Statistics (dB):")
        for key, value in stats.items():
            if key != 'num_samples':
                print(f"{key}: {value:.2f}")

    # Calculate aggregated statistics across all datasets
    all_nmse_combined = np.concatenate(all_nmse_values)
    all_results['combined'] = calculate_statistics(all_nmse_combined)

    # Plot combined histogram
    combined_plot_path = output_dir / 'nmse_distribution_combined.png'
    plot_combined_histogram(all_nmse_values, dataset_names, combined_plot_path)

    # Save all results
    results_file = output_dir / 'sampling_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll results saved to {output_dir}")
    print("\nCombined Statistics across all datasets:")
    for key, value in all_results['combined'].items():
        if key != 'num_samples':
            print(f"{key}: {value:.2f}")


if __name__ == '__main__':
    main()