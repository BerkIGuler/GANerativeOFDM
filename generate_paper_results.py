import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy.io as sio
from collections import defaultdict
from dataloader import get_test_dataloaders

from generator import UNet
import matplotlib.pyplot as plt


def create_parser():
    parser = argparse.ArgumentParser(description='Generate paper results')

    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best model checkpoint')

    # Data parameters
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to SNR_test_set in test folder')
    parser.add_argument('--test_noisy_path', type=str, required=True,
                        help='Path to SNR_test_set in test_noisy folder')
    parser.add_argument('--sample_path', type=str, required=True,
                        help='Path to sample folder for qualitative results')
    parser.add_argument('--pilot_dims', nargs=2, type=int, default=[18, 2],
                        help='Dimensions of pilot signal (height width)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for sampling')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='paper_results',
                        help='Directory to save results')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for sampling')

    return parser


def calculate_avg_nmse_db(generator, dataloader, device):
    """Calculate average 10*log10(NMSE) for the dataset"""
    generator.eval()
    nmse_values = []

    with torch.no_grad():
        for inputs, targets, _ in tqdm(dataloader, desc='Calculating NMSE'):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Generate samples
            outputs = generator(inputs)

            # Calculate squared error
            squared_error = torch.nn.functional.mse_loss(outputs, targets, reduction='none')

            # Calculate power of the target signal
            target_power = torch.mean(targets ** 2, dim=(1, 2, 3))

            # Calculate NMSE
            epsilon = 1e-10
            nmse = torch.mean(squared_error, dim=(1, 2, 3)) / (target_power + epsilon)

            # Convert to dB scale
            nmse_db = 10 * torch.log10(nmse)

            nmse_values.extend(nmse_db.cpu().numpy())

    return float(np.mean(nmse_values))


def process_quantitative_results(generator, dataloaders, device, output_dir, is_noisy):
    """Process quantitative results and save to CSV"""
    results = defaultdict(float)

    for dataset_name, dataloader in dataloaders:
        # Extract SNR value from dataset name (e.g., 'SNR_0' -> 0)
        snr = int(dataset_name.split('_')[1])

        # Calculate average NMSE for this SNR
        avg_nmse = calculate_avg_nmse_db(generator, dataloader, device)
        results[snr] = avg_nmse

    # Create DataFrame and sort by SNR
    df = pd.DataFrame(list(results.items()), columns=['snr', 'value'])
    df = df.sort_values('snr')

    # Save to CSV
    dataset_type = 'test_noisy' if is_noisy else 'test'
    csv_path = output_dir / f'cGAN_{dataset_type}_snr.csv'
    df.to_csv(csv_path, index=False)

    return csv_path


def generate_qualitative_samples(generator, sample_path, output_dir, device):
    """Generate and save samples for qualitative analysis"""
    generator.eval()
    sample_dir = Path(sample_path)

    # Create output directory for samples
    samples_output_dir = output_dir / 'samples'
    samples_output_dir.mkdir(parents=True, exist_ok=True)

    # Lists to store outputs for grid visualization
    all_outputs = []
    file_names = []

    # Process each .mat file
    for mat_file in tqdm(list(sample_dir.glob('*.mat')), desc='Generating samples'):
        # Load .mat file
        data = sio.loadmat(str(mat_file))

        # Prepare input tensor
        complex_input = torch.tensor(data["H"][:, :, 1], dtype=torch.cfloat)
        two_ch_input = torch.cat([
            torch.real(complex_input).unsqueeze(0),
            torch.imag(complex_input).unsqueeze(0)
        ], dim=0).to(device).unsqueeze(0)

        # Generate sample
        with torch.no_grad():
            output = generator(two_ch_input)

        # Convert to numpy and ensure correct shape (2, 120, 14)
        output_np = output.squeeze(0).cpu().numpy()

        # Save individual .npy file
        output_path = samples_output_dir / f"{mat_file.stem}.npy"
        np.save(output_path, output_np)

        # Store for grid visualization
        all_outputs.append(output_np)
        file_names.append(mat_file.stem)

    # Create a grid visualization of magnitudes
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    for idx, (output, fname) in enumerate(zip(all_outputs, file_names)):
        # Calculate magnitude from real and imaginary parts
        magnitude = np.sqrt(output[0] ** 2 + output[1] ** 2)

        # Plot magnitude
        row = idx // 2
        col = idx % 2
        im = axes[row, col].imshow(magnitude, cmap='viridis', aspect='auto')
        axes[row, col].set_title(f'{fname}')
        plt.colorbar(im, ax=axes[row, col])

    # Adjust layout and save
    plt.tight_layout()
    grid_path = samples_output_dir / 'samples_grid.png'
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Individual samples and grid visualization saved in {samples_output_dir}")

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

    # Process quantitative results for test set
    print("\nProcessing test set...")
    test_dataloaders = get_test_dataloaders(args.test_path, vars(args))
    test_csv = process_quantitative_results(generator, test_dataloaders, device, output_dir, is_noisy=False)
    print(f"Test results saved to {test_csv}")

    # Process quantitative results for test_noisy set
    print("\nProcessing test_noisy set...")
    test_noisy_dataloaders = get_test_dataloaders(args.test_noisy_path, vars(args))
    test_noisy_csv = process_quantitative_results(generator, test_noisy_dataloaders, device, output_dir, is_noisy=True)
    print(f"Test_noisy results saved to {test_noisy_csv}")

    # Generate qualitative samples
    print("\nGenerating qualitative samples...")
    generate_qualitative_samples(generator, args.sample_path, output_dir, device)
    print(f"Samples saved in {output_dir}/samples")


if __name__ == '__main__':
    main()