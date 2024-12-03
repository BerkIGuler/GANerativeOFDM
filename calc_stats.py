import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import MatDataset
from pathlib import Path
import json
import argparse
from tqdm import tqdm

def create_parser():
    """Create parser for command line arguments."""
    parser = argparse.ArgumentParser(description='OFDM Channel Data Analysis Tool')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--pilot_dims', type=int, nargs=2, default=[2, 4],
                        help='Dimensions of pilot pattern (height width)')

    # Analysis parameters
    parser.add_argument('--num_batches', type=int, default=None,
                        help='Number of batches to analyze (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loading')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='channel_analysis_results',
                        help='Directory to save analysis results')

    # Device selection
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computations (cuda/cpu)')

    return parser


def calculate_correlation(data):
    """Calculate correlation matrix for complex data."""
    # Ensure data is 2D
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    # Convert to numpy for complex correlation calculation
    data_np = data.cpu().numpy()
    # Calculate correlation matrix
    n = data_np.shape[1]
    corr = np.zeros((n, n), dtype=np.complex64)

    for i in range(n):
        for j in range(n):
            # Calculate complex correlation coefficient
            x = data_np[:, i]
            y = data_np[:, j]
            corr[i, j] = np.mean(np.conjugate(x) * y)
    return np.abs(corr)  # Return magnitude of correlation


class OFDMChannelAnalyzer:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze_channel_statistics(self, num_batches=None):
        """Calculate various channel statistics across the dataset."""
        # Initialize statistics containers
        all_magnitudes = []
        all_phases = []
        frequency_coherence = []
        time_coherence = []

        print("Analyzing channel statistics...")

        for batch_idx, (real_input, real_target, _) in enumerate(tqdm(self.dataloader)):
            if num_batches and batch_idx >= num_batches:
                break

            # Move to device
            channel = real_target.to(self.device)

            # Calculate magnitude and phase
            magnitude = torch.abs(channel)
            phase = torch.angle(channel)

            all_magnitudes.append(magnitude.cpu().numpy())
            all_phases.append(phase.cpu().numpy())

            # Calculate frequency correlation (across subcarriers)
            for t in range(channel.shape[2]):  # For each OFDM symbol
                freq_corr = calculate_correlation(channel[..., t])
                frequency_coherence.append(freq_corr)

            # Calculate time correlation (across OFDM symbols)
            for f in range(channel.shape[1]):  # For each subcarrier
                time_corr = calculate_correlation(channel[:, f, :])
                time_coherence.append(time_corr)

        # Combine statistics
        all_magnitudes = np.concatenate(all_magnitudes, axis=0)
        all_phases = np.concatenate(all_phases, axis=0)

        stats = {
            'magnitude_mean': float(np.mean(all_magnitudes)),
            'magnitude_std': float(np.std(all_magnitudes)),
            'magnitude_max': float(np.max(all_magnitudes)),
            'magnitude_min': float(np.min(all_magnitudes)),
            'phase_mean': float(np.mean(all_phases)),
            'phase_std': float(np.std(all_phases)),
            'frequency_coherence_mean': float(np.mean([np.mean(np.abs(fc)) for fc in frequency_coherence])),
            'time_coherence_mean': float(np.mean([np.mean(np.abs(tc)) for tc in time_coherence]))
        }

        return stats, all_magnitudes, all_phases, frequency_coherence, time_coherence

    @staticmethod
    def plot_channel_characteristics(magnitudes, phases, output_dir='channel_analysis'):
        """Generate plots for channel characteristics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 1. Magnitude distribution with proper normalization and fewer bins
        plt.figure(figsize=(10, 6))
        mag_data = magnitudes.flatten()

        # Use fewer bins (50 bins instead of Freedman-Diaconis rule)
        n_bins = 50

        # Plot normalized histogram
        counts, bins, _ = plt.hist(mag_data, bins=n_bins, density=True)
        plt.clf()  # Clear the figure

        # Replot with proper normalization
        plt.hist(mag_data, bins=bins, weights=np.ones_like(mag_data) / len(mag_data),
                 alpha=0.7, color='blue', edgecolor='black')
        plt.title('Channel Magnitude Distribution (Normalized)')
        plt.xlabel('Magnitude')
        plt.ylabel('Probability')


        plt.savefig(output_dir / 'magnitude_distribution.png')
        plt.close()

        # 2. Phase distribution with proper normalization and fewer bins
        plt.figure(figsize=(10, 6))
        phase_data = phases.flatten()

        # Use fewer bins (50 bins instead of Freedman-Diaconis rule)
        n_bins = 50

        # Plot normalized histogram
        counts, bins, _ = plt.hist(phase_data, bins=n_bins, density=True)
        plt.clf()  # Clear the figure

        # Replot with proper normalization
        plt.hist(phase_data, bins=bins, weights=np.ones_like(phase_data) / len(phase_data),
                 alpha=0.7, color='blue', edgecolor='black')
        plt.title('Channel Phase Distribution (Normalized)')
        plt.xlabel('Phase (radians)')
        plt.ylabel('Probability')

        plt.savefig(output_dir / 'phase_distribution.png')
        plt.close()


def main():
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments for reproducibility
    with open(output_dir / 'analysis_args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Initialize dataset and dataloader
    dataset = MatDataset(
        data_dir=args.data_dir,
        pilot_dims=tuple(args.pilot_dims),
        return_type="complex_zero"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Initialize analyzer
    analyzer = OFDMChannelAnalyzer(dataloader)

    # Perform analysis
    stats, magnitudes, phases, freq_coherence, time_coherence = analyzer.analyze_channel_statistics(
        num_batches=args.num_batches
    )

    # Generate plots and save results
    analyzer.plot_channel_characteristics(magnitudes, phases, output_dir=args.output_dir)

    # Print summary statistics
    print("\nChannel Statistics Summary:")
    print(json.dumps(stats, indent=4))


if __name__ == '__main__':
    main()