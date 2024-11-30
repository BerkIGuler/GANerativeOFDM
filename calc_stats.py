import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import MatDataset  # Using your existing dataloader
import seaborn as sns
from pathlib import Path
import json


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

        for batch_idx, (real_input, real_target, _) in enumerate(self.dataloader):
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
                freq_corr = torch.corrcoef(channel[..., t].T)
                frequency_coherence.append(freq_corr.cpu().numpy())

            # Calculate time correlation (across OFDM symbols)
            for f in range(channel.shape[1]):  # For each subcarrier
                time_corr = torch.corrcoef(channel[:, f, :].T)
                time_coherence.append(time_corr.cpu().numpy())

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
            'frequency_coherence_mean': float(np.mean([np.mean(fc) for fc in frequency_coherence])),
            'time_coherence_mean': float(np.mean([np.mean(tc) for tc in time_coherence]))
        }

        return stats, all_magnitudes, all_phases, frequency_coherence, time_coherence

    def plot_channel_characteristics(self, stats, magnitudes, phases, freq_coherence, time_coherence,
                                     output_dir='channel_analysis'):
        """Generate plots for channel characteristics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 1. Magnitude distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(magnitudes.flatten(), bins=50)
        plt.title('Channel Magnitude Distribution')
        plt.xlabel('Magnitude')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'magnitude_distribution.png')
        plt.close()

        # 2. Phase distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(phases.flatten(), bins=50)
        plt.title('Channel Phase Distribution')
        plt.xlabel('Phase (radians)')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'phase_distribution.png')
        plt.close()

        # 3. Average frequency correlation
        plt.figure(figsize=(10, 8))
        avg_freq_corr = np.mean(freq_coherence, axis=0)
        sns.heatmap(avg_freq_corr, cmap='coolwarm')
        plt.title('Average Frequency Correlation')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Subcarrier Index')
        plt.savefig(output_dir / 'frequency_correlation.png')
        plt.close()

        # 4. Average time correlation
        plt.figure(figsize=(10, 8))
        avg_time_corr = np.mean(time_coherence, axis=0)
        sns.heatmap(avg_time_corr, cmap='coolwarm')
        plt.title('Average Time Correlation')
        plt.xlabel('OFDM Symbol Index')
        plt.ylabel('OFDM Symbol Index')
        plt.savefig(output_dir / 'time_correlation.png')
        plt.close()

        # Save statistics to JSON
        with open(output_dir / 'channel_statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)


def main():
    # Initialize dataset and dataloader (example parameters)
    dataset = MatDataset(data_dir='path/to/your/data', pilot_dims=(2, 4))  # Adjust parameters
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Initialize analyzer
    analyzer = OFDMChannelAnalyzer(dataloader)

    # Perform analysis
    stats, magnitudes, phases, freq_coherence, time_coherence = analyzer.analyze_channel_statistics(
        num_batches=100  # Adjust based on your needs
    )

    # Generate plots and save results
    analyzer.plot_channel_characteristics(
        stats, magnitudes, phases, freq_coherence, time_coherence,
        output_dir='channel_analysis_results'
    )

    # Print summary statistics
    print("\nChannel Statistics Summary:")
    print(json.dumps(stats, indent=4))


if __name__ == '__main__':
    main()