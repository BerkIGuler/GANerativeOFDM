"""
Script for loading and visualizing wireless channel response data from .mat files.
The script processes complex-valued channel response data and generates plots
of magnitude and phase responses across different channels.
"""

from dataloader import MatDataset
from pathlib import Path
from torch.utils.data import DataLoader
from utils import ChannelVisualizer
import matplotlib.pyplot as plt

# Directory configuration for dataset
DATA_DIR = Path("./dataset").resolve()
TRAIN_DIR = DATA_DIR / 'train'

# Channel response matrix dimensions (18 subcarriers × 2 antennas)
PILOT_DIMS = (18, 2)

# Data transformation settings
TRANSFORM = None
RETURN_TYPE = "complex"  # Return complex-valued channel responses
BATCH_SIZE = 8  # Number of samples per batch


def main():
    """
    Main function to load channel response data and generate visualization plots.

    The function performs the following steps:
    1. Creates a dataset object for loading .mat files
    2. Initializes a DataLoader for batch processing
    3. Extracts one batch of channel responses
    4. Generates and saves magnitude and phase response plots
    """
    # Initialize dataset with specified parameters
    mat_dataset = MatDataset(
        data_dir=TRAIN_DIR,
        pilot_dims=PILOT_DIMS,
        transform=None,
        return_type=RETURN_TYPE)

    # Create DataLoader for batch processing
    dataloader = DataLoader(mat_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get first batch of data
    # h_estimated: Estimated channel responses
    # h_ideal: Ground truth channel responses
    # _: Ignored additional data (if any)
    first_batch = next(dataloader.__iter__())
    h_estimated, h_ideal, _ = first_batch

    # Print tensor dimensions for verification
    print(h_estimated.size())  # Expected: [batch_size, num_subcarriers, num_antennas]
    print(h_ideal.size())  # Expected: [batch_size, num_subcarriers, num_antennas]

    # Create visualizer object using ground truth channel responses
    visualizer = ChannelVisualizer(h_ideal)

    # Generate plots for magnitude and phase responses
    mag_fig = visualizer.plot_magnitudes()  # Plot magnitude responses
    phase_fig = visualizer.plot_phases()  # Plot phase responses

    # Save generated figures with high resolution
    mag_fig.savefig('channel_magnitude_responses.png', bbox_inches='tight', dpi=300)
    phase_fig.savefig('channel_phase_responses.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()