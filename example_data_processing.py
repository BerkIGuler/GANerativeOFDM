"""
Script for loading and visualizing wireless channel response data from .mat files.
The script processes complex-valued channel response data and generates plots
of magnitude and phase responses across different channels.
"""

from dataloader import MatDataset
from pathlib import Path
from torch.utils.data import DataLoader
from utils import ChannelVisualizer

# Directory configuration for dataset
DATA_DIR = Path("./dataset").resolve()
TRAIN_DIR = DATA_DIR / 'train'

# Channel response matrix dimensions (18 subcarriers × 2 antennas)
PILOT_DIMS = (18, 2)

# Data transformation settings
TRANSFORM = None
RETURN_TYPE = "complex_zero"  # Return complex-valued channel responses
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

    # Create visualizer object using ground truth channel responses
    vis_est = ChannelVisualizer(h_estimated)
    vis_ideal = ChannelVisualizer(h_ideal)

    # Generate plots for magnitude and phase responses
    est_fig = vis_est.plot_magnitudes()  # Plot magnitude responses
    ideal_fig = vis_ideal.plot_magnitudes()  # Plot phase responses

    # Save generated figures with high resolution
    est_fig.savefig('mag_est.png', bbox_inches='tight', dpi=300)
    ideal_fig.savefig('mag_ideal.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()