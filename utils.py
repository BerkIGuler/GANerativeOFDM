"""
Utility functions for OFDM channel estimation.

This module provides various utility functions for processing, visualizing,
and analyzing OFDM channel estimation data, including complex channel matrices,
error calculations, and model statistics.
"""

from dataclasses import dataclass
from typing import Dict, List, Union
import re
import numpy as np
import matplotlib.pyplot as plt
import torch

# Type aliases
ComplexTensor = torch.Tensor
ChannelData = Dict[Union[str, int], Dict[str, ComplexTensor]]
StatsType = Dict[int, float]


class ChannelVisualizer:
    def __init__(self, channels):
        """
        Initialize the Channel Visualizer.

        Args:
            channels (torch.Tensor): Complex tensor of shape (batch_size, n_subcarriers, n_symbols)
        """
        if not isinstance(channels, torch.Tensor):
            raise TypeError("Channels must be a PyTorch tensor")

        if channels.dim() != 3:
            raise ValueError("Channels must be a 3D tensor")

        self.channels = channels
        self.batch_size = channels.shape[0]
        self.n_subcarriers = channels.shape[1]
        self.n_symbols = channels.shape[2]

    def plot_magnitudes(self, figsize=(20, 8), cmap='viridis'):
        """
        Plot magnitude of channels in a 2x4 grid.

        Args:
            figsize (tuple): Figure size (width, height)
            cmap (str): Colormap for the magnitude plots

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.1])
        fig.suptitle('OFDM Channel Magnitude Response', fontsize=16, y=1.02)

        axes = []
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)

        magnitudes = torch.abs(self.channels)
        vmax = magnitudes.max()
        vmin = magnitudes.min()

        for i in range(self.batch_size):
            im = axes[i].imshow(
                magnitudes[i].numpy(),
                aspect='auto',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin='lower'
            )

            axes[i].set_title(f'Channel {i + 1}')
            axes[i].set_xlabel('Time Symbol Index')
            axes[i].set_ylabel('Subcarrier Index')

        # Add colorbar in the dedicated space
        cax = fig.add_subplot(gs[:, -1])
        fig.colorbar(im, cax=cax, label='Magnitude')

        plt.tight_layout()
        return fig

    def plot_phases(self, figsize=(20, 8), cmap='hsv'):
        """
        Plot phase of channels in a 2x4 grid.

        Args:
            figsize (tuple): Figure size (width, height)
            cmap (str): Colormap for the phase plots

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.1])
        fig.suptitle('OFDM Channel Phase Response', fontsize=16, y=1.02)

        axes = []
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)

        phases = torch.angle(self.channels)

        for i in range(self.batch_size):
            im = axes[i].imshow(
                phases[i].numpy(),
                aspect='auto',
                cmap=cmap,
                vmin=-np.pi,
                vmax=np.pi,
                origin='lower'
            )

            axes[i].set_title(f'Channel {i + 1}')
            axes[i].set_xlabel('Time Symbol Index')
            axes[i].set_ylabel('Subcarrier Index')

        # Add colorbar in the dedicated space
        cax = fig.add_subplot(gs[:, -1])
        fig.colorbar(im, cax=cax, label='Phase (radians)')

        plt.tight_layout()
        return fig


@dataclass
class ChannelInfo:
    """Container for channel information extracted from filenames."""
    file_number: torch.Tensor
    snr: torch.Tensor
    delay_spread: torch.Tensor
    max_doppler_shift: torch.Tensor
    pilot_placement_frequency: torch.Tensor
    channel_type: List[str]


    def as_tuple(self):
        """
        Returns all fields as a tuple in the order they're defined in the dataclass.

        Returns:
            tuple: (file_number, snr, delay_spread, max_doppler_shift,
                   pilot_placement_frequency, channel_type)
        """
        return (
            self.file_number,
            self.snr,
            self.delay_spread,
            self.max_doppler_shift,
            self.pilot_placement_frequency,
            self.channel_type
        )


class ComplexChannelProcessor:
    """Handles processing of complex channel matrices."""

    @staticmethod
    def concat_complex_channel(channel_matrix: ComplexTensor) -> ComplexTensor:
        """
        Convert complex channel matrix to real matrix by concatenating real and imaginary parts.

        Args:
            channel_matrix: Complex channel matrix of shape (B, F, T)
                where B=batch, F=frequency, T=time

        Returns:
            Real-valued matrix of shape (B, F, 2*T)
        """
        return torch.cat(
            (torch.real(channel_matrix), torch.imag(channel_matrix)),
            dim=1
        )

    @staticmethod
    def inverse_concat_complex_channel(channel_matrix: ComplexTensor) -> ComplexTensor:
        """
        Reconstruct complex channel matrix from concatenated real matrix.

        Args:
            channel_matrix: Real-valued matrix of shape (B, F, 2*T)

        Returns:
            Complex matrix of shape (B, F, T)
        """
        split_idx = channel_matrix.shape[-1] // 2
        return torch.complex(
            channel_matrix[:, :split_idx],
            channel_matrix[:, split_idx:]
        )

    @staticmethod
    def extract_values(file_name: str) -> ChannelInfo:
        """
        Extract channel information from filename.

        Args:
            file_name: Name of the file to parse

        Returns:
            ChannelInfo containing extracted values

        Raises:
            ValueError: If filename format is invalid
        """
        pattern = r'(\d+)_SNR-(\d+)_DS-(\d+)_DOP-(\d+)_N-(\d+)_([A-Z\-]+)\.mat'
        match = re.match(pattern, file_name)

        if not match:
            raise ValueError(f"Invalid filename format: {file_name}")

        return ChannelInfo(
            file_number=torch.tensor([int(match.group(1))], dtype=torch.float),
            snr=torch.tensor([int(match.group(2))], dtype=torch.float),
            delay_spread=torch.tensor([int(match.group(3))], dtype=torch.float),
            max_doppler_shift=torch.tensor([int(match.group(4))], dtype=torch.float),
            pilot_placement_frequency=torch.tensor([int(match.group(5))], dtype=torch.float),
            channel_type=[match.group(6)]
        )
