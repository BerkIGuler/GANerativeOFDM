"""Module for loading and processing .mat files containing channel estimates for PyTorch."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

from utils import ComplexChannelProcessor, ChannelInfo

__all__ = ["MatDataset", "get_test_dataloaders"]


class ReturnType(Enum):
    """Enumeration of supported return types for channel data."""
    TWO_CHANNEL = "2channel"
    COMPLEX = "complex"
    CONCAT_TIME = "concat_time"
    INTERPOLATED_COMPLEX = "interpolated_complex"

    @classmethod
    def from_string(cls, value: str) -> "ReturnType":
        """Convert string to ReturnType, case-insensitive."""
        try:
            return next(t for t in cls if t.value == value.lower())
        except StopIteration:
            raise ValueError(
                f"Invalid return_type: {value}. Must be one of: "
                f"{', '.join(t.value for t in cls)}"
            )


@dataclass
class PilotDimensions:
    """Container for pilot signal dimensions."""
    num_subcarriers: int
    num_ofdm_symbols: int

    def __post_init__(self):
        """Validate dimensions after initialization."""
        if self.num_subcarriers <= 0 or self.num_ofdm_symbols <= 0:
            raise ValueError("Pilot dimensions must be positive integers")

    def as_tuple(self) -> Tuple[int, int]:
        """Return dimensions as a tuple."""
        return (self.num_subcarriers, self.num_ofdm_symbols)


class MatDataset(Dataset):
    """Dataset for loading and formatting .mat files containing channel estimates.

    Processes .mat files containing channel estimation data and converts them into
    PyTorch tensors. Supports multiple return formats for different processing needs.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        pilot_dims: Tuple[int, int],
        transform: Optional[Callable] = None,
        return_type: str = "2channel"
    ) -> None:
        """Initialize the MatDataset.

        Args:
            data_dir: Path to the directory containing .mat files.
            pilot_dims: Dimensions of pilot data as (num_subcarriers, num_ofdm_symbols).
            transform: Optional transformation to apply to samples.
            return_type: Format of returned tensors. One of: "2channel", "complex",
                "concat_time", or "interpolated_complex".

        Raises:
            ValueError: If return_type is not recognized or pilot dimensions are invalid.
            FileNotFoundError: If data_dir doesn't exist.
        """
        self.data_dir = Path(data_dir)
        self.pilot_dims = PilotDimensions(*pilot_dims)
        self.transform = transform
        self.return_type = ReturnType.from_string(return_type)
        self.channel_processor = ComplexChannelProcessor()

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.file_list = list(self.data_dir.glob("*.mat"))
        if not self.file_list:
            raise FileNotFoundError(f"No .mat files found in {self.data_dir}")

    def __len__(self) -> int:
        """Return the total number of files in the dataset."""
        return len(self.file_list)

    def _process_2channel_complex(
        self,
        h_ideal: torch.Tensor,
        mat_data: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process data for 2channel and complex return types.

        Args:
            h_ideal: Ground truth channel tensor.
            mat_data: Loaded .mat file data.

        Returns:
            Tuple of (processed channel estimate, processed ground truth).

        Raises:
            ValueError: If the data format is unexpected.
        """
        try:
            # Extract LS channel estimate including zero entries
            hzero_ls = torch.tensor(mat_data["H"][:, :, 1], dtype=torch.cfloat)

            # Remove zero entries, keep only pilot values (non-zero values)
            zero_complex = torch.complex(torch.tensor(0.0), torch.tensor(0.0))
            hp_ls = hzero_ls[hzero_ls != zero_complex]

            if hp_ls.numel() != (self.pilot_dims.num_subcarriers *
                               self.pilot_dims.num_ofdm_symbols):
                raise ValueError("Unexpected number of non-zero elements in channel estimate")

            hp_ls = hp_ls.unsqueeze(dim=1).view(
                self.pilot_dims.num_ofdm_symbols,
                self.pilot_dims.num_subcarriers
            ).t()

            if self.return_type == ReturnType.TWO_CHANNEL:
                # Split and concatenate real and imaginary parts
                h_est = torch.cat([
                    torch.real(hp_ls).unsqueeze(0),
                    torch.imag(hp_ls).unsqueeze(0)
                ], dim=0)

                h_ideal = torch.cat([
                    torch.real(h_ideal).unsqueeze(0),
                    torch.imag(h_ideal).unsqueeze(0)
                ], dim=0)
            else:  # ReturnType.COMPLEX
                h_est = hp_ls

            return h_est, h_ideal

        except (RuntimeError, IndexError) as e:
            raise ValueError(f"Error processing channel data: {str(e)}")

    def _process_time_based(
        self,
        h_ideal: torch.Tensor,
        mat_data: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process data for concat_time and interpolated_complex return types.

        Args:
            h_ideal: Ground truth channel tensor.
            mat_data: Loaded .mat file data.

        Returns:
            Tuple of (processed channel estimate, processed ground truth).

        Raises:
            ValueError: If the data format is unexpected.
        """
        try:
            h_ls_interp = torch.tensor(mat_data["H"][:, :, 2], dtype=torch.cfloat)

            if self.return_type == ReturnType.CONCAT_TIME:
                return (
                    self.channel_processor.concat_complex_channel(h_ls_interp),
                    self.channel_processor.concat_complex_channel(h_ideal)
                )
            else:  # ReturnType.INTERPOLATED_COMPLEX
                return h_ls_interp, h_ideal

        except (RuntimeError, IndexError) as e:
            raise ValueError(f"Error processing time-based data: {str(e)}")

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Load and process a .mat file at the given index.

        Args:
            idx: Index of the file to load.

        Returns:
            Tuple containing:
                - Processed LS channel estimate tensor
                - Ground truth channel estimate tensor
                - Metadata extracted from filename

        Raises:
            ValueError: If file format is invalid or processing fails.
            IndexError: If idx is out of range.
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        try:
            # Load .mat file
            mat_data = sio.loadmat(self.file_list[idx])
            if "H" not in mat_data or mat_data["H"].shape[-1] < 3:
                raise ValueError("Invalid .mat file format: missing required data")

            # Ground truth channel
            h_ideal = torch.tensor(mat_data["H"][:, :, 0], dtype=torch.cfloat)

            # Process data based on return type
            if self.return_type in (ReturnType.TWO_CHANNEL, ReturnType.COMPLEX):
                h_est, h_ideal = self._process_2channel_complex(h_ideal, mat_data)
            else:
                h_est, h_ideal = self._process_time_based(h_ideal, mat_data)

            # Extract metadata from filename
            meta_data = self.channel_processor.extract_values(self.file_list[idx].name)
            if meta_data is None:
                raise ValueError(f"Unrecognized filename format: {self.file_list[idx].name}")

            # Apply optional transforms
            if self.transform:
                h_est = self.transform(h_est)
                h_ideal = self.transform(h_ideal)

            return h_est, h_ideal, meta_data.as_tuple()

        except Exception as e:
            raise ValueError(f"Error processing file {self.file_list[idx]}: {str(e)}")


def get_test_dataloaders(
    dataset_dir: Union[str, Path],
    params: dict,
    return_type: str
) -> List[Tuple[str, DataLoader]]:
    """Create DataLoaders for each subdirectory in the dataset directory.

    Args:
        dataset_dir: Path to main directory containing dataset subdirectories.
        params: Configuration parameters including:
            - pilot_dims: Tuple of (num_subcarriers, num_ofdm_symbols)
            - batch_size: Number of samples per batch
        return_type: Format of returned tensors (e.g., "2channel", "complex")

    Returns:
        List of tuples containing (subdirectory_name, corresponding_dataloader)

    Raises:
        FileNotFoundError: If dataset_dir doesn't exist.
        ValueError: If params are invalid or no valid subdirectories are found.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if not isinstance(params, dict) or "pilot_dims" not in params or "batch_size" not in params:
        raise ValueError("params must contain 'pilot_dims' and 'batch_size'")

    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {dataset_dir}")

    test_datasets = [
        (
            subdir.name,
            MatDataset(
                subdir,
                params["pilot_dims"],
                return_type=return_type
            )
        )
        for subdir in subdirs
    ]

    return [
        (name, DataLoader(
            dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=0  # Safer default, can be overridden through params
        ))
        for name, dataset in test_datasets
    ]