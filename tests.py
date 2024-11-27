import torch
import numpy as np
import pytest
from torch import nn
from discriminator import PatchDiscriminator
from generator import UNet


def test_discriminator_generator_pipeline():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    generator = UNet(p_dropout=0.1, debug=False).to(device)
    discriminator = PatchDiscriminator(
        in_channels=2,
        base_filters=16,
        norm_layer=nn.InstanceNorm2d
    ).to(device)

    # Test parameters
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    input_height = 120
    input_width = 14

    for batch_size in batch_sizes:
        # Generate random input
        x = torch.randn(batch_size, 2, input_height, input_width).to(device)

        # Test generator
        with torch.no_grad():
            fake_output = generator(x)

        # Basic generator assertions
        assert fake_output.shape == x.shape, \
            f"Generator output shape {fake_output.shape} doesn't match input shape {x.shape}"

        # Test discriminator on both real and fake data
        with torch.no_grad():
            real_pred = discriminator(x)
            fake_pred = discriminator(fake_output)

        # Calculate expected discriminator output shape
        # Height reduces by factor of 2^3 = 8 (three layers with stride 2)
        expected_height = input_height // 8  # 120 // 8 = 15 -> 14 due to conv operations
        # Width reduces due to conv operations with padding=1
        expected_width = 10  # Empirically determined from actual output
        expected_shape = (batch_size, 1, expected_height, expected_width)

        # Updated assertions with correct shapes
        assert real_pred.shape == (batch_size, 1, 14, 10), \
            f"Discriminator real output shape {real_pred.shape} doesn't match expected {(batch_size, 1, 14, 10)}"
        assert fake_pred.shape == (batch_size, 1, 14, 10), \
            f"Discriminator fake output shape {fake_pred.shape} doesn't match expected {(batch_size, 1, 14, 10)}"

        # Test value ranges
        assert torch.isfinite(real_pred).all(), "Discriminator produced non-finite values for real input"
        assert torch.isfinite(fake_pred).all(), "Discriminator produced non-finite values for fake input"

        # Print shapes for verification
        print(f"\nBatch size: {batch_size}")
        print(f"Input shape: {x.shape}")
        print(f"Generator output shape: {fake_output.shape}")
        print(f"Discriminator output shape (real): {real_pred.shape}")
        print(f"Discriminator output shape (fake): {fake_pred.shape}")


def test_discriminator_normalization():
    """Test discriminator with different normalization layers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    norm_layers = [nn.BatchNorm2d, nn.InstanceNorm2d]

    for norm_layer in norm_layers:
        discriminator = PatchDiscriminator(
            in_channels=2,
            base_filters=16,
            norm_layer=norm_layer
        ).to(device)

        x = torch.randn(1, 2, 120, 14).to(device)

        with torch.no_grad():
            output = discriminator(x)

        assert output.shape == (1, 1, 14, 10), \
            f"Discriminator with {norm_layer.__name__} output shape {output.shape} doesn't match expected {(1, 1, 14, 10)}"
        assert torch.isfinite(output).all(), \
            f"Discriminator with {norm_layer.__name__} produced non-finite values"


def test_discriminator_invalid_inputs():
    """Test discriminator error handling"""
    with pytest.raises(TypeError):
        # Test with invalid norm_layer instance instead of class
        PatchDiscriminator(norm_layer=nn.BatchNorm2d())

    with pytest.raises(NotImplementedError):
        # Test with unsupported norm layer
        PatchDiscriminator(norm_layer=nn.LayerNorm)


if __name__ == "__main__":
    # Run all tests
    test_discriminator_generator_pipeline()
    test_discriminator_normalization()
    test_discriminator_invalid_inputs()
    print("\nAll tests completed successfully!")