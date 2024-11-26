import torch


class PatchDiscriminator(torch.nn.Module):
    """A discriminator that classifies patches of an image as real or fake.

    This implements the PatchGAN architecture from the pix2pix paper, which classifies
    local patches of an image rather than the whole image.

    Args:
        in_channels (int): Number of input channels. Default: 2
        base_filters (int, optional): Number of filters in first conv layer. Default: 16
        norm_layer (nn.Module, optional): Normalization layer to use. Default: BatchNorm2d

    The network consists of:
        - 4 convolutional layers with increasing number of filters
        - LeakyReLU activations
        - Normalization after each conv except last
        - Asymmetric striding (2,1) to handle rectangular inputs
    """

    def __init__(self,
                 in_channels: int = 2,
                 base_filters: int = 16,
                 norm_layer: torch.nn.Module = torch.nn.BatchNorm2d):
        super().__init__()

        if not isinstance(norm_layer, type):
            raise ValueError("norm_layer should be a class, not an instance")

        if norm_layer not in [torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d]:
            raise NotImplementedError(f"Unsupported normalization layer: {norm_layer}")

        # while using batch norm, bias in conv filters become redundant
        use_bias = norm_layer == torch.nn.InstanceNorm2d

        # Common conv layer settings
        kernel_size = 4
        padding = 1

        # Build network as sequence of layers
        layers = [
            # First conv layer: in_channels -> base_filters
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel_size=kernel_size,
                stride=(2, 1),  # Asymmetric stride for rectangular input
                padding=padding,
                bias=use_bias  # Use same bias rule as other layers with norm
            ),
            torch.nn.LeakyReLU(0.2, True),
            norm_layer(base_filters),

            # Second conv layer: base_filters -> base_filters*2
            torch.nn.Conv2d(
                in_channels=base_filters,
                out_channels=base_filters * 2,
                kernel_size=kernel_size,
                stride=(2, 1),
                padding=padding,
                bias=use_bias
            ),
            torch.nn.LeakyReLU(0.2, True),
            norm_layer(base_filters * 2),

            # Third conv layer: base_filters*2 -> base_filters*4
            torch.nn.Conv2d(
                in_channels=base_filters * 2,
                out_channels=base_filters * 4,
                kernel_size=kernel_size,
                stride=(2, 1),
                padding=padding,
                bias=use_bias
            ),
            torch.nn.LeakyReLU(0.2, True),
            norm_layer(base_filters * 4),

            # Final conv layer: base_filters*4 -> 1 (logits)
            torch.nn.Conv2d(
                in_channels=base_filters * 4,
                out_channels=1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=True  # Always use bias in final layer
            )
        ]

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Patch predictions of shape (batch_size, 1, height', width')
                         where height and width are reduced spatial dimensions.
                         Each value represents logit of patch being real.
        """
        return self.model(x)