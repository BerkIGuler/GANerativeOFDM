import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout=0.3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, p_dropout=0.3, debug=False):
        super().__init__()

        self.debug = debug
        # Encoder (contracting path)
        self.enc1 = DoubleConv(2, 64, p_dropout)
        self.enc2 = DoubleConv(64, 128, p_dropout)
        self.enc3 = DoubleConv(128, 256, p_dropout)

        # Decoder (expansive path)
        self.dec3 = DoubleConv(256 + 128, 128, p_dropout)
        self.dec2 = DoubleConv(128 + 64, 64, p_dropout)
        self.dec1 = DoubleConv(64, 32, p_dropout)

        # Final convolution
        self.final_conv = nn.Conv2d(32, 2, kernel_size=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)

    def forward(self, x):

        # Encoder
        enc1 = self.enc1(x)
        self._debug("enc1", enc1)
        x = self.pool(enc1)
        self._debug("x", x)

        enc2 = self.enc2(x)
        self._debug("enc2", enc2)
        x = self.pool(enc2)
        self._debug("x", x)

        x = self.enc3(x)
        self._debug("x", x)

        # Decoder
        x = self.upsample(x)
        self._debug("x", x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)
        self._debug("x", x)

        x = self.upsample(x)
        self._debug("x", x)
        x = torch.cat([x, enc1], dim=1)
        self._debug("x", x)
        x = self.dec2(x)
        self._debug("x", x)

        x = self.dec1(x)
        self._debug("x", x)
        x = self.final_conv(x)
        self._debug("x", x)

        return x

    def _debug(self, name, x):
        if self.debug:
            print(f"{name}: {x.size()}")


# Example usage
def test_unet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(p_dropout=0.1, debug=True).to(device)

    # Test with batch_size=1 to demonstrate instance normalization behavior
    batch_size = 1

    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x = torch.randn(batch_size, 2, 120, 14).to(device)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    return model


if __name__ == "__main__":
    model = test_unet()