import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


def _up_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class UNetPlusPlus3D(nn.Module):
    """Lightweight 3D UNet++ (nested skip connections)."""

    def __init__(self, in_channels=4, out_channels=3, base_filters=32):
        super().__init__()
        f = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 16]

        self.pool = nn.MaxPool3d(2)

        self.conv0_0 = _conv_block(in_channels, f[0])
        self.conv1_0 = _conv_block(f[0], f[1])
        self.conv2_0 = _conv_block(f[1], f[2])
        self.conv3_0 = _conv_block(f[2], f[3])
        self.conv4_0 = _conv_block(f[3], f[4])

        self.up1_0 = _up_block(f[1], f[0])
        self.up2_0 = _up_block(f[2], f[1])
        self.up3_0 = _up_block(f[3], f[2])
        self.up4_0 = _up_block(f[4], f[3])

        self.conv0_1 = _conv_block(f[0] + f[0], f[0])
        self.conv1_1 = _conv_block(f[1] + f[1], f[1])
        self.conv2_1 = _conv_block(f[2] + f[2], f[2])
        self.conv3_1 = _conv_block(f[3] + f[3], f[3])

        self.up1_1 = _up_block(f[1], f[0])
        self.up2_1 = _up_block(f[2], f[1])
        self.up3_1 = _up_block(f[3], f[2])

        self.conv0_2 = _conv_block(f[0] * 3, f[0])
        self.conv1_2 = _conv_block(f[1] * 3, f[1])
        self.conv2_2 = _conv_block(f[2] * 3, f[2])

        self.up1_2 = _up_block(f[1], f[0])
        self.up2_2 = _up_block(f[2], f[1])

        self.conv0_3 = _conv_block(f[0] * 4, f[0])
        self.conv1_3 = _conv_block(f[1] * 4, f[1])

        self.up1_3 = _up_block(f[1], f[0])

        self.conv0_4 = _conv_block(f[0] * 5, f[0])

        self.out_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1))

        return self.out_conv(x0_4)


class DeepMedic3D(nn.Module):
    """Simplified DeepMedic-style dual-path 3D CNN baseline."""

    def __init__(self, in_channels=4, out_channels=3, base_filters=24):
        super().__init__()
        self.high = nn.Sequential(
            _conv_block(in_channels, base_filters),
            _conv_block(base_filters, base_filters),
        )
        self.low = nn.Sequential(
            _conv_block(in_channels, base_filters),
            _conv_block(base_filters, base_filters),
        )
        self.low_down = nn.AvgPool3d(2)
        self.fuse = nn.Sequential(
            _conv_block(base_filters * 2, base_filters),
            nn.Conv3d(base_filters, out_channels, kernel_size=1),
        )

    def forward(self, x):
        high = self.high(x)
        low = self.low(self.low_down(x))
        low = F.interpolate(low, size=high.shape[2:], mode="trilinear", align_corners=False)
        fused = torch.cat([high, low], dim=1)
        return self.fuse(fused)


class TransBTS3D(nn.Module):
    """Simplified TransBTS-style hybrid CNN + Transformer baseline."""

    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        base_filters=16,
        num_heads=4,
        transformer_layers=2,
        mlp_ratio=2,
    ):
        super().__init__()
        f = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]

        self.enc1 = _conv_block(in_channels, f[0])
        self.enc2 = _conv_block(f[0], f[1])
        self.enc3 = _conv_block(f[1], f[2])
        self.bottleneck = _conv_block(f[2], f[3])
        self.pool = nn.MaxPool3d(2)

        dim = f[3]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * mlp_ratio,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.up3 = _up_block(f[3], f[2])
        self.dec3 = _conv_block(f[2] + f[2], f[2])
        self.up2 = _up_block(f[2], f[1])
        self.dec2 = _conv_block(f[1] + f[1], f[1])
        self.up1 = _up_block(f[1], f[0])
        self.dec1 = _conv_block(f[0] + f[0], f[0])

        self.out_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))

        b, c, d, h, w = xb.shape
        tokens = xb.view(b, c, d * h * w).permute(0, 2, 1)
        tokens = self.transformer(tokens)
        xb = tokens.permute(0, 2, 1).view(b, c, d, h, w)

        x = self.up3(xb)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.out_conv(x)
