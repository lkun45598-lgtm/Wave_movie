from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class Temporal3DUNet(nn.Module):
    """3D-convolution U-Net for temporal LR windows stored as NHWC channels.

    OceanNPY temporal windows concatenate LR frames along the last channel as
    [frame_0_vars, frame_1_vars, ..., frame_T_vars]. This adapter reshapes the
    tensor to NCTHW, preserves temporal depth through the U-Net, and returns the
    center frame in the framework's expected NHWC format.
    """

    def __init__(self, args):
        super().__init__()
        in_channels = int(args.get("in_channels", 15))
        out_channels = int(args.get("out_channels", 1))
        temporal_window = int(args.get("temporal_window", 5))
        channels_per_frame = int(args.get("channels_per_frame", 0) or 0)
        init_features = int(args.get("init_features", 16))

        if temporal_window < 1 or temporal_window % 2 == 0:
            raise ValueError(
                f"temporal_window must be a positive odd integer, got {temporal_window}"
            )
        if channels_per_frame <= 0:
            if in_channels % temporal_window != 0:
                raise ValueError(
                    "in_channels must be divisible by temporal_window when "
                    "channels_per_frame is not set"
                )
            channels_per_frame = in_channels // temporal_window
        if in_channels != temporal_window * channels_per_frame:
            raise ValueError(
                "in_channels must equal temporal_window * channels_per_frame "
                f"({in_channels} != {temporal_window} * {channels_per_frame})"
            )
        if init_features < 1:
            raise ValueError(f"init_features must be >= 1, got {init_features}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_window = temporal_window
        self.channels_per_frame = channels_per_frame
        self.center_index = temporal_window // 2

        features = init_features
        self.encoder1 = self._block(channels_per_frame, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.bottleneck = self._block(features * 4, features * 8, name="bottleneck")

        self.upconv3 = nn.ConvTranspose3d(
            features * 8,
            features * 4,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.decoder3 = self._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4,
            features * 2,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.decoder2 = self._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2,
            features,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, H, W, C], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[-1]}"
            )

        batch, height, width, _ = x.shape
        x = x.contiguous().view(
            batch,
            height,
            width,
            self.temporal_window,
            self.channels_per_frame,
        )
        x = x.permute(0, 4, 3, 1, 2).contiguous()

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.conv(dec1)
        out = out[:, :, self.center_index]
        return out.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv3d(
                            in_channels,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_norm1", nn.BatchNorm3d(features)),
                    (name + "_act1", nn.GELU()),
                    (
                        name + "_conv2",
                        nn.Conv3d(
                            features,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm3d(features)),
                    (name + "_act2", nn.GELU()),
                ]
            )
        )
