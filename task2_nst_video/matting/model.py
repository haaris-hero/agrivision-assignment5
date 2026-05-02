"""matting/model.py — U-Net for human alpha matting.

Input:  RGB image (3, H, W)
Output: alpha matte (1, H, W) in [0, 1]

Architecture: standard U-Net with 4 encoder/decoder stages + skip connections.
No pretrained weights — trained from scratch on AISegment as required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample + skip connection + DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if skip and x sizes differ (input not perfectly divisible by 16)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNetMatting(nn.Module):
    """U-Net for alpha matte prediction.

    Encoder: 3 → 64 → 128 → 256 → 512
    Bottleneck: 512 → 1024
    Decoder: 1024+512 → 512 → 256+256 → 256 → 128+128 → 128 → 64+64 → 64
    Head: 64 → 1, sigmoid
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = Down(64, 128)
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)
        # Bottleneck
        self.bottleneck = Down(512, 1024)
        # Decoder
        self.dec4 = Up(1024 + 512, 512)
        self.dec3 = Up(512  + 256, 256)
        self.dec2 = Up(256  + 128, 128)
        self.dec1 = Up(128  + 64,   64)
        # Output head
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)           # 64,  H,    W
        s2 = self.enc2(s1)          # 128, H/2,  W/2
        s3 = self.enc3(s2)          # 256, H/4,  W/4
        s4 = self.enc4(s3)          # 512, H/8,  W/8
        bn = self.bottleneck(s4)    # 1024,H/16, W/16

        x = self.dec4(bn, s4)       # 512, H/8
        x = self.dec3(x,  s3)       # 256, H/4
        x = self.dec2(x,  s2)       # 128, H/2
        x = self.dec1(x,  s1)       # 64,  H

        return torch.sigmoid(self.head(x))   # (B, 1, H, W) in [0,1]

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_matting_model():
    model = UNetMatting()
    print(f"[matting] UNetMatting: {model.count_params():,} parameters")
    return model
