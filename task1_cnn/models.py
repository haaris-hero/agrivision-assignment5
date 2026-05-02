"""models.py — Model A (BaselineCNN) and Model B (DeeperCNN) for seed count regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0, l2=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
        self.l2 = l2  # stored for weight decay reference; actual decay applied in optimizer

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """Model A: 3 conv blocks (32→64→128), Global Average Pooling, Dense → output.
    Designed to stay under 1.5M parameters.
    """

    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)      # Global Average Pooling
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),                   # regression output: seed count
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.head(x).squeeze(1)          # shape: (batch,)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Optional residual connection for Model B (same in/out channels)."""

    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class DeeperCNN(nn.Module):
    """Model B: 4 conv blocks (32→64→128→256), dropout 0.3, L2 weight decay,
    residual connection on the last two-block group.
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.block1 = ConvBlock(3, 32, use_bn=True, dropout=0.0)
        self.block2 = ConvBlock(32, 64, use_bn=True, dropout=0.0)
        self.block3 = ConvBlock(64, 128, use_bn=True, dropout=dropout)
        self.residual = ResidualBlock(128, dropout=dropout)   # residual at 128-ch level
        self.block4 = ConvBlock(128, 256, use_bn=True, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.residual(x)    # residual before downsampling
        x = self.block4(x)
        x = self.gap(x)
        return self.head(x).squeeze(1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(name, cfg=None):
    """Factory: returns (model, l2_weight_decay)."""
    if name == "model_a":
        model = BaselineCNN()
        wd = 0.0
    elif name == "model_b":
        dropout = cfg["model_b"]["dropout"] if cfg else 0.3
        model = DeeperCNN(dropout=dropout)
        wd = cfg["model_b"]["l2_weight_decay"] if cfg else 1e-4
    else:
        raise ValueError(f"Unknown model: {name}")

    n = model.count_params()
    print(f"[model] {model.__class__.__name__}: {n:,} parameters")
    if name == "model_a":
        assert n <= 1_500_000, f"Model A exceeds 1.5M params: {n}"
    return model, wd
