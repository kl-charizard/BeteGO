from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import POLICY_SIZE


@dataclass
class ModelConfig:
    channels: int = 128
    residual_blocks: int = 5
    policy_head_channels: int = 32
    value_head_channels: int = 32


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x)
        return x


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: ModelConfig, input_channels: int = 18) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, cfg.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.channels),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(*[ResidualBlock(cfg.channels) for _ in range(cfg.residual_blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.policy_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.policy_head_channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(cfg.policy_head_channels * 8 * 8, POLICY_SIZE)

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.value_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.value_head_channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(cfg.value_head_channels * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, 8, 8]
        x = self.stem(x)
        x = self.resblocks(x)

        p = self.policy_head(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # [B, 4672]

        v = self.value_head(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # [-1, 1]

        return p, v.squeeze(-1)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            return self.forward(x) 