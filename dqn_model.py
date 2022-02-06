import torch
import torch.nn as nn
import numpy as np
from typing import Union


class OdorDQN(nn.Module):
    def __init__(self, device: Union[str, int, torch.device] = "cpu"):
        super().__init__()
        self.device = device
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, obs, state=None, info={}):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        # if not isinstance(obs, torch.Tensor):
        #     obs = torch.tensor(obs, dtype=torch.float)
        logits = self.cnn_layer(obs)
        logits = self.fc_layer(logits)
        return logits, state
