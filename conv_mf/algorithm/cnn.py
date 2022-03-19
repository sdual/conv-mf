import torch.nn as nn
import numpy as np


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self._layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16)
        )

    def forward(self, X: np.ndarray):
        pass
