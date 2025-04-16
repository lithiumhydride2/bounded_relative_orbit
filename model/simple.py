import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class simple_mle(nn.Module):

    def __init__(self):
        """
         Simple MLE model for regression.
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,
                                           64), nn.ReLU(), nn.Linear(64, 64),
                                 nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)
