"""Module of torch NN model"""

from torch import Tensor
from torch import nn


class Net(nn.Module):
    """
    Torch model class
    """

    def __init__(self):
        super().__init__()

        self.neural_network = nn.Sequential(
            nn.Conv2d(3, 6, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(6, 16, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Flatten(),
            nn.Linear(270400, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, data: Tensor):
        """Make prediction on data
        Args:
            data (tensor): image like tensor
        """
        return self.neural_network(data)
