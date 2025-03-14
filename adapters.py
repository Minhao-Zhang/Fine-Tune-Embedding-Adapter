import torch
import torch.nn as nn


class BaseAdapter(nn.Module):
    """Basic adapter class."""

    def __init__(self, input_dim: int):
        """Initialize BasicAdapter."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for BasicAdapter."""
        return x


class LinearAdapter(BaseAdapter):
    """Linear adapter class."""

    def __init__(self, input_dim: int):
        """Initialize LinearAdapter."""
        super().__init__(input_dim)
        self.linear = nn.Linear(input_dim, input_dim, bias=True)

        # Initialize weight as identity matrix
        # with torch.no_grad():  # Disable gradient updates for initialization
        #     self.linear.weight.copy_(torch.eye(input_dim))
        #     self.linear.bias.zero_()  # Set bias to zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LinearAdapter."""
        return self.linear(x)


class MLPAdapter(BaseAdapter):
    """MLP adapter class."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """Initialize MLPAdapter."""
        super().__init__(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLPAdapter."""
        x = self.linear1(x)
        x = self.linear2(x)
        return x
