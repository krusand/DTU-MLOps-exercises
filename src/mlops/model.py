from torch import nn
from torch.nn.functional import log_softmax, relu


class Mnist_clf(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """Forward pass through the network, returns the output logits."""
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Add dropout layers to divide learning across nodes
        x = self.dropout(relu(self.fc1(x)))
        x = self.dropout(relu(self.fc2(x)))
        x = self.dropout(relu(self.fc3(x)))
        return log_softmax(self.fc4(x), dim=1)


if __name__ == "__main__":
    model = Mnist_clf()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
