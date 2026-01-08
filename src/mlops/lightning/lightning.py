from pytorch_lightning import LightningModule, Trainer
from torch import nn, optim
import torch


def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    train_images = torch.load("data/preprocessed/train_images.pt")
    train_target = torch.load("data/preprocessed/train_target.pt")
    test_images = torch.load("data/preprocessed/test_images.pt")
    test_target = torch.load("data/preprocessed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        """Forward pass"""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def training_step(self, batch):
        img, target = batch
        y_pred = self(img)
        return self.loss_fn(y_pred, target)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
if __name__ == '__main__':
    model = Model()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


    train, test = corrupt_mnist()
    train = torch.utils.data.DataLoader(train, num_workers=4, persistent_workers=True)
    test = torch.utils.data.DataLoader(test, num_workers=4, persistent_workers=True)

    trainer = Trainer(max_epochs=10, limit_train_batches=0.2)

    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=test)
    
