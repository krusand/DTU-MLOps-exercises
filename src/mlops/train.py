import matplotlib.pyplot as plt
import torch
import typer
from typing import Annotated
from torch import nn
from tqdm import tqdm

from mlops.data import corrupt_mnist
from mlops.model import Mnist_clf

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

app = typer.Typer()


@app.command()
def train(
        lr: Annotated[float, typer.Option("--learning-rate", "-lr")] = 1e-3, 
        n_epochs: Annotated[int, typer.Option("--n-epochs", "-ne")] = 10,
        output: Annotated[str, typer.Option("--output", "-o")] = "models/model.pt"
    ):
    """
    Trains the model and saves it to the models directory.
    Additionally saves a plot of training loss pr. epoch

    Parameters:
        lr (float): Learning rate for optimizer
        n_epochs (int): Number of epochs to train model
    Returns:
        None
    """

    model = Mnist_clf().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr
    )  # AdamW is faster optimizer compared to Adam/SGD

    train_losses = []
    for _ in tqdm(range(n_epochs)):
        model.train()
        batch_loss = 0
        for images, labels in train_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        train_losses.append(batch_loss)

    torch.save(model.state_dict(), output)

    plt.plot(range(0, n_epochs), train_losses)
    plt.title("Training loss")
    plt.savefig("reports/figures/training_loss.png")


if __name__ == "__main__":
    app()
