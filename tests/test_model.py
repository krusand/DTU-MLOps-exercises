from mlops.model import Mnist_clf
import torch
from loguru import logger


def test_model_output_shape():
    model = Mnist_clf()

    # Dummy data
    img = torch.rand(size=(1, 784))
    output = model(img)
    assert output.shape == (1, 10)
