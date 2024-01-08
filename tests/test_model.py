from working_mnist.models.model import MyAwesomeModel
import torch

def test_output_shape():
    model = MyAwesomeModel()
    input = torch.ones(1, 28, 28)
    assert model(input).shape == torch.Size([1, 10])