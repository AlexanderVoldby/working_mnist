import torch
from torch.utils.data import Dataset, DataLoader
import click
from working_mnist.models.model import MyAwesomeModel
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        if data.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if data.shape[1] != 1 or data.shape[2] != 28 or data.shape[3] != 28:
            raise ValueError("Expected input to be shape (1, 28, 28)")
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        # You may need to add additional preprocessing if necessary
        return image


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.argument("model_checkpoint")
@click.argument("image_file")
def predict(model_checkpoint, image_file):
    """Evaluate a trained model."""

    model = MyAwesomeModel()
    # Load previously saved model checkpoint
    state_dict = torch.load(model_checkpoint)
    # Initiate model with pretrain parameters
    model.load_state_dict(state_dict)
    dataset = CustomDataset(file_path=image_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    pred = predict(model, dataloader)

    return pred

if __name__ == "__main__":
    cli()
