import torch
from torch.utils.data import DataLoader , TensorDataset
import matplotlib.pyplot as plt
from models.model import MyAwesomeModel


def train(dataloader, lr, e):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    
    model = MyAwesomeModel()

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss = []
    for i in range(e):
        print(f"Epoch {i+1} of {e}")
        for images, labels in dataloader:
            # Add dimensions to fit convolution kernel
            images_reshape = images.unsqueeze(1)
            optim.zero_grad()
            logits = model(images_reshape)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            running_loss.append(loss.item())
    

    # Save model checkpoint to use for training later
    torch.save(model.state_dict(), 'models/checkpoint.pth')

    # Save graph of training loss
    x = list(range(len(running_loss)))
    plt.figure()
    plt.plot(x, running_loss)
    plt.xlabel("Batch number")
    plt.ylabel("Training loss")
    plt.savefig("reports/figures/training_loss.png")


if __name__ == "__main__":
    # get data from data/processed and convert to dataloader
    X = torch.load("data/processed/train_img.pt")
    y = torch.load("data/processed/train_target.pt")
    dataset = TensorDataset(X , y)
    trainloader = DataLoader(dataset , batch_size = 64, shuffle=True)
    e = 3
    lr = 0.001
    train(trainloader, lr, e)