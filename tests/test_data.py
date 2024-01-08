import torch
import pytest
import os
from torch.utils.data import TensorDataset

@pytest.mark.skipif(not os.path.exists("data/processed/train_img.pt"), reason="Data files not found")

def test_data():
    train_tensors = torch.load("data/processed/train_img.pt")
    train_target = torch.load("data/processed/train_target.pt")
    train_mnist = TensorDataset(train_tensors, train_target)

    test_tensor = torch.load("data/processed/test_img.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_mnist = TensorDataset(test_tensor, test_target)
    # Tests
    assert train_tensors[0].shape == (28, 28)
    assert len(train_mnist) == 30000
    assert len(test_mnist) == 5000
    unique = torch.unique(train_target)
    labels = [0,1,2,3,4,5,6,7,8,9]
    assert  all([x in unique for x in labels])