import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_dataloaders


def test_dataloader():
    train_loader, test_loader = get_dataloaders(batch_size=32)

    # Get one batch
    images, labels = next(iter(train_loader))

    # Check shapes
    assert images.shape[1:] == (3, 32, 32)
    assert images.shape[0] == labels.shape[0]

    # Check data type
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
