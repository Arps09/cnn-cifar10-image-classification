import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.cnn import SimpleCNN


def test_model_output_shape():
    model = SimpleCNN()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)

    # Output should be [batch_size, num_classes]
    assert y.shape == (1, 10)
