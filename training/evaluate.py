import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_dataloaders
from models.cnn import SimpleCNN


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloaders(batch_size=64)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cnn_cifar10.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate()
