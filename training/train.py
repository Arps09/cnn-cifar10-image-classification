import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_dataloaders
from models.cnn import SimpleCNN


def train():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, _ = get_dataloaders(batch_size=64)

    # Initialize model
    model = SimpleCNN().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "cnn_cifar10.pth")
    print("Model saved as cnn_cifar10.pth")


if __name__ == "__main__":
    train()
