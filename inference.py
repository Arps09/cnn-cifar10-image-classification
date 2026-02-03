import torch
import sys
import os
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.cnn import SimpleCNN


# CIFAR-10 class names
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transformations (same as test)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cnn_cifar10.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    print(f"Predicted class: {CLASSES[predicted.item()]}")


if __name__ == "__main__":
    # Example image path (change this)
    predict("sample.jpg")
