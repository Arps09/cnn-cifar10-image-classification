# End-to-End Image Classification using PyTorch (CIFAR-10)

This project implements a complete image classification pipeline using **PyTorch** and the **CIFAR-10** dataset.  
It covers data loading, model training, evaluation, inference, and unit testing.

---

## 📌 Project Features
- CIFAR-10 data loading with augmentation
- Convolutional Neural Network (CNN) model
- Training loop with loss tracking and model checkpointing
- Evaluation on test dataset
- Single-image inference script
- Unit tests using pytest

---

## 📁 Project Structure

projectcifar10/
│
├── data/
│ └── dataset.py
│
├── models/
│ └── cnn.py
│
├── training/
│ ├── train.py
│ └── evaluate.py
│
├── tests/
│ ├── test_dataset.py
│ └── test_model.py
│
├── inference.py
├── requirements.txt
├── cnn_cifar10.pth
└── README.md


---

## ⚙️ Setup Instructions

### 1️⃣ Create virtual environment
py -m venv venv
venv\Scripts\activate

### 2️⃣ Install dependencies
pip install -r requirements.txt

## 🚀 Training the Model
py training/train.py

This will:
- Train the CNN for 5 epochs
- Save the trained model as cnn_cifar10.pth

## 📊 Model Evaluation
py training/evaluate.py

Example output:
Test Accuracy: ~71%

## 🖼️ Inference (Single Image Prediction)

- Place an image in the project root
- Rename it to sample.jpg

Run:
py inference.py

Example output:
Predicted class: ship



## 🧪 Run Unit Tests
pytest
Expected result:
2 passed


## 🧠 Notes
- CIFAR-10 images are 32×32, so real-world images may not always be predicted accurately
- This project focuses on correctness, structure, and ML fundamentals

## ✅ Technologies Used
- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- pytest