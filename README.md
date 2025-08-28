🖼️ CIFAR-10 Image Classification with CNN
📌 Overview

This project focuses on classifying images from the CIFAR-10 dataset into 10 categories using a Convolutional Neural Network (CNN) built in PyTorch.
The dataset consists of small natural images, and the model is trained to recognize objects such as airplanes, cars, cats, dogs, and ships.

Our CNN model successfully learned visual patterns and achieved strong accuracy on test data.

📂 Dataset

The CIFAR-10 dataset contains:

Training images: 50,000

Test images: 10,000

Image size: 32×32 pixels (RGB)

Classes (10 categories): Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

👉 The dataset is provided by Torchvision and is widely used for benchmarking image classification models.

🎯 Objectives

To understand image classification using the CIFAR-10 dataset.

To learn the importance of CNN architecture (convolution, pooling, and fully connected layers).

To evaluate and improve model performance on test data.

⚙️ Model & Training

Framework: PyTorch

Input Image Size: 32×32

Batch Size: 32

Optimizer: SGD

Loss Function: CrossEntropyLoss

Device: GPU (CUDA)

The model includes:

Convolutional layers for feature extraction

Pooling layers for dimensionality reduction

Fully connected layers for classification

📊 Results

Epochs: 8

Training Accuracy: 82.28%

Test Accuracy: 76.18%

✅ The model shows steady improvements across epochs.
✅ Achieves good classification accuracy on unseen test images.

✅ Key Takeaways

CNNs effectively learn hierarchical features from image data.

Achieved solid accuracy on the CIFAR-10 dataset with a relatively simple architecture.

Provides a foundation for experimenting with deeper networks, augmentation, and transfer learning.

📦 Requirements

torch

torchvision

numpy
