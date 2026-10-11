# Handwritten Digit Recognition Using CNN

## Project Overview

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from 0 to 9 using the MNIST dataset.

## Objectives

- Build a deep learning model for handwritten digit classification.
- Preprocess grayscale images for model training.
- Train and evaluate a CNN model on unseen test data.
- Visualize predictions for handwritten digits.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## Dataset

The project uses the MNIST handwritten digit dataset, which contains 60,000 training images and 10,000 test images. Each image is a 28 × 28 grayscale image representing a handwritten digit from 0 to 9.

## Model Architecture

- Convolutional layers for feature extraction
- Max pooling layers for dimensionality reduction
- Flatten layer
- Fully connected dense layer
- Softmax output layer for 10 digit classes

## Model Training

The CNN model was trained using the following configuration:

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Epochs:** 5
- **Batch Size:** 64
- **Validation Split:** 10%

## Evaluation Results

The trained model was evaluated on 10,000 unseen MNIST test images.

- **Test Accuracy:** 99.01%
- **Test Loss:** 0.0350
- **Sample Predictions:** 10 out of 10 correct predictions on the selected sample images.

These results demonstrate the model's performance on the MNIST handwritten digit classification task.

## How to Run

1. Open the project in Google Colab or a local Python environment.
2. Install the required libraries: TensorFlow, NumPy, Matplotlib, and Scikit-learn.
3. Run the Python source code to load the dataset, preprocess images, train the model, and evaluate its performance.

## Author

Irshad Ali

## Learning Outcomes

- Gained practical experience with Convolutional Neural Networks.
- Learned image preprocessing and normalization techniques.
- Practiced deep learning model training and evaluation.
- Evaluated classification performance using test accuracy and loss.
