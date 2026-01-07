# PyTorch Sports Ball Classifier 
A Convolutional Neural Network (CNN) built from scratch using **PyTorch** to identify and classify different types of sports balls. This project demonstrates a full end-to-end deep learning pipeline, from custom data preprocessing to model evaluation.

## Project Overview 
The goal of this project is to distinguish between three types of sports equipment:
- **Tennis Balls** (Focusing on color and texture)
- **Basketballs** (Focusing on color and line patterns)
- **Soccer Balls** (Focusing on high-contrast geometric patterns)
By using a CNN, the model learns to identify these specific visual features rather than just memorizing pixels.

## Tech Stack 
- **Language**: Python
- **Framework**: PyTorch
- **Data Handling**: Torchvision (transforms, ImageFolder)
- **Visualization**: Matplotlib, NumPy

## Model Architecture 
The model is based on a **LeNet-style architecture**, modified to handle RGB color images and a custom input resolution.
- **Convolutional Layers**: Extract spatial features (edges, textures, shapes).
- **Max Pooling**: Reduces dimensionality while preserving the most important signals.
Fully Connected (Linear) Layers: Acts as the "brain" to classify the extracted features into one of the 3 classes.
- **Activation**: ReLU (Rectified Linear Unit) for non-linearity.

# Dataset Structure 
The project uses a custom dataset organized for easy loading via 
```torchvision.datasets.ImageFolder```:
```
data/
├── train/
│   ├── tennis/
│   ├── basketball/
│   └── soccer/
└── test/
    ├── tennis/
    ├── basketball/
    └── soccer/
```

## Learning Milestones 
- **Data Augmentation**: Implemented transforms to resize and normalize raw images.
- **The Training Loop**: Implemented Forward/Backward passes and gradient updates.
- **Generalization**: Verified learning by testing the model on a separate "held-out" test set to ensure the model isn't just memorizing data (overfitting).

## Results
- **Training Epochs**: 2 (or more)
- **Optimizer**: SGD with Momentum
- **Loss Function**: CrossEntropyLoss
- **Final Accuracy**: 68.3%
- **Final Accuracy per Class**: basketball: 41.9%, football: 78.8%, tennis: 74.0%
