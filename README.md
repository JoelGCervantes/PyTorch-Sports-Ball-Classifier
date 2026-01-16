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

## Training Improvements (Hyperparameter Tuning)
The initial model achieved 58.3% test accuracy. By tuning key training hyperparameters, I improved performance to **82.5%**.

Changes made:
- **Batch size:** increased from `10` → `5`
- **Momentum (SGD):** increased from `0.9` → `0.5`
- **Epochs:** increased from `5` → `20`

Why this helped:
- **Batch size** affects the stability/noisiness of gradient updates (tradeoff: stability vs generalization).
- **Momentum** accelerates learning in consistent directions and helps reduce oscillations.
- **More epochs** gives the model more opportunities to fit useful features (with the risk of overfitting, checked using a held-out test set).

## Results
- **Training Epochs**: 20
- **Optimizer**: SGD with Momentum=5
- **Loss Function**: CrossEntropyLoss
- **Final Accuracy**: 82.5%
- **Final Accuracy per Class**: basketball: 72.1%, football: 84.8%, tennis: 87.0%
