# Convolutional Neural Network (CNN) Implementation

A comprehensive implementation of Convolutional Neural Networks for image classification tasks.

## Overview

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network using TensorFlow and Keras. CNNs are specifically designed for processing grid-like data such as images.

## What is a Convolutional Neural Network?

A **Convolutional Neural Network (CNN)** is a specialized type of neural network designed for processing structured grid data, particularly images. CNNs use a mathematical operation called convolution to automatically learn spatial hierarchies of features.

### Key Components:

1. **Convolutional Layers**: Extract features using learnable filters
2. **Pooling Layers**: Reduce spatial dimensions
3. **Activation Functions**: Introduce non-linearity (ReLU)
4. **Fully Connected Layers**: Perform final classification
5. **Batch Normalization**: Normalize layer inputs
6. **Dropout**: Regularization technique

## Architecture

```
Input (28x28x1)
    ↓
Conv2D (32 filters, 3x3) → BatchNorm → ReLU
    ↓
Conv2D (32 filters, 3x3) → BatchNorm → ReLU
    ↓
MaxPooling (2x2) → Dropout(0.25)
    ↓
Conv2D (64 filters, 3x3) → BatchNorm → ReLU
    ↓
Conv2D (64 filters, 3x3) → BatchNorm → ReLU
    ↓
MaxPooling (2x2) → Dropout(0.25)
    ↓
Flatten
    ↓
Dense (256) → BatchNorm → Dropout(0.5)
    ↓
Dense (128) → BatchNorm → Dropout(0.5)
    ↓
Output (10 classes, Softmax)
```

## Topics Covered

1. **Image Preprocessing**
   - Reshaping for CNN input
   - Normalization (0-1 scaling)
   - One-hot encoding labels

2. **CNN Architecture**
   - Convolutional layers
   - Pooling layers
   - Batch normalization
   - Dropout regularization

3. **Data Augmentation**
   - Rotation
   - Zoom
   - Width/height shifts
   - Improving generalization

4. **Training Techniques**
   - Early stopping
   - Learning rate reduction
   - Validation monitoring

5. **Visualization**
   - Feature maps
   - Convolutional filters
   - Prediction analysis

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Usage

```bash
cd Implementation/CNN
jupyter notebook CNN.ipynb
```

## Dataset

### MNIST Dataset (Default)
- **Images**: 60,000 training + 10,000 test
- **Size**: 28x28 grayscale images
- **Classes**: 10 (digits 0-9)
- **Task**: Multi-class image classification

### Using Custom Datasets

For custom image datasets:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical'
)
```

## Model Performance

Expected performance on MNIST:
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~99%
- **Training Time**: 5-10 minutes (CPU)
- **Parameters**: ~500,000

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Conv Layers | 4 | Feature extraction layers |
| Filters | 32, 32, 64, 64 | Number of filters per layer |
| Kernel Size | 3x3 | Convolutional filter size |
| Pool Size | 2x2 | Max pooling dimensions |
| Activation | ReLU | Hidden layer activation |
| Output Activation | Softmax | Multi-class classification |
| Optimizer | Adam | Adaptive learning rate |
| Loss Function | Categorical Crossentropy | Multi-class loss |
| Batch Size | 64 | Samples per update |
| Epochs | 30 | Maximum training iterations |
| Dropout | 0.25, 0.5 | Regularization rates |

## Key Concepts

### 1. Convolution Operation
- Applies filters to extract features
- Each filter detects specific patterns
- Early layers: edges, corners
- Deep layers: complex patterns

### 2. Pooling
- **Max Pooling**: Takes maximum value in region
- Reduces spatial dimensions
- Provides translation invariance
- Reduces computation

### 3. Stride and Padding
- **Stride**: Step size of filter movement
- **Padding**: Adds border pixels
- Controls output dimensions

### 4. Feature Maps
- Output of convolutional layers
- Each filter produces one feature map
- Visualizes learned features

### 5. Receptive Field
- Region of input that affects a neuron
- Increases with deeper layers
- Captures larger context

## CNN Layer Types

### Convolutional Layer
```python
Conv2D(filters=32, kernel_size=(3,3), activation='relu')
```
- **filters**: Number of output channels
- **kernel_size**: Size of convolutional window
- **activation**: Non-linearity function

### Pooling Layer
```python
MaxPooling2D(pool_size=(2,2))
```
- Reduces spatial dimensions by half
- Retains most important features

### Batch Normalization
```python
BatchNormalization()
```
- Normalizes layer inputs
- Stabilizes training
- Allows higher learning rates

## Data Augmentation Techniques

| Technique | Range | Effect |
|-----------|-------|--------|
| Rotation | 10° | Rotates images slightly |
| Zoom | 0.1 | Zooms in/out |
| Width Shift | 0.1 | Shifts horizontally |
| Height Shift | 0.1 | Shifts vertically |

Benefits:
- Increases effective dataset size
- Improves generalization
- Reduces overfitting
- Makes model robust to variations

## When to Use CNN

### Perfect For:
- Image classification
- Object detection
- Facial recognition
- Medical image analysis
- Satellite imagery
- Video analysis
- Any spatial/grid data

### Not Ideal For:
- Sequential/time-series data (use RNN/LSTM)
- Tabular data (use ANN or traditional ML)
- Text data (use RNN/Transformer)
- Small datasets (use transfer learning)

## Advantages

1. **Automatic Feature Learning**: No manual feature engineering
2. **Parameter Sharing**: Fewer parameters than fully connected
3. **Translation Invariance**: Detects patterns anywhere in image
4. **Hierarchical Learning**: Learns from simple to complex features
5. **State-of-the-Art**: Best performance for computer vision

## Limitations

1. **Data Hungry**: Requires large labeled datasets
2. **Computational**: Needs GPU for efficient training
3. **Black Box**: Difficult to interpret decisions
4. **Memory**: Large models need significant memory
5. **Overfitting**: Can memorize training data

## Best Practices

1. **Use Data Augmentation**: Improves generalization
2. **Batch Normalization**: Stabilizes training
3. **Dropout**: Prevents overfitting
4. **Start Simple**: Gradually increase complexity
5. **Use Pretrained Models**: Transfer learning for small datasets
6. **Monitor Validation**: Watch for overfitting
7. **Learning Rate Scheduling**: Adjust during training

## CNN vs ANN

| Feature | CNN | ANN |
|---------|-----|-----|
| Best For | Images | Tabular data |
| Parameters | Fewer (shared) | More |
| Spatial Awareness | Yes | No |
| Feature Learning | Automatic | Manual |
| Translation Invariance | Yes | No |
| Computation | Higher | Lower |

## Common Architectures

### Classic CNNs:
- **LeNet-5**: First successful CNN (1998)
- **AlexNet**: ImageNet winner (2012)
- **VGG**: Deep uniform architecture (2014)
- **ResNet**: Residual connections (2015)
- **Inception**: Multi-scale filters (2014)

### Modern CNNs:
- **EfficientNet**: Compound scaling
- **MobileNet**: Lightweight for mobile
- **DenseNet**: Dense connections

## Transfer Learning

Using pretrained models:

```python
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False)
# Add custom layers on top
```

Benefits:
- Less training data needed
- Faster training
- Better performance

## Visualization Techniques

1. **Filter Visualization**: See what filters learned
2. **Feature Maps**: Activation patterns
3. **Grad-CAM**: Class activation mapping
4. **t-SNE**: Visualize learned representations

## Common Issues and Solutions

### Problem: Overfitting
**Solutions**:
- Add more data augmentation
- Increase dropout rate
- Use batch normalization
- Early stopping
- Reduce model complexity

### Problem: Underfitting
**Solutions**:
- Increase model depth
- Add more filters
- Train longer
- Reduce dropout
- Add more layers

### Problem: Slow Training
**Solutions**:
- Use GPU acceleration
- Reduce image size
- Decrease batch size
- Use simpler architecture

### Problem: Poor Accuracy
**Solutions**:
- More training data
- Data augmentation
- Pretrained models (transfer learning)
- Tune hyperparameters
- Ensure proper normalization

## File Structure

```
CNN/
├── CNN.ipynb              # Main Jupyter notebook
├── README.md              # This file
└── cnn_mnist_model.h5     # Saved model (generated)
```

## Future Improvements

- [ ] Implement ResNet architecture
- [ ] Add transfer learning examples
- [ ] Try different datasets (CIFAR-10, ImageNet)
- [ ] Implement object detection
- [ ] Add Grad-CAM visualization
- [ ] Experiment with different optimizers
- [ ] Implement custom CNN architectures

## Learning Resources

### Documentation
- [TensorFlow CNN Guide](https://www.tensorflow.org/tutorials/images/cnn)
- [Keras Conv2D Documentation](https://keras.io/api/layers/convolution_layers/)

### Papers
- [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Very Deep CNNs (VGG)](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)

### Tutorials
- [Stanford CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Videos
- [3Blue1Brown: CNNs](https://www.youtube.com/watch?v=aircAruvnKk)
- [Andrej Karpathy: CNNs](https://www.youtube.com/watch?v=u6aEYuemt0M)

## Contributing

Contributions welcome! Feel free to:
- Add new architectures
- Improve documentation
- Add more datasets
- Optimize performance

## License

Part of the Machine Learning repository - MIT License

## Author

Pravin Kumar S

## Acknowledgments

- TensorFlow and Keras teams
- MNIST dataset creators
- Computer vision research community
