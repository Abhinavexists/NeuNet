# NeuNet

<div align="center">

![NeuNet Logo](https://img.shields.io/badge/NeuNet-Neural%20Network%20from%20Scratch-blue?style=for-the-badge)

<**A complete neural network built from scratch with NumPy**>

[![GitHub Pages](https://img.shields.io/badge/Documentation-Live-brightgreen?style=flat-square)](https://abhinavexists.github.io/NeuNet/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)](https://python.org)
[![NumPy](https://img.shields.io/badge/Built%20with-NumPy-orange?style=flat-square)](https://numpy.org)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[Documentation](https://abhinavexists.github.io/NeuNet/) • [Quick Start](#quick-start) • [Features](#features)

</div>

## Overview

NeuNet is a comprehensive neural network implemented entirely from scratch using NumPy. This project demonstrates the fundamental concepts of deep learning.

## Quick Start

### Installation

```bash
git clone https://github.com/Abhinavexists/NeuNet.git
cd NeuNet
pip install numpy matplotlib plotly networkx pandas
```

### Basic Usage

```python
from src.models.neural_network import NeuralNetwork
from src.layers.core import Dense
from src.layers.activations import ReLU, Softmax
from src.layers.regularization import BatchNormalization, Dropout
from src.layers.losses import CategoricalCrossentropy
from src.layers.dataset import create_data

# Create synthetic dataset
X, Y = create_data(samples=100, classes=3, plot=True)

# Build the network
model = NeuralNetwork()
model.add(Dense(2, 128, learning_rate=0.002, optimizer='adam'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.1))

model.add(Dense(128, 64, learning_rate=0.002, optimizer='adam'))
model.add(ReLU())

model.add(Dense(64, 3, learning_rate=0.002, optimizer='adam'))
model.add(Softmax())

# Set loss function with regularization
model.set_loss(CategoricalCrossentropy(regularization_l2=0.0001))

# Train with advanced features
model.train(X, Y, epochs=500, batch_size=32, patience=30, verbose=True)

# Evaluate
predictions = model.predict(X)
accuracy = model.predict_proba(X)
print(f"Final Accuracy: {accuracy:.4f}")
```

## Features

### Core Components

- **Dense Layers**: Fully connected layers with He initialization
- **Activation Functions**: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: Categorical crossentropy with L1/L2 regularization
- **Optimizers**: SGD with momentum, Adam optimizer with bias correction
- **Regularization**: Batch normalization, dropout, weight penalties

### Performance

- **High Accuracy**: 95%+ on synthetic datasets
- **Fast Training**: 3-5x speed improvement with Adam optimizer
- **Robust**: Early stopping prevents overfitting
- **Stable**: Numerical stability built into all operations

## Documentation

Our comprehensive documentation is available at **[abhinavexists.github.io/NeuNet](https://abhinavexists.github.io/NeuNet/)**

### Documentation Sections

- **[Home](https://abhinavexists.github.io/NeuNet/)**: Project overview and quick navigation
- **[Implementation Guide](https://abhinavexists.github.io/NeuNet/implementation-guide.html)**: Detailed technical walkthrough
- **[Development Timeline](https://abhinavexists.github.io/NeuNet/development-timeline.html)**: Evolution of the project
- **[Reference](https://abhinavexists.github.io/NeuNet/reference.html)**: Complete component documentation

## Advanced Examples

### Multi-layer Classification Network

```python
# Sophisticated architecture with all features
model = NeuralNetwork()

# Input processing with normalization
model.add(Dense(2, 128, learning_rate=0.002, optimizer='adam'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.1))

# Deep feature extraction
model.add(Dense(128, 64, learning_rate=0.002, optimizer='adam'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.1))

# Final classification
model.add(Dense(64, 32, learning_rate=0.002, optimizer='adam'))
model.add(ReLU())
model.add(Dense(32, 3, learning_rate=0.002, optimizer='adam'))
model.add(Softmax())

# Advanced loss with regularization
model.set_loss(CategoricalCrossentropy(regularization_l2=0.0001))

# Professional training setup
model.train(X, Y, epochs=500, batch_size=32, patience=30, verbose=True)
```

### Comprehensive Evaluation

```python
from src.utils.metrics import calculate_accuracy, confusion_matrix, precision_recall_f1

# Get predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Calculate metrics
accuracy = calculate_accuracy(y_true, probabilities)
cm = confusion_matrix(y_true, predictions, num_classes=3)
precision, recall, f1 = precision_recall_f1(y_true, predictions, num_classes=3)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
```

### Interactive Visualization

```python
from src.utils.network_data import export_network
from src.utils.Visualization import network_visualization

# Export network structure
dense_layers = [layer for layer in model.layers if hasattr(layer, 'weights')]
export_network(*dense_layers[:4])

# Create interactive visualization
fig = network_visualization("src/utils/network_data.json")
# Opens interactive HTML visualization
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Classification Accuracy** | 95%+ |
| **Training Speed (vs SGD)** | 3-5x faster |
| **Components** | 15+ core modules |
| **Activation Functions** | 5 implemented |
| **Optimization Algorithms** | 2 advanced |
| **Regularization Techniques** | 3 methods |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
