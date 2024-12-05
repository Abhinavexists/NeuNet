# Neural Network From Scratch

## Overview
This project implements a basic neural network from scratch using NumPy. It currently includes a spiral dataset generator and a simple feedforward and Backpropogation with gradient decent neural network with dense layers, ReLU, and Softmax activation functions.

## Current Features
- Synthetic spiral dataset generation for multi-class classification
- Dense (fully connected) layer implementation
- ReLU activation function
- Softmax activation function for output layer
- Basic visualization tools for dataset plotting

## Project Structure
```
.
├── dataset.py      # Generates synthetic spiral dataset
├── Network.py      # Neural network implementation
├── plot.py         # Visualization utilities
├── softmax.py      # Softmax activation example
├── CONTRIBUTING.md # Contribution guidelines
├── LICENSE         # MIT License
└── requirements.txt
```

### Components
- **dataset.py**: Creates a synthetic spiral dataset with configurable number of classes and samples per class. Uses NumPy for data generation and includes random noise for more realistic data.

- **Network.py**: Contains the core neural network implementation with:
  - `Layer_Dense`: Implements fully connected layers
  - `Activation_ReLU`: Implements ReLU activation function
  - `Activation_softmax`: Implements Softmax activation for classification

- **plot.py**: Provides visualization functionality for the spiral dataset using matplotlib.

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Currently, you can run the network with:
```python
python Network.py
```

This will:
1. Generate a spiral dataset
2. Create a neural network with two dense layers
3. Process the data through the network
4. Output probabilities for the first 5 samples

To visualize the dataset:
```python
python plot.py
```

## Work in Progress
This project is actively under development. Planned features include:
- Loss function implementation
- Backpropagation
- Training loop
- Model evaluation metrics
- Model saving/loading functionality
- Additional activation functions
- Documentation improvements
- Testing suite

## Contributing
We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to:
- Submit issues
- Submit pull requests
- Follow coding standards
- Contribute to documentation

## Dependencies
- NumPy
- Matplotlib

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Recent Updates in `Network.py`
### Dynamic Learning Rate Decay
The network now dynamically adjusts the learning rate using an exponential decay mechanism based on the epoch count.

### Categorical Cross-Entropy Loss
A complete implementation of the categorical cross-entropy loss function is included, with both forward and backward passes to compute gradients.

### Accuracy Calculation
An accuracy calculation method compares predictions with true labels and outputs the accuracy score.

## Training Process
When you run `Network.py`, the following steps are performed:
1. A synthetic dataset is generated using `dataset.py`.
2. The neural network, with two dense layers and ReLU activation, processes the data.
3. Training is performed for a user-specified number of epochs, with dynamic learning rate decay applied.
4. The model calculates and displays:
   - Softmax probabilities of the first 5 samples.
   - The final loss value.
   - The overall accuracy.

### Example Usage
Run the training script:
```bash
python Network.py
```

Expected output includes the loss at intervals and final metrics like probabilities, loss, and accuracy.
