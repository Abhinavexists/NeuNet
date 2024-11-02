# Neural Network Implementation Project

## Overview
This project implements a basic neural network from scratch using NumPy. It currently includes a spiral dataset generator and a simple feedforward neural network with dense layers, ReLU, and Softmax activation functions.

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
