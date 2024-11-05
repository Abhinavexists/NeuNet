import sys
import numpy as np
from dataset import create_data  # Import a function to create sample data
from plot import scatter_plot  # Import a function to visualize data (optional)

# Set a random seed for reproducibility
np.random.seed(0)

# Define a dense (fully-connected) layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with a small random value and biases as zeros
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Perform a forward pass: calculate the output as the dot product of inputs and weights, plus biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Define the ReLU activation function class
class Activation_ReLU:
    def forward(self, inputs):
        # Apply ReLU (Rectified Linear Unit): max(0, x)
        self.output = np.maximum(0, inputs)

# Define the softmax activation function class
class Activation_softmax:
    def forward(self, inputs):
        # Calculate exponentiated values to prevent overflow, then normalize to obtain probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Base class for calculating loss
class Loss:
    def calculate(self, output, y):
        # Calculate the mean of losses for each sample in the batch
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Define the categorical cross-entropy loss class, inheriting from Loss
class Loss_Categoricalcrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples in the batch
        sample = len(y_pred)
        # Clip predictions to avoid log(0) errors
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # For labels in integer form
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample), y_true]

        # For one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        # Calculate the negative log likelihood
        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood

# Generate a dataset with 100 samples and 3 classes
X, Y = create_data(samples=100, classes=3)

# Initialize layers and activation functions
hidden_layer1 = Layer_Dense(2, 8)  # First hidden layer: 2 input features, 8 neurons
activation1 = Activation_ReLU()

hidden_layer2 = Layer_Dense(8, 6)  # Second hidden layer: 8 inputs (from previous layer), 6 neurons
activation2 = Activation_ReLU()

hidden_layer3 = Layer_Dense(6, 3)  # Output layer: 6 inputs, 3 output classes
activation3 = Activation_softmax()

# Perform forward pass through each layer and activation function
hidden_layer1.forward(X)
activation1.forward(hidden_layer1.output)

hidden_layer2.forward(activation1.output)
activation2.forward(hidden_layer2.output)

hidden_layer3.forward(activation2.output)
activation3.forward(hidden_layer3.output)

# Display the output of the final layer (softmax probabilities) for the first 5 samples
print(activation3.output[:5])

# Calculate the loss using categorical cross-entropy
Loss_function = Loss_Categoricalcrossentropy()
loss = Loss_function.calculate(activation3.output, Y)

# Print the final loss value
print("Loss", loss)
