import numpy as np
from dataset import create_data  # Import a function to create sample data
from plot import scatter_plot  # Import a function to visualize data (optional)

# Set a random seed for reproducibility
np.random.seed(0)

# Define a dense (fully-connected) layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, learning_rate=0.01, decay_rate=0.01):
        # Initialize weights with a small random value and biases as zeros
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.initial_learning_rate = learning_rate  # Store the initial learning rate

    def forward(self, inputs):
        # Perform a forward pass
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, epoch):
        # Update the learning rate dynamically
        self.learning_rate = self.initial_learning_rate * np.exp(-self.decay_rate * epoch)

        # Gradient on Parameters (Calculate gradients)
        self.dweight = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)  # keepdims=1 to match bias shape

        # Update weights and biases using decayed learning rate
        self.weights -= self.learning_rate * self.dweight
        self.biases -= self.learning_rate * self.dbiases

# Define the ReLU activation function class
class Activation_ReLU:
    def forward(self, inputs):
        # Apply ReLU (Rectified Linear Unit)
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative or zero
        self.dinputs[self.inputs <= 0] = 0

# Define the softmax activation function class
class Activation_softmax:
    def forward(self, inputs):
        # Calculate exponentiated values to prevent overflow, then normalize
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Number of samples
        batch_size = dvalues.shape[0]

        # Gradient of softmax output
        # Using the softmax gradient formula
        self.dinputs = self.output.copy()  # shape (batch_size, n_classes)
        for i in range(batch_size):
            # Jacobian matrix for softmax
            jacobian_matrix = np.diag(self.output[i]) - np.dot(self.output[i].reshape(-1, 1), self.output[i].reshape(1, -1))
            self.dinputs[i] = np.dot(jacobian_matrix, dvalues[i])  # Applying the gradient to the Jacobian

        return self.dinputs

# Base class for calculating loss
class Loss:
    def calculate(self, output, y):
        # Calculate the mean of losses for each sample in the batch
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Define the categorical cross-entropy loss class
class Loss_Categoricalcrossentropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood

    def backward(self, y_pred, y_true):
        sample_size = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            self.dinputs = -1 / y_pred_clipped[range(sample_size), y_true]
            self.dinputs = self.dinputs / sample_size  # Average the gradients
        elif len(y_true.shape) == 2:
            self.dinputs = -y_true / y_pred_clipped
            self.dinputs = self.dinputs / sample_size  # Average the gradients

        return self.dinputs

# Generate a dataset with 100 samples and 3 classes
X, Y = create_data(samples=100, classes=3)

# Initialize layers and activation functions
hidden_layer1 = Layer_Dense(2, 8, learning_rate=0.02)  # First hidden layer
activation1 = Activation_ReLU()

hidden_layer2 = Layer_Dense(8, 6, learning_rate=0.01)  # Second hidden layer
activation2 = Activation_ReLU()

output_layer = Layer_Dense(6, 3, learning_rate=0.02)  # Output layer
activation3 = Activation_softmax()

# Calculate the loss using categorical cross-entropy
Loss_function = Loss_Categoricalcrossentropy()

# User inputs for epochs
epochs = int(input("Enter the number of epochs: "))

# Training loop
for epoch in range(epochs):
    # Forward pass through layers
    hidden_layer1.forward(X)
    activation1.forward(hidden_layer1.output)

    hidden_layer2.forward(activation1.output)
    activation2.forward(hidden_layer2.output)

    output_layer.forward(activation2.output)
    activation3.forward(output_layer.output)

    loss_value = Loss_function.calculate(activation3.output, Y)  # Get the float loss value
    loss_gradient = Loss_function.backward(activation3.output, Y)  # Backward pass for loss
    activation3.backward(loss_gradient)  # Backward pass for the output layer

    output_layer.backward(activation3.dinputs, epoch)  # Backward pass for the output layer
    activation2.backward(output_layer.dinputs)  # Backward pass for second hidden layer
    hidden_layer2.backward(activation2.dinputs)  # Backward pass for first hidden layer
    activation1.backward(hidden_layer2.dinputs)  # Backward pass for the input layer
    hidden_layer1.backward(activation1.dinputs)  # Backward pass for the input layer

# Display the output of the final layer (softmax probabilities) for the first 5 samples
print(activation3.output[:5])

loss = Loss_function.calculate(activation3.output, Y)

# Print the final loss value
print("Loss", loss)
