import sys
import numpy as np
from dataset import create_data
from plot import scatter_plot

np.random.seed(0)
class Layer_Dense:
    def __init__(self , n_inputs , n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1 , n_neurons))
    def forward(self , inputs):
        self.output = np.dot(inputs ,self.weights) + self.biases

class Activation_ReLU:
    def forward(self , inputs):
        self.output = np.maximum(0 , inputs)

class Activation_softmax:
    def forward(self ,inputs):
        exp_values = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))
        probabilities = exp_values / np.sum(exp_values , axis=1 , keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output , y):
        sample_losses = self.forward(output , y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_Categoricalcrossentropy(Loss):
    def forward(self , y_pred , y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred , 1e-7 , 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample) , y_true]
        
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_true , axis = 1)

        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood
        
X,Y = create_data(samples=100, classes=3)

hidden_layer1 = Layer_Dense(2, 8) # 2 input features and 3 neurons
activation1 = Activation_ReLU()

hidden_layer2 = Layer_Dense(8, 6) # with 3 input features (from the previous layer) and 64 output neurons.
activation2 = Activation_ReLU()

hidden_layer3 = Layer_Dense(6, 3)
activation3 = Activation_softmax()

hidden_layer1.forward(X)
activation1.forward(hidden_layer1.output)

hidden_layer2.forward(activation1.output)
activation2.forward(hidden_layer2.output)

hidden_layer3.forward(activation2.output)
activation3.forward(hidden_layer3.output)

print(activation3.output[:5])

Loss_function = Loss_Categoricalcrossentropy()
loss = Loss_function.calculate(activation3.output , Y)

print("Loss" , loss)

