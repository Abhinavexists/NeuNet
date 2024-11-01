import sys
import numpy as np
import matplotlib
from dataset import create_data

np.random.seed(0)

X = [[1.2 , 3.4 , 5.9 , 3.6], #output numbers treated as input for upcoming layers
          [2.0 , 5.0 , 4.1 , 2.0],
          [-1.5 , 2.7 , 3.3 , -0.8]]

X ,Y = create_data(100 , 3)
print(create_data)

class Layer_Dense:
    def __init__(self , n_inputs , n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1 , n_neurons))
    def forward(self , inputs):
        self.output = np.dot(inputs ,self.weights) + self.biases

class Activation_ReLU:
    def forward(self , inputs):
        self.output = np.maximum(0 , inputs)


layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(5,2)
activation2 = Activation_ReLU()

layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
activation2.forward(layer2.output)

# weights = [[3.9 , 7.6 , 4.3 , 1.0], # weights of each input (random number)
#           [3.6 , 7.0 , 5.0 , 1.7],
#           [6.9 , 6.6 , 4.9 , 1.8]]


# biases = [3,2,1] #unique bias

# weights2 = [[0,1 , -0.14 , 0.5], # weights of each input (random number)
#           [-0.5 , 0.12 , -0.33],
#           [-0.44 , 0.73 , -0.13 ]]


# biases2 = [-1,2,-0.5] #unique bias
# # weights = [3.9 , 7.6 , 4.3 , 1.0]
# # bias = 2

# layer1_output = np.dot(inputs , np.array(weights).T) + biases # dot product

# layer2_output = np.dot(layer1_output , np.array(weights2).T) + biases2
# print(layer2_output)

# layer_outputs = []
# for neuron_weights , neuron_bias in zip(weights , biases): # zip combines 2 list into a list of list
#   neuron_output = 0
#   for n_input , weight in zip(inputs , neuron_weights):
#     neuron_output += n_input*weight
#   neuron_output += neuron_bias
#   layer_outputs.append(neuron_output)

# print(layer_outputs)