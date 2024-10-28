import sys
import numpy as np
import matplotlib

inputs = [1.2 , 3.4 , 5.9 , 3.6] #output numbers treated as input for upcoming layers
weights = [[3.9 , 7.6 , 4.3 , 1.0], # weights of each input (random number)
          [3.6 , 7.0 , 5.0 , 1.7],
          [6.9 , 6.6 , 4.9 , 1.8]]

biases = [3,2,1] #unique bias
# weights = [3.9 , 7.6 , 4.3 , 1.0]
# bias = 2

output = np.dot(weights , inputs) + biases
print(output)

# layer_outputs = []
# for neuron_weights , neuron_bias in zip(weights , biases): # zip combines 2 list into a list of list
#   neuron_output = 0
#   for n_input , weight in zip(inputs , neuron_weights):
#     neuron_output += n_input*weight
#   neuron_output += neuron_bias
#   layer_outputs.append(neuron_output)

# print(layer_outputs)