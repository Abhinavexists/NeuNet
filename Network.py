import sys
import numpy as np
import matplotlib

inputs = [1.2,3.4,5.9] #output numbers treated as input for upcoming layers
weights = [3.9 , 7.6 , 4.3] # weights of each input (random number)
bias = 3 #unique bias

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)