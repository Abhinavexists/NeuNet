import sys
import numpy as np
import matplotlib

inputs = [1.2 , 3.4 , 5.9 , 3.6] #output numbers treated as input for upcoming layers
weights1 = [3.9 , 7.6 , 4.3 , 1.0] # weights of each input (random number)
weights2 = [3.6 , 7.0 , 5.0 , 1.7]
weights3 = [6.9 , 6.6 , 4.9 , 1.8]
bias1 = 3 #unique bias
bias2 = 2
bias3 = 1

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1, # list of output layer 
inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)

