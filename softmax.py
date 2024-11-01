import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2], 
                 [1.41, 1.051, 0.020]]

exp_values = np.exp(layer_outputs)

print(np.sum(layer_outputs , axis=1 , keepdims=True))

norm_value = exp_values / np.sum(exp_values , axis = 1 , keepdims=True)

print(norm_value)
