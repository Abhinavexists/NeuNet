import numpy as np
import matplotlib.pyplot as plt

# The create_data function generates a synthetic dataset with data points arranged in a spiral pattern,
# separated into multiple classes. This can be useful for testing classification algorithms because the
# spiral pattern creates a challenging, non-linear classification problem.
np.random.seed(0)

def create_data(samples=50 , classes=3):
    X = np.zeros((samples*classes , 2))
    Y = np.zeros(samples*classes , dtype = 'uint8')
    for class_number in range(classes):
        ix = range(samples*class_number , samples*(class_number+1))
        r = np.linspace(0.0,1 , samples) # radius
        t = np.linspace(class_number*4 , (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5) , r*np.cos(t*2.5)]
        Y[ix] = class_number
    return X ,Y

print("here")


