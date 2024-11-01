import numpy as np


# The create_data function generates a synthetic dataset with data points arranged in a spiral pattern,
# separated into multiple classes. This can be useful for testing classification algorithms because the
# spiral pattern creates a challenging, non-linear classification problem.
np.random.seed(0)

def create_data(points , classes):
    X = np.zeros((points*classes , 2))
    Y = np.zeros(points*classes , dtype = 'uint8')
    for class_number in range(classes):
        ix = range(points*class_number , points*(class_number+1))
        r = np.linspace(0.0,1 , points) # radius
        t = np.linspace(class_number*4 , (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5) , r*np.cos(t*2.5)]
        Y[ix] = class_number
    return X ,Y

import matplotlib.pyplot as plt

print("here")
X ,Y = create_data(100,3) # 3 classes of each feature sets each

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="plasma", edgecolor="k", s=40)
plt.title("Spiral Dataset")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.colorbar(label="Class Label")
plt.show()