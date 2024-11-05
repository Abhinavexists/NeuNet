import matplotlib.pyplot as plt
from dataset import create_data

X, Y = create_data(100,3)

def scatter_plot():
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="plasma", edgecolor="k", s=40)
    plt.title("Spiral Dataset")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.colorbar(label="Class Label")
    plt.show()