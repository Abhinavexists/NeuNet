import numpy as np
import matplotlib.pyplot as plt

def create_data(samples=100, classes=3, plot=True):
    """
    Create a synthetic dataset with extremely well-separated clusters for easier classification.
    
    Args:
        samples: Number of data points to generate
        classes: Number of classes to generate
        plot: Whether to visualize the data
        
    Returns:
        X: Feature data, shape (samples, 2)
        y: Labels, shape (samples,)
    """
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Generate synthetic data
    X = np.zeros((samples, 2))
    y = np.zeros(samples, dtype=np.int32)
    
    # Create extremely well-separated clusters for each class
    points_per_class = samples // classes
    centers = [
        [-5, -5],  # Class 0
        [0, 5],    # Class 1
        [5, 0]     # Class 2
    ]
    
    for class_number in range(classes):
        # Get center for this class
        center_x, center_y = centers[class_number]
        
        # Create tight clusters around centers
        ix = range(points_per_class * class_number, points_per_class * (class_number + 1))
        X[ix] = np.random.randn(points_per_class, 2) * 0.3 + np.array([center_x, center_y])
        y[ix] = class_number
    
    # Visualize the data if requested
    if plot:
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(label='Class')
        plt.title('Synthetic Classification Dataset - Highly Separable')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('scatter_plot.png')  # Save the plot
        print("Scatter Plot")
    
    return X, y 