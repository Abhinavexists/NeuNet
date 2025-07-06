import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.neural_network import NeuralNetwork
from src.layers.core import Dense
from src.layers.activations import ReLU, Softmax
from src.layers.regularization import BatchNormalization, Dropout
from src.layers.losses import CategoricalCrossentropy
from src.layers.dataset import create_data, plot_decision_boundary
from src.utils.network_data import export_network
from src.utils.metrics import calculate_accuracy

def main():
    X, Y = create_data(samples=100, classes=3, noise=0.2)
    
    model = NeuralNetwork()
    
    model.add(Dense(2, 128, learning_rate=0.002, optimizer='adam'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    
    model.add(Dense(128, 64, learning_rate=0.002, optimizer='adam'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    
    model.add(Dense(64, 32, learning_rate=0.002, optimizer='adam'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    
    model.add(Dense(32, 3, learning_rate=0.002, optimizer='adam'))
    model.add(Softmax())
    
    model.set_loss(CategoricalCrossentropy(regularization_l2=0.0001))
    
    print("Training on spiral dataset")
    epochs = int(input("Enter the number of epochs (recommended 500+): "))
    batch_size = int(input("Enter the batch size (recommended 16-32, or 0 for full batch): "))
    
    model.train(X, Y, epochs=epochs, batch_size=batch_size)
    
    predictions = model.predict_proba(X)
    accuracy = calculate_accuracy(Y, predictions)
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    
    dense_layers = [layer for layer in model.layers if hasattr(layer, 'weights')]
    if len(dense_layers) >= 4:
        export_network(*dense_layers[:4])
        print("Network exported successfully!")

    try:
        plot_decision_boundary(model.predict, X, Y)
    except Exception as e:
        print(f"Could not plot decision boundary: {e}")

if __name__ == "__main__":
    main()