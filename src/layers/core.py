import numpy as np
from .optimisers import Optimizer_SGD_Momentum, Optimizer_Adam

class BaseLayer:
    """Base class for all layers"""
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, dvalues):
        raise NotImplementedError

class Dense(BaseLayer):
    def __init__(self, n_inputs, n_neurons, learning_rate=0.01, decay_rate=0.02, momentum=0.9, optimizer=None):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))

        self.dweight = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate 
        self.initial_learning_rate = learning_rate 
        self.momentum = momentum

        if optimizer == 'adam':
            self.optimiser = Optimizer_Adam(learning_rate=learning_rate)
        else:
            self.optimiser = Optimizer_SGD_Momentum(
                learning_rate=learning_rate,
                momentum=self.momentum
            )

    def forward(self, inputs):
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues, epoch): 
        self.learning_rate = self.initial_learning_rate * np.exp(-self.decay_rate * epoch)
        self.optimiser.learning_rate = self.learning_rate
        self.dweight = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) 
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.optimiser.update_params(self)

        return self.dinputs