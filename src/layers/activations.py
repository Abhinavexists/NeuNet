import numpy as np

class BaseActivation:
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, dvalues):
        raise NotImplementedError

class ReLU(BaseActivation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class LeakyReLU(BaseActivation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, inputs * self.alpha)
        return self.output
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= self.alpha
        return self.dinputs

class Tanh(BaseActivation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output
        
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - np.square(self.output))
        return self.dinputs

class Sigmoid(BaseActivation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500))) 
        return self.output
        
    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)
        return self.dinputs

class Softmax(BaseActivation):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        batch_size = len(dvalues)        
        self.dinputs = np.zeros_like(dvalues)
        
        for i in range(batch_size):
            output_single = self.output[i].reshape(-1, 1)
            jacobian_matrix = output_single * (np.eye(len(output_single)) - output_single.T)
            self.dinputs[i] = np.dot(jacobian_matrix, dvalues[i])
        
        return self.dinputs