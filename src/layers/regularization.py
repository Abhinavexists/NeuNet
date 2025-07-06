import numpy as np

class BaseRegularization:
    def forward(self, inputs, training=True):
        raise NotImplementedError
    
    def backward(self, dvalues):
        raise NotImplementedError

class BatchNormalization(BaseRegularization):
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon 
        self.momentum = momentum 
        self.running_mean = None
        self.running_var = None
        self.gamma = None 
        self.beta = None  
        self.dgamma = None
        self.dbeta = None

    def forward(self, inputs, training=True):
        self.inputs = inputs
        input_shape = inputs.shape
        
        if self.gamma is None:
            self.gamma = np.ones(input_shape[1])
            self.beta = np.zeros(input_shape[1])
            self.running_mean = np.zeros(input_shape[1])
            self.running_var = np.ones(input_shape[1])
        
        if training:
            mean = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0)
            
            if self.running_mean is None:
                self.running_mean = mean
                self.running_var = var
            else:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            self.x_centered = inputs - mean
            self.std = np.sqrt(var + self.epsilon)
            self.x_norm = self.x_centered / self.std
        else:
            self.x_norm = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        self.output = self.gamma * self.x_norm + self.beta
        
        return self.output

    def backward(self, dvalues):
        batch_size = dvalues.shape[0]
        
        self.dgamma = np.sum(dvalues * self.x_norm, axis=0)
        self.dbeta = np.sum(dvalues, axis=0)
        
        dx_norm = dvalues * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * self.std**(-3), axis=0)
        dmean = np.sum(dx_norm * -1 / self.std, axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)
        self.dinputs = dx_norm / self.std + dvar * 2 * self.x_centered / batch_size + dmean / batch_size
        
        return self.dinputs

class Dropout(BaseRegularization):
    def __init__(self, rate):
        self.rate = rate
        self.binary_mask = None
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        
        if not training:
            self.output = inputs
            return self.output
            
        self.binary_mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
        
        self.output = inputs * self.binary_mask
        return self.output
        
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
        return self.dinputs