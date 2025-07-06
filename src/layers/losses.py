import numpy as np

class BaseLoss:
    def calculate(self, output, y, layer=None):
        sample_losses = self.forward(output, y, layer)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def forward(self, y_pred, y_true, layer=None):
        raise NotImplementedError
    
    def backward(self, y_pred, y_true, layer=None):
        raise NotImplementedError

class CategoricalCrossentropy(BaseLoss):
    def __init__(self, regularization_l2=0.0, regularization_l1=0.0):
        self.regularization_l2 = regularization_l2
        self.regularization_l1 = regularization_l1

    def forward(self, y_pred, y_true, layer=None):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidence)
        data_loss = np.mean(negative_log_likelihood)
         
        regularization_loss = 0
        if layer is not None:
            if self.regularization_l2 > 0:
                regularization_loss += self.regularization_l2 * np.sum(layer.weights**2)
                regularization_loss += self.regularization_l2 * np.sum(layer.biases**2)

            if self.regularization_l1 > 0:
                regularization_loss += self.regularization_l1 * np.sum(np.abs(layer.weights))
                regularization_loss += self.regularization_l1 * np.sum(np.abs(layer.biases))
                
        return data_loss + regularization_loss

    def backward(self, y_pred, y_true, layer=None):
        sample_size = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            self.dinputs = np.zeros_like(y_pred)
            self.dinputs[range(sample_size), y_true] = -1 / y_pred_clipped[range(sample_size), y_true]
            self.dinputs = self.dinputs / sample_size 
        elif len(y_true.shape) == 2:
            self.dinputs = -y_true / y_pred_clipped
            self.dinputs = self.dinputs / sample_size 

        if layer is not None:
            if not hasattr(layer, 'dweight'):
                layer.dweight = np.zeros_like(layer.weights)
            if not hasattr(layer, 'dbiases'):
                layer.dbiases = np.zeros_like(layer.biases)
                
            if self.regularization_l2 > 0:
                layer.dweight += 2 * self.regularization_l2 * layer.weights
                layer.dbiases += 2 * self.regularization_l2 * layer.biases
            
            if self.regularization_l1 > 0:
                layer.dweight += self.regularization_l1 * np.sign(layer.weights)
                layer.dbiases += self.regularization_l1 * np.sign(layer.biases)

        return self.dinputs