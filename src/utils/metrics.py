import numpy as np

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy for predictions"""
    predictions = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    return np.mean(predictions == y_true)

def confusion_matrix(y_true, y_pred, num_classes):
    """Calculate confusion matrix"""
    predictions = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    cm = np.zeros((num_classes, num_classes))
    for true, pred in zip(y_true, predictions):
        cm[true][pred] += 1
    return cm

def precision_recall_f1(y_true, y_pred, num_classes):
    """Calculate precision, recall, and F1 score"""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return precision, recall, f1
