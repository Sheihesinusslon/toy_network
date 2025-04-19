import numpy as np


class Metrics:
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray): ...

    def __repr__(self): ...


class Accuracy(Metrics):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Simple metric for classification tasks"""
        y_pred = np.round(y_pred)
        correct_predictions = np.sum(y_pred == y_true)
        total_predictions = y_true.size
        return correct_predictions / total_predictions

    def __repr__(self):
        return "Accuracy"


class R2(Metrics):
    def __call__(self, y_pred, y_true):
        """R-squared metric for regression tasks"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def __repr__(self):
        return "R-squared"
