import numpy as np

from src.tensor import Tensor


class Loss:
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor: ...

    def derivative(self, y_pred: Tensor, y_true: Tensor) -> Tensor: ...

    def __repr__(self):
        return self.__class__.__name__


class MSE(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return ((y_pred - y_true) ** 2).mean()

    def derivative(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return (y_pred - y_true) * (2 / y_pred.value.size)


class BinaryCrossEntropy(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred.value, eps, 1 - eps)
        term1 = y_true.value * np.log(y_pred_clipped)
        term2 = (1 - y_true.value) * np.log(1 - y_pred_clipped)
        return Tensor(-np.mean(term1 + term2))

    def derivative(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return Tensor(y_pred.value - y_true.value)
