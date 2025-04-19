from src.tensor import Tensor


class Loss:
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor: ...

    def derivative(self, y_pred: Tensor, y_true: Tensor) -> Tensor: ...


class MSE(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return ((y_pred - y_true) ** 2).mean()

    def derivative(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return (y_pred - y_true) * (2 / y_pred.value.size)
