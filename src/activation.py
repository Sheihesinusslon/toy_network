import numpy as np


class ActivationFunction:
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

    def derivative(self, output: np.ndarray) -> np.ndarray: ...


class Identity(ActivationFunction):
    """Linear/passthrough activation"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, output: np.ndarray) -> np.ndarray:
        return np.ones_like(output)


class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, output: np.ndarray) -> np.ndarray:
        return (output > 0).astype(float)


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, output: np.ndarray) -> np.ndarray:
        return output * (1 - output)


class Tanh(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, output: np.ndarray) -> np.ndarray:
        return 1 - output**2
