import numpy as np
from typing import Sequence

from src.activation import ActivationFunction, ReLU
from src.losses import Loss
from src.tensor import Tensor


class Neuron:
    def __init__(self, n_inputs: int, activation: ActivationFunction):
        self.weights = Tensor(np.random.uniform(-1, 1, size=(n_inputs, 1)))
        self.bias = Tensor(np.zeros((1, 1)))
        self.activation = activation
        self.inputs: Tensor | None = None
        self.output: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        z = inputs @ self.weights + self.bias
        self.output = Tensor(self.activation(z.value))
        return self.output

    def backward(self, dvalue) -> None:
        dz = dvalue * Tensor(self.activation.derivative(self.output.value))

        self.weights.grad += self.inputs.value.T @ dz.value
        self.bias.grad += dz.sum(axis=0, keepdims=True).value

    def __repr__(self) -> str:
        return f"Neuron(weights={self.weights}, bias={self.bias}, activation={self.activation.__class__.__name__})"


class Layer:
    def __init__(
        self, n_inputs: int, n_neurons: int, activation: ActivationFunction = ReLU
    ):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return Tensor(np.hstack([out.value for out in outputs]))

    def backward(self, dloss) -> None:
        dvalues = [dloss] * len(self.neurons)
        for neuron, dvalue in zip(self.neurons, dvalues):
            neuron.backward(dvalue)

    def __repr__(self):
        return f"Layer(neurons={self.neurons})"


class NeuralNetwork:
    def __init__(
        self,
        layers: Sequence[Layer],
        loss_f: Loss,
        learning_rate: float = 0.01,
    ):
        self.layers = layers
        self.loss_f = loss_f
        self.learning_rate = learning_rate
        self._loss: float | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, dloss: Tensor) -> None:
        for layer in reversed(self.layers):
            layer.backward(dloss)

    def zero_grad(self) -> None:
        """Zero the gradients of all weights and biases in the network."""
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights.zero_grad()
                neuron.bias.zero_grad()

    def update(self) -> None:
        """Update the weights and biases of the network using gradient descent."""
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights.value -= self.learning_rate * neuron.weights.grad
                neuron.bias.value -= self.learning_rate * neuron.bias.grad

    def loss(self, y_pred: Tensor, y: Tensor) -> Tensor:
        return self.loss_f(y_pred, y)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 32,
    ) -> None:
        """Train the neural network using gradient descent."""
        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = Tensor(X[i : i + batch_size])
                y_batch = Tensor(y[i : i + batch_size])

                self.zero_grad()

                # Forward pass
                output = self.forward(X_batch)

                # Compute loss
                loss = self.loss(output, y_batch)
                epoch_loss += float(loss.value.mean())

                # Backward pass
                dloss = self.loss_f.derivative(output, y_batch)
                self.backward(dloss)

                # Update weights and biases
                self.update()

            self._loss = epoch_loss / (len(X) // batch_size)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Loss: {self._loss}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)  # Convert vector to single-row matrix

        X_tensor = Tensor(X)

        with self._no_grad():
            output = self.forward(X_tensor)

        return output.value

    def _no_grad(self):
        """Context manager to temporarily disable gradient tracking"""
        return NoGradContext()

    def __repr__(self):
        return f"NeuralNetwork(layers={self.layers})"


class NoGradContext:
    """Context manager to not keep track of Tensor children during forward pass to optimize memory load"""

    def __enter__(self):
        self.prev_state = Tensor._GRAD_ENABLED
        Tensor._GRAD_ENABLED = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tensor._GRAD_ENABLED = self.prev_state
