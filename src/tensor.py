from typing import Callable, Set, Self, Sequence

import numpy as np

type Scalar = float | int
type Array = np.ndarray | Sequence[Scalar] | Sequence[Sequence[Scalar]]


class Tensor:
    _GRAD_ENABLED = True

    def __init__(self, value: Scalar | Array, _children=()):
        """
        Initialize a Tensor with either:
        - Scalar value (float/int)
        - Vector/matrix value (numpy array)
        """
        self.value: np.ndarray = (
            np.array(value, dtype=np.float32)
            if not isinstance(value, np.ndarray)
            else value
        )
        self.grad: np.ndarray = np.zeros_like(value, dtype=np.float32)
        self._children: Set = set(_children)
        self._backward: Callable = lambda: None
        if not Tensor._GRAD_ENABLED:
            self._children = ()

    def zero_grad(self):
        self.grad.fill(0)

    def backward(self) -> None:
        """Compute the gradient of the tensor with respect to its parents, using a topological order graph traversal."""
        topo = []
        visited = set()

        def build_topo(node: Tensor):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = np.ones_like(
            self.value, dtype=np.float32
        )  # Seed the output gradient; dL/dL = 1
        for node in reversed(topo):
            node._backward()

    def __add__(self, other) -> Self:
        other = other if isinstance(other, Tensor) else Tensor(other)
        new = Tensor(self.value + other.value, (self, other))

        def _backward():
            self.grad += 1.0 * new.grad
            other.grad += 1.0 * new.grad

        new._backward = _backward

        return new

    def __mul__(self, other) -> Self:
        other = other if isinstance(other, Tensor) else Tensor(other)
        new = Tensor(self.value * other.value, (self, other))

        def _backward():
            self_grad = other.value * new.grad
            other_grad = self.value * new.grad

            # Sum gradients over broadcasted dimensions for self
            while self_grad.ndim > self.value.ndim:
                self_grad = self_grad.sum(axis=0)
            for i, dim in enumerate(self.value.shape):
                if dim == 1:
                    self_grad = self_grad.sum(axis=i, keepdims=True)

            # Sum gradients over broadcasted dimensions for other
            while other_grad.ndim > other.value.ndim:
                other_grad = other_grad.sum(axis=0)
            for i, dim in enumerate(other.value.shape):
                if dim == 1:
                    other_grad = other_grad.sum(axis=i, keepdims=True)

            self.grad += self_grad
            other.grad += other_grad

        new._backward = _backward

        return new

    def __matmul__(self, other) -> Self:
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.value.shape[-1] != other.value.shape[0]:
            raise ValueError("Incompatible shapes for matrix multiplication")
        if 0 in self.value.shape or 0 in other.value.shape:
            raise ValueError("Cannot perform matrix multiplication with empty arrays")
        new = Tensor(self.value @ other.value, (self, other))

        def _backward():
            if len(new.grad.shape) == 1:
                # vector
                self.grad += np.outer(new.grad, other.value)
                other.grad += self.value.T @ new.grad
            else:
                # matrix
                self.grad += new.grad @ other.value.T
                other.grad += self.value.T @ new.grad

        new._backward = _backward
        return new

    def sum(self, axis=None, keepdims=False) -> Self:
        new = Tensor(np.sum(self.value, axis=axis, keepdims=keepdims), (self,))

        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.value, dtype=np.float32) * new.grad
            else:
                grad_shape = np.ones_like(self.value, dtype=np.float32)
                grad_shape[axis] = self.value.shape[axis]
                self.grad += np.broadcast_to(new.grad, self.value.shape) * grad_shape

        new._backward = _backward
        return new

    def mean(self, axis=None) -> Self:
        new = Tensor(np.mean(self.value, axis=axis), (self,))

        def _backward():
            if axis is None:
                self.grad += (
                    np.ones_like(self.value, dtype=np.float32)
                    * new.grad
                    / self.value.size
                )
            else:
                grad_shape = np.ones_like(self.value, dtype=np.float32)
                grad_shape[axis] = self.value.shape[axis]
                self.grad += (
                    np.broadcast_to(new.grad, self.value.shape)
                    * grad_shape
                    / self.value.size
                )

        new._backward = _backward
        return new

    def __pow__(self, power: Scalar) -> Self:
        if not isinstance(power, (int | float)):
            raise TypeError("Power must be a numeric value")
        new = Tensor(self.value**power, (self,))

        def _backward():
            self.grad += power * (self.value ** (power - 1)) * new.grad

        new._backward = _backward

        return new

    def __sub__(self, other) -> Self:
        return self + (-other)

    def __truediv__(self, other) -> Self:
        return self * (other**-1)

    def __neg__(self) -> Self:
        return self * -1

    def __radd__(self, other) -> Self:
        other = (
            np.array(other) if not isinstance(other, (Tensor, np.ndarray)) else other
        )
        return Tensor(other) + self

    def __rmul__(self, other) -> Self:
        other = (
            np.array(other) if not isinstance(other, (Tensor, np.ndarray)) else other
        )
        return Tensor(other) * self

    def __rmatmul__(self, other) -> Self:
        other = (
            np.array(other) if not isinstance(other, (Tensor, np.ndarray)) else other
        )
        return Tensor(other) @ self

    def __rpow__(self, other) -> Self:
        other = (
            np.array(other) if not isinstance(other, (Tensor, np.ndarray)) else other
        )
        return Tensor(other) ** self

    def __rsub__(self, other) -> Self:
        other = (
            np.array(other) if not isinstance(other, (Tensor, np.ndarray)) else other
        )
        return Tensor(other) - self

    def __rtruediv__(self, other) -> Self:
        other = (
            np.array(other) if not isinstance(other, (Tensor, np.ndarray)) else other
        )
        return Tensor(other) / self

    def __float__(self) -> float:
        if self.value.size != 1:
            raise ValueError("Cannot convert multi-element tensor to float")
        return float(self.value.item())

    def __repr__(self) -> str:
        return f"Tensor(value={self.value}, grad={self.grad})"
