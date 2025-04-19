import numpy as np
import pytest

from src.tensor import Tensor


class TestTensorOpsScalar:
    def test_tensor_add(self):
        t = Tensor(5)
        result = t + 5

        assert isinstance(result, Tensor)
        assert result.value == 10

        result2 = 5 + t

        assert isinstance(result2, Tensor)
        assert result2.value == 10

        result3 = t + Tensor(5)
        assert isinstance(result3, Tensor)
        assert result3.value == 10

    def test_tensor_sub(self):
        t = Tensor(5)
        result = t - 2

        assert isinstance(result, Tensor)
        assert result.value == 3

        result2 = 5 - t

        assert isinstance(result2, Tensor)
        assert result2.value == 0

        result3 = t - Tensor(2)

        assert isinstance(result3, Tensor)
        assert result3.value == 3

    def test_tensor_mul(self):
        t = Tensor(5)
        result = t * 2

        assert isinstance(result, Tensor)
        assert result.value == 10

        result2 = 2 * t

        assert isinstance(result2, Tensor)
        assert result2.value == 10

        result3 = t * Tensor(2)

        assert isinstance(result3, Tensor)
        assert result3.value == 10

    def test_tensor_div(self):
        t = Tensor(10)
        result = t / 2

        assert isinstance(result, Tensor)
        assert result.value == 5

        result2 = 10 / t

        assert isinstance(result2, Tensor)
        assert result2.value == 1

        result3 = t / Tensor(2)

        assert isinstance(result3, Tensor)
        assert result3.value == 5

    def test_tensor_pow(self):
        t = Tensor(2)
        result = t**3

        assert isinstance(result, Tensor)
        assert result.value == 8


class TestTensorBackprop:
    def test_tensor_backprop(self):
        """Test the backward pass of a simple func
        L=(x+y)×z
        """
        x = Tensor(2.0)
        y = Tensor(3.0)
        z = Tensor(4.0)

        q = x + y  # q = x + y
        L = q * z  # L = q * z

        L.backward()

        # Now we expect:
        # dL/dx = z = 4.0
        # dL/dy = z = 4.0
        # dL/dz = q = 5.0 (because 2+3=5)

        assert abs(x.grad - 4.0) < 1e-6
        assert abs(y.grad - 4.0) < 1e-6
        assert abs(z.grad - 5.0) < 1e-6


class TestTensorOpsMatrix:
    def test_array_creation(self):
        t = Tensor([1, 2, 3])
        assert isinstance(t, Tensor)
        assert t.value.shape == (3,)

        t2 = Tensor(
            np.zeros(
                4,
            )
        )
        assert isinstance(t2, Tensor)
        assert t2.value.shape == (4,)

    def test_matrix_creation(self):
        t = Tensor([[1, 2], [3, 4]])
        assert isinstance(t, Tensor)
        assert t.value.shape == (2, 2)

        t2 = Tensor(np.array([[1, 2], [3, 4]]))
        assert isinstance(t2, Tensor)
        assert t2.value.shape == (2, 2)

    def test_array_addition(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = a + b
        assert np.array_equal(result.value, np.array([5, 7, 9]))

        # Test broadcasting
        result2 = a + 1
        assert np.array_equal(result2.value, np.array([2, 3, 4]))

    def test_matrix_multiplication(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5], [6]])
        result = a @ b
        assert result.value.shape == (2, 1)
        assert np.allclose(result.value, np.array([[17], [39]]))

    def test_elementwise_multiplication(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = a * b
        assert np.array_equal(result.value, np.array([4, 10, 18]))

    def test_array_power(self):
        a = Tensor([1, 2, 3])
        result = a**2
        assert np.array_equal(result.value, np.array([1, 4, 9]))

    def test_array_division(self):
        a = Tensor([4, 9, 16])
        result = a / 2
        assert np.array_equal(result.value, np.array([2, 4.5, 8]))

    def test_sum_operation(self):
        a = Tensor([[1, 2], [3, 4]])
        result = a.sum()
        assert result.value == 10

        result2 = a.sum(axis=0)
        assert np.array_equal(result2.value, np.array([4, 6]))

    def test_mean_operation(self):
        a = Tensor([[1, 2], [3, 4]])
        result = a.mean()
        assert result.value == 2.5

        result2 = a.mean(axis=0)
        assert np.array_equal(result2.value, np.array([2, 3]))


class TestTensorArrayBackprop:
    def test_simple_array_backprop(self):
        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])
        z = x * y  # [3.0, 8.0]
        z.backward()

        # dz/dx = y
        assert np.allclose(x.grad, np.array([3.0, 4.0]))
        # dz/dy = x
        assert np.allclose(y.grad, np.array([1.0, 2.0]))

    def test_matrix_multiplication_backprop(self):
        W = Tensor([[1, 2], [3, 4]])  # 2x2
        x = Tensor([[5], [6]])  # 2x1
        y = W @ x  # 2x1

        y.backward()

        # dy/dW = x^T (but properly broadcast)
        assert np.allclose(W.grad, np.array([[5, 6], [5, 6]]))
        # dy/dx = W^T
        assert np.allclose(x.grad, np.array([[1 + 3], [2 + 4]]))

    def test_broadcast_backprop(self):
        x = Tensor([[1], [2]])  # 2x1
        y = Tensor([3, 4])  # 1x2
        z = x * y  # Broadcasts to 2x2

        z.backward()

        # dz/dx = sum over y's dimensions
        assert np.allclose(x.grad, np.array([[3 + 4], [3 + 4]]))
        # dz/dy = sum over x's dimensions
        assert np.allclose(y.grad, np.array([1 + 2, 1 + 2]))

    def test_sum_backprop(self):
        x = Tensor([[1, 2], [3, 4]])
        y = x.sum()  # 10

        y.backward()

        # dy/dx = ones_like(x)
        assert np.allclose(x.grad, np.array([[1, 1], [1, 1]]))

    def test_chain_rule_array(self):
        x = Tensor([2.0, 3.0])
        y = Tensor([1.0, 4.0])
        z = (x * y).sum()  # 2*1 + 3*4 = 14

        z.backward()

        # dz/dx = y
        assert np.allclose(x.grad, np.array([1.0, 4.0]))
        # dz/dy = x
        assert np.allclose(y.grad, np.array([2.0, 3.0]))

    def test_multi_layer_backprop(self):
        # Simple neural network layer
        W = Tensor([[1, 2], [3, 4]])
        x = Tensor([5.0, 6.0])
        b = Tensor([0.1, 0.2])

        # Forward pass
        h = (W @ x) + b  # [1*5+2*6 + 0.1, 3*5+4*6 + 0.2] = [17.1, 39.2]
        y = h.sum()  # 56.3

        # Backward pass
        y.backward()

        # dy/dh = 1 (for both elements)
        # dh/dW = x (broadcast properly)
        assert np.allclose(W.grad, np.array([[5, 6], [5, 6]]))
        # dh/dx = W^T
        assert np.allclose(x.grad, np.array([1 + 3, 2 + 4]))
        # dh/db = 1 (for both elements)
        assert np.allclose(b.grad, np.array([1.0, 1.0]))


class TestTensorArrayEdgeCases:
    def test_incompatible_shapes(self):
        a = Tensor([[1, 2], [3, 4]])  # 2x2
        b = Tensor([1, 2, 3])  # 3

        with pytest.raises(ValueError):
            _ = a + b  # incompatible shapes

        with pytest.raises(ValueError):
            _ = a @ b  # inner dimensions don't match

    def test_empty_tensor(self):
        a = Tensor([])
        b = a + 1
        assert b.value.shape == (0,)

        with pytest.raises(ValueError):
            _ = a @ a  # Matrix multiplication with empty arrays

    def test_scalar_array_mix(self):
        a = Tensor(5)
        b = Tensor([1, 2, 3])

        result = a + b
        assert np.array_equal(result.value, np.array([6, 7, 8]))

        result2 = b * a
        assert np.array_equal(result2.value, np.array([5, 10, 15]))
