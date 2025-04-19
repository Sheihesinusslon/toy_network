import numpy as np

from src.activation import Sigmoid
from src.nn import NeuralNetwork, Layer
from src.losses import MSE


class TestNN:
    def test_nn(self):
        network = NeuralNetwork(
            [
                Layer(2, 4, Sigmoid()),  # 2 inputs -> 3 neurons
                Layer(4, 1, Sigmoid()),  # 3 inputs -> 1 neuron
            ],
            loss_f=MSE(),
            learning_rate=0.5,
        )

        X = np.array([[0.5, 0.2], [0.1, 0.3], [-0.4, -0.8], [-0.9, -0.1]])
        y = np.array([[1.0], [1.0], [0.0], [0.0]])

        network.fit(X, y, n_epochs=100, batch_size=2)

        test_input = np.array([[0.3, 0.1], [-0.5, -0.7]])
        predictions = network.predict(test_input)
        print("Predictions:", predictions)

        assert isinstance(predictions, np.ndarray)
        rounded = np.round(predictions)
        assert np.allclose(rounded, [[1.0], [0.0]])
        accuracy = np.mean(rounded == [[1.0], [0.0]])
        assert accuracy >= 0.9
