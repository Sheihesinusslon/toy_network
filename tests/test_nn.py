import numpy as np

from src.activation import Sigmoid, ReLU
from src.metrics import Accuracy, R2
from src.nn import NeuralNetwork, Layer
from src.losses import MSE, BinaryCrossEntropy


class TestNN:
    def test_nn_classification(self):
        network = NeuralNetwork(
            [
                Layer(2, 4, Sigmoid()),  # 2 inputs -> 4 neurons
                Layer(4, 1, Sigmoid()),  # 3 inputs -> 1 neuron
            ],
            loss_f=BinaryCrossEntropy(),
            metrics=Accuracy(),
            learning_rate=0.1,
        )

        X = np.array([[0.5, 0.2], [0.1, 0.3], [-0.4, -0.8], [-0.9, -0.1]])
        y = np.array([[1.0], [1.0], [0.0], [0.0]])

        network.fit(X, y, n_epochs=100, batch_size=2)

        test_input = np.array([[0.3, 0.1], [-0.5, -0.7]])
        predictions = network.predict(test_input, predict_proba=False)
        print("Predictions:", predictions)

        assert isinstance(predictions, np.ndarray)
        y_exp = np.array([[1.0], [0.0]])
        assert np.array_equal(y_exp, predictions)
        assert network.score(predictions, y_exp) == 1

    def test_nn_regression(self):
        network = NeuralNetwork(
            [
                Layer(1, 8, ReLU()),  # 1 input -> 4 neurons
                Layer(8, 1),  # 4 inputs -> 1 neuron (linear output)
            ],
            loss_f=MSE(),
            metrics=R2(),
            learning_rate=0.1,
        )

        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([[0.21], [0.59], [1.02], [1.38], [1.81]])

        network.fit(X, y, n_epochs=200, batch_size=2)

        test_input = np.array([[0.2], [0.4], [0.6]])
        predictions = network.predict(test_input, predict_proba=True)
        print("Predictions:", predictions)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (3, 1)
        test_y = np.array([[0.4], [0.8], [1.2]])
        r2_score = network.score(predictions, test_y)
        assert r2_score > 0.9
