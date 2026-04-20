import numpy as np

"""
Реализуйте стандартное обучаемое преобразование:
умножение на матрицу, добавление вектора и применение функции «активации».
Преобразование должно пересчитывать градиент для входа и параметров.
Должны поддерживаться не менее трёх функции «активации». Желательно реализовать: тождественную, ReLU, tanh.
"""


class SimpleNeuralLayer:
    """
    W - weights (wrong but sorry)
    b - biases
    X - input
    Y - output
    """

    def __init__(
        self, input_size, output_size, activation_function="identity", biases=None
    ):
        self.W = np.random.randn(input_size, output_size) * 0.01
        if biases is None:
            biases = np.zeros((1, output_size))
        self.b = biases
        self.activation_function = activation_function

    def activation(self, s):
        if self.activation_function == "identity":
            return s
        elif self.activation_function == "relu":
            return np.maximum(0, s)
        elif self.activation_function == "tanh":
            return np.tanh(s)
        else:
            raise ValueError(f"Unexpected activation: {self.activation_function}")

    def activation_derivative(self, s):
        if self.activation_function == "identity":
            return np.ones_like(s)
        elif self.activation_function == "relu":
            return np.where(s > 0, 1, 0)
        elif self.activation_function == "tanh":
            return 1 - np.tanh(s) ** 2
        else:
            raise ValueError(f"Unexpected activation: {self.activation_function}")

    def forward(self, X):
        self.last_X = X
        self.S = np.dot(X, self.W) + self.b
        self.Y = self.activation(self.S)
        return self.Y

    def backward(self, dY, learning_rate=0.01):
        dS = dY * self.activation_derivative(self.S)
        dW = np.dot(self.last_X.T, dS)
        db = np.sum(dS, axis=0, keepdims=True) * 1 / self.last_X.shape[0]
        dX = np.dot(dS, self.W.T)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dX
