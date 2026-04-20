import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad

    def fit(self, X, y, epochs, learning_rate, batch_size):
        num_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred = self.forward(X_batch)
                dloss = self.dloss(y_pred, y_batch)
                self.backward(dloss, learning_rate)

            if epoch % 100 == 0:
                y_pred_full = self.forward(X)
                print(f"Epoch {epoch}: loss = {self.loss(y_pred_full, y)}")

    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def dloss(self, y_pred, y):
        return (2 / y.shape[0]) * (y_pred - y)

    def predict(self, x):
        return self.forward(x)
