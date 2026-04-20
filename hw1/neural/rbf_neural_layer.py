import numpy as np


class RBFNeuralLayer:
    def __init__(self, num_inputs, num_outputs, beta=1.0):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.beta = beta

        # Инициализируем центры случайно
        self.centers = np.random.randn(num_outputs, num_inputs)
        # Инициализируем веса случайно
        self.weights = np.random.randn(num_outputs)

    def _radial_basis_function(self, x, center):
        # Используем функцию Гаусса в качестве RBF
        diff = x - center
        return np.exp(-self.beta * np.dot(diff, diff))

    def forward(self, x_batch):
        # Вычисляем выходы слоя для каждого образца в батче
        self.outputs = np.array(
            [
                [self._radial_basis_function(xi, center) for center in self.centers]
                for xi in x_batch
            ]
        )
        self.last_x_batch = x_batch
        res = np.dot(self.outputs, self.weights)
        return res.reshape(-1, 1)

    def backward(self, grad, learning_rate):
        # Усредним обновления для центров и весов по всем образцам в батче
        grad_centers = np.zeros_like(self.centers)
        x_batch = self.last_x_batch

        for xi, gi in zip(x_batch, grad):
            for i, center in enumerate(self.centers):
                diff = xi - center
                rbf_value = np.array([self._radial_basis_function(xi, self.centers[i])])
                gradient = 2 * self.beta * gi * self.weights[i] * rbf_value * diff
                grad_centers[i] += gradient

        # Усредняем градиенты центров
        self.centers -= learning_rate * (grad_centers / len(x_batch))

        # Обновление весов
        weight_gradient = np.dot(grad.T, self.outputs) / len(x_batch)
        weight_gradient = weight_gradient.flatten()  # Плоский массив для обновления весов
        self.weights -= learning_rate * weight_gradient
        
        return grad_centers
