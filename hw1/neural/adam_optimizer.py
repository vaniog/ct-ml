import numpy as np


class AdamOptimizer:
    def __init__(
        self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]
        self.t = 0

    def update(self, gradients):
        self.t += 1
        lr_t = self.learning_rate * (
            np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        )

        for i, (param, grad) in enumerate(zip(self.parameters, gradients)):
            # первый момент
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # второй момент
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)
            # корректировка
            param -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + self.epsilon))
