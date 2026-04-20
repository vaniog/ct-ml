import numpy as np


def stable_softmax(logits):
    exp_shifted = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


def cross_entropy_loss(probs, labels):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return -np.sum(labels * np.log(probs)) / labels.shape[0]


def softargmax_crossentropy_with_logits(logits, labels):
    probs = stable_softmax(logits)
    loss = cross_entropy_loss(probs, labels)
    grad = probs - labels
    grad /= labels.shape[0]

    return loss, grad
