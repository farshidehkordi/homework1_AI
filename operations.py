import numpy as np


def add_forward(a, b):
    return a.data + b.data


def add_backward(a, b, gradient):
    a.grad += gradient
    b.grad += gradient


def sub_forward(a, b):
    # TODO
    raise NotImplementedError()


def sub_backward(a, b, gradient):
    # TODO
    raise NotImplementedError()


def mul_forward(a, b):
    # TODO
    raise NotImplementedError()


def mul_backward(a, b, gradient):
    # TODO
    raise NotImplementedError()


def div_forward(a, b):
    # TODO
    raise NotImplementedError()


def div_backward(a, b, gradient):
    # TODO
    raise NotImplementedError()


def matmul_forward(a, b):
    # TODO
    raise NotImplementedError()


def matmul_backward(a, b, gradient):
    # TODO
    raise NotImplementedError()


def relu_forward(a):
    # TODO
    raise NotImplementedError()


def relu_backward(a, gradient):
    # TODO
    raise NotImplementedError()


def sigmoid_forward(a):
    # TODO
    raise NotImplementedError()


def sigmoid_backward(a, gradient):
    # TODO
    raise NotImplementedError()


def log_forward(a):
    # TODO
    raise NotImplementedError()


def log_backward(a, gradient):
    # TODO
    raise NotImplementedError()


def nll_forward(scores, label):
    _scores = scores.data - np.max(scores.data)
    exp = np.exp(_scores)
    softmax_out = exp / np.expand_dims(np.sum(exp, axis=1), axis=1)

    mask = np.full(softmax_out.shape, False)
    for i, l in enumerate(label.data):
        mask[i, l] = True

    return -np.log(softmax_out[mask] + 1e-12)


def nll_backward(scores, label, gradient):
    _scores = scores.data - np.max(scores.data)
    exp = np.exp(_scores)
    softmax_out = exp / np.expand_dims(np.sum(exp, axis=1), axis=1)

    mask = np.full(softmax_out.shape, False)
    for i, l in enumerate(label.data):
        mask[i, l] = True

    grad = np.copy(softmax_out)
    grad[mask] -= 1

    scores.grad = grad * gradient
