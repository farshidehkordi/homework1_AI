import numpy as np


def add_forward(a, b):
    return a.data + b.data


def add_backward(a, b, gradient):
    a.grad += gradient
    b.grad += gradient


def sub_forward(a, b):
    return a.data - b.data
   


def sub_backward(a, b, gradient):
   a.grad += gradient
    b.grad -= gradient


def mul_forward(a, b):
   return a.data * b.data


def mul_backward(a, b, gradient):
    a.grad += gradient * b.data
    b.grad += gradient * a.data


def div_forward(a, b):
    return a.data / b.data


def div_backward(a, b, gradient):
    a.grad += gradient / b.data
    b.grad += -gradient * a.data / (b.data ** 2)


def matmul_forward(a, b):
    return a.data @ b.data


def matmul_backward(a, b, gradient):
    a.grad += gradient @ b.data.T
    b.grad += a.data.T @ gradient


def relu_forward(a):
     return max(0, a)


def relu_backward(a, gradient):
   a.grad += gradient * (a.data > 0)

def sigmoid_forward(a):
   return 1 / (1 + np.exp(-a.data))



def sigmoid_backward(a, gradient):
   a.grad += gradient * (a.data * (1 - a.data))


def log_forward(a):
     return np.log(a.data)


def log_backward(a, gradient):
   a.grad += gradient / a.data


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
