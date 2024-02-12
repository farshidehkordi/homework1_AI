import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout

from operations import *


class Variable:
    def __init__(self, data, name='', _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._name = name

    def zero_grad(self):
        self.grad = 0

    def _generic_unop(self, forward, backward, op):
        out = Variable(forward(self), _children=(self,), _op=op)

        def _backward():
            backward(self, out.grad)

        out._backward = _backward

        return out

    def _generic_binop(self, other, forward, backward, op):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(forward(self, other), _children=(self, other), _op=op)

        def _backward():
            backward(self, other, out.grad)

        out._backward = _backward

        return out

    def __add__(self, other):
        return self._generic_binop(other, add_forward, add_backward, '+')

    def __sub__(self, other):
        return self._generic_binop(other, sub_forward, sub_backward, '-')

    def __mul__(self, other):
        return self._generic_binop(other, mul_forward, mul_backward, '*')

    def __truediv__(self, other):
        return self._generic_binop(other, div_forward, div_backward, '/')

    def __matmul__(self, other):
        return self._generic_binop(other, matmul_forward, matmul_backward, '@')

    def __neg__(self):
        return self * -1

    def relu(self):
        return self._generic_unop(relu_forward, relu_backward, 'ReLU')

    def sigmoid(self):
        return self._generic_unop(sigmoid_forward, sigmoid_backward, 'σ')

    def log(self):
        return self._generic_unop(log_forward, log_backward, 'log')

    def nll(self, other):
        return self._generic_binop(other, nll_forward, nll_backward, 'NLL')

    def backward(self, _initial_grad=None):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        if _initial_grad is None:
            _initial_grad = np.array([1])

        self.grad = _initial_grad
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f'Value({("name=" + self._name + " ") if self._name != "" else ""}data={self.data}, grad={self.grad})'

    def show(self):
        graph = nx.DiGraph()
        labels = dict()
        seen = set()

        def size_to_str(size):
            return '(' + ', '.join(['%d' % v for v in size]) + ')'

        def add_nodes(var, parent=None):
            if var in seen:
                return
            var_id = str(id(var))
            graph.add_node(var_id)
            has_shape = hasattr(var.data, 'shape')
            labels[var_id] = f'{var._name if var._name != "" else "Var"}\n' + (
                size_to_str(var.data.shape) if has_shape else '')
            if parent is not None:
                graph.add_edge(parent, var_id)
            new_parent = var_id

            is_op = var._op != ''
            if is_op:
                op_id = var_id + '_op_' + var._op
                graph.add_node(op_id)
                labels[op_id] = var._op
                graph.add_edge(var_id, op_id)
                new_parent = op_id
            seen.add(var)
            for c in var._prev:
                add_nodes(c, new_parent)

        add_nodes(self)

        plt.figure(figsize=(10, 15))
        pos = graphviz_layout(graph, prog="dot")
        nx.draw(graph, pos=pos, labels=labels, with_labels=True, node_size=1e3, node_color='lightblue', node_shape='s')
        plt.show()
