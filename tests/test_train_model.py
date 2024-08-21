import unittest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn

if True:
    import sys
    sys.path.insert(
        0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-1/src')
    from train_model import normalize, initialize_parameters, forward_propagation, calculate_cost, back_propagation, update_parameters


class test_normalize(unittest.TestCase):

    def test_basic(self):
        input = np.reshape(np.array([1, 2, 3, 5]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(
            result, expected, decimal=6, verbose=True)

    def test_zero(self):
        input = np.reshape(np.array([0, 0, 0, 0]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(
            result, expected, decimal=6, verbose=True)

    def test_negative(self):
        input = np.reshape(np.array([-1, -2, -8, -3]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(
            result, expected, decimal=6, verbose=True)

    def test_extreme_small(self):
        input = np.reshape(
            np.array([0.0001, 0.0002, 0.0003, -0.0004]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(
            result, expected, decimal=6, verbose=True)

    def test_extreme_large(self):
        input = np.reshape(
            np.array([10000, 200000, 50000, 10000, 2000000]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(
            result, expected, decimal=6, verbose=True)

    def test_same(self):
        input = np.reshape(np.array([5, 5]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(
            result, expected, decimal=6, verbose=True)


class test_initialize_parameters(unittest.TestCase):

    def test_basic(self):
        parameters = initialize_parameters()
        w, b = parameters['w'], parameters['b']
        self.assertGreaterEqual(w, 0)
        self.assertLessEqual(w, 0.1)
        self.assertEqual(b, 0)


class test_forward_propagation(unittest.TestCase):

    def test_basic(self):
        x = np.random.rand(30, 1)
        w = np.random.rand(1, 1) * 0.1
        b = np.zeros((1, 1))
        parameters = {'w': w, 'b': b}
        result = forward_propagation(x, parameters)

        linear = nn.Linear(1, 1)
        linear.weight.data = torch.tensor(w)
        linear.bias.data = torch.tensor(b)
        expected = linear(torch.tensor(x)).detach().numpy()

        np.testing.assert_allclose(result, expected)


class test_calculate_cost(unittest.TestCase):

    def test_basic(self):
        yhat = np.random.rand(30, 1)
        y = np.random.rand(30, 1)
        result, intermediate_values = calculate_cost(yhat, y)

        loss = torch.nn.MSELoss()
        expected = loss(torch.tensor(yhat), torch.tensor(y)).detach().numpy()
        np.testing.assert_allclose(result, expected)

class test_forward_and_backward_pass(unittest.TestCase):

    def test_basic(self):
        m = 30
        x = np.random.randn(m,1)
        y = np.random.rand(m, 1)
        parameters = initialize_parameters()
        yhat = forward_propagation(x, parameters)
        cost, intermediate_values = calculate_cost(yhat, y)
        grads = back_propagation(x, intermediate_values)
        dw_result = grads['dw']
        db_result = grads['db']

        x = torch.tensor(x)
        y = torch.tensor(y)
        w, b = torch.tensor(parameters['w'], requires_grad=True), torch.tensor(parameters['b'], requires_grad=True)
        yhat = x@w + b
        cost = torch.sum((yhat - y) ** 2, dim=0, keepdim=True)/m
        cost.backward()
        dw_expected = w.grad.detach().numpy()
        db_expected = b.grad.detach().numpy()
        
        np.testing.assert_allclose(dw_result, dw_expected)
        np.testing.assert_allclose(db_result, db_expected)

class test_update_parameters(unittest.TestCase):

    def test_basic(self):
        parameters = {'w':np.array([0.1]),'b':np.array([0])}
        grads = {'dw':np.array([-0.1]), 'db':np.array([0.1])}
        learning_rate = 0.1
        parameters = update_parameters(parameters, grads, learning_rate)
        w_result = parameters['w']
        b_result = parameters['b']
        w_expected = np.array([0.11])
        b_expected = np.array([-0.01])

        np.testing.assert_allclose(w_result, w_expected)
        np.testing.assert_allclose(b_result, b_expected)

unittest.main()
