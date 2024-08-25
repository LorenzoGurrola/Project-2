import unittest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn

if True:
    import sys
    sys.path.insert(
        0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-2/src')
    from train_model import initialize_params, forward_prop, calculate_cost, back_prop, sigmoid


class test_forward_and_back_prop(unittest.TestCase):

    def test_basic(self):
        m = 80
        x = np.random.randn(m, 1)
        y = np.random.randint(0, 2, (m, 1))
        params = initialize_params()
        yhat, inter_vals = forward_prop(x, params)
        cost = calculate_cost(yhat, y)
        grads = back_prop(x, yhat, y, inter_vals)
        dw_result = grads['dw']
        db_result = grads['db']

        x = torch.tensor(x)
        y = torch.tensor(y)
        w, b = torch.tensor(params['w'], requires_grad=True), torch.tensor(params['b'], requires_grad=True)
        z = x @ w + b
        yhat = 1/(1 + torch.exp(-z))
        losses = (y * torch.log(yhat)) + ((1 - y) * torch.log(1 - yhat))
        cost = -torch.sum(losses, dim=0, keepdims=True)/m
        cost.backward()
        dw_expected = w.grad.detach().numpy()
        db_expected = b.grad.detach().numpy()
        
        np.testing.assert_allclose(dw_result, dw_expected)
        np.testing.assert_allclose(db_result, db_expected)


unittest.main()
