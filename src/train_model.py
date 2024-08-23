from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def load_data():
    mean = 5
    std_dev = 1
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
    x_train, x_test = train_test_split(x, train_size=0.8)
    y_train = (x_train>6).astype(int)
    y_test = (x_test>6).astype(int)

    x_train = np.reshape(x_train, (x_train.shape[0], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    train_data = {'x_train':x_train, 'y_train':y_train}
    test_data = {'x_test':x_test, 'y_test':y_test}
    return train_data, test_data

def initialize_params():
    w = np.random.rand(1,1) * 0.1
    b = np.zeros((1,1))
    params = {'w':w,'b':b}
    return params

def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a

def forward_prop(x, params):
    w, b = params['w'], params['b']
    z = np.matmul(x, w) + b
    inter_vals = {'z':z}
    yhat = sigmoid(z)
    return yhat, inter_vals

def calculate_cost(yhat, y):
    m = y.shape[0]
    losses = (y * np.log(yhat)) + ((1 - y) * np.log(1 - yhat))
    cost = -np.sum(losses, axis=0, keepdims=True)/m
    return cost

def back_prop(x, yhat, y, inter_vals):
    m = y.shape[0]
    z = inter_vals['z']
    dc_dyhat = (-1/m) * ((y/yhat) - ((1-y)/(1-yhat)))
    assert dc_dyhat.shape == (80,1)
    dyhat_dz = sigmoid(z) * (1 - sigmoid(z))
    assert dyhat_dz.shape == (80,1)
    dz_dc = dyhat_dz * dc_dyhat
    dw_dc = np.sum(dz_dc * x, axis=0, keepdims=True)
    db_dc = np.sum(dz_dc, axis=0, keepdims=True)
    assert dw_dc.shape == (1,1)
    assert db_dc.shape == (1,1)
    grads = {'db':db_dc, 'dw':dw_dc}
    return grads