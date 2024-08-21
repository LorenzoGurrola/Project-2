
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def normalize(data):
        if(data.all() == 0):
              return data
        min = np.min(data)
        max = np.max(data)
        range = max-min
        if(range == 0):
            return np.zeros_like(data)
        data_norm = (data - min)/range
        return data_norm

def initialize_parameters():
    w = np.random.rand(1,1) * 0.1
    b = np.zeros((1,1))
    parameters = {'w':w, 'b':b}
    return parameters

def forward_propagation(x, parameters):
    w, b = parameters['w'], parameters['b']
    yhat = np.matmul(x, w) + b
    yhat = np.reshape(yhat, (yhat.shape[0], 1))
    return yhat

def calculate_cost(yhat, y):
    m = y.shape[0]
    error = yhat - y
    squared_error = error ** 2
    cost = np.sum(squared_error, 0)/m
    intermediate_values = {'error':error}
    return cost, intermediate_values

def back_propagation(x, intermediate_values):
    m = x.shape[0]
    error = intermediate_values['error']
    dw = x * (2 * error) * (1/m)
    dw = np.sum(dw, axis=0, keepdims=True)
    db = (2 * error) * (1/m)
    db = np.sum(db, axis=0, keepdims=True)
    grads = {'dw':dw, 'db':db}
    return grads

def update_parameters(parameters, grads, learning_rate=0.1):
    w, b = parameters['w'], parameters['b']
    dw, db = grads['dw'], grads['db']
    w = w - learning_rate * dw
    b = b - learning_rate * db
    parameters = {'w':w, 'b':b}
    return parameters


if __name__ == '__main__':
    data = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
    
    train, test = train_test_split(data, train_size=0.75, random_state=10)

    x_train_raw = np.array(train['Experience Years'])
    y_train_raw = np.array(train['Salary'])
    x_test_raw = np.array(test['Experience Years'])
    y_test_raw = np.array(test['Salary'])

    x_train_raw = np.reshape(x_train_raw, (x_train_raw.shape[0], 1))
    y_train_raw = np.reshape(y_train_raw, (y_train_raw.shape[0], 1))
    x_test_raw = np.reshape(x_test_raw, (x_test_raw.shape[0], 1))
    y_test_raw = np.reshape(y_test_raw, (y_test_raw.shape[0], 1))

    x_train = normalize(x_train_raw)
    y_train = normalize(y_train_raw)
    x_test = normalize(x_test_raw)
    y_test = normalize(y_test_raw)
