import unittest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn

if True:
    import sys
    sys.path.insert(
        0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-2/src')
    from train_model import normalize, initialize_parameters, forward_propagation, calculate_cost, back_propagation, update_parameters


class test_normalize(unittest.TestCase):

    def test_basic(self):
        pass

unittest.main()
