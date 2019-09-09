import json

import numpy as np

from model.priors import BasePrior

from pprint import pprint

from config.config import cfg


class ConditionalPrior(BasePrior):

    def __init__(self, matrix_path):

        self.prior_matrix = self.load_matrix(matrix_path)
        print('loaded matrix')
        pprint(self.prior_matrix)

    def load_matrix(self, matrix_path):
        with open(matrix_path, 'r') as f_matrix:
            matrix = json.load(f_matrix)
        return matrix

    def compute_pk(self, y_true):
        '''
        y_true: (BS, K)
        output ok: (BS, K)
        '''
        pk = np.zeros_like(y_true)
        assert pk.shape == (cfg.BATCH_SIZE, 20)

        for example in y_true:
            assert example.shape == (20,)
            pk =

        return pk

    def combine(self, sk, pk):
        raise NotImplementedError

    def pick_relabel(self, yk, y_true):
        raise NotImplementedError
