import json

import numpy as np

from data.pascalvoc.preprocessing.utils import load_ids
from model.priors import BasePrior

from pprint import pprint

from config.config import cfg


class ConditionalPrior(BasePrior):
    '''
    prior based on a matrix counting the co-occurrences of different superclasses together
    superclasses for pascalvoc are:
    - animal
    - indoor
    - person
    - vehicle
    '''

    def __init__(self, matrix_path):

        class_info = load_ids()
        self.id2superclass = {v['id']: v['superclass'] for v in class_info.values()}

        self.prior_matrix = self.load_matrix(matrix_path)
        print('loaded matrix')
        pprint(self.prior_matrix)

        self.threshold = 0.5

    def load_matrix(self, matrix_path):
        with open(matrix_path, 'r') as f_matrix:
            matrix = json.load(f_matrix)
        return matrix

    def compute_pk(self, y_true):
        '''
        y_true: (BS, K)
        output ok: (BS, K)

        For each line (each example, we compute the example pk which is of size K)
        1 - for each value of the pk (one scalar), we compute the 'other key', which is of the form 'a0_i1_v1'
        2 - we retrieve the conditional probability for the value -- like p(p1|a0_i1_v1)

        TODO:
        '''
        pk = np.zeros_like(y_true)
        assert pk.shape == (cfg.BATCH_SIZE, 20)

        for example in y_true:
            assert example.shape == (20,)
            onehot_example = self.apply_threshold(example)

            for i in range(len(example)):
                # get the keys
                class_letter, class_nb, context_key = self.get_keys(i, onehot_example)

                # retrieve the conditional probability
                prob = self.get_conditional(class_letter, class_nb, context_key)
                pk[example][i] = prob

        return pk

    def apply_threshold(self, examples):
        '''
        for now just apply a 0.5 threshold
        '''
        return np.where(examples > self.threshold, 1, 0)

    def get_keys(self, index, example):
        '''
        get the key from the other classes of the example (every class except the considered class index)
        example is already onehot
        '''

        class_letter = self.id2superclass[index][0]
        assert class_letter in ['a', 'i', 'p', 'v']
        class_nb = int(example[index])
        assert class_nb in [0, 1]

        context = {'a': 0, 'i': 0, 'p': 0, 'v': 0}    # no default dict to ensure we don't have weird letters coming in // python 3 -> ordered
        for i in range(len(example)):
            letter = self.id2superclass[i][0]
            nb = int(example[i])

            context[letter] += nb

        context_key = ['%s%s' % (k, min(1, v)) for k, v in context.items() if k != class_letter]

        return class_letter, class_nb, context_key

    def get_conditional(self, class_letter, class_nb, context_key):
        '''
        '''
        if class_nb == 0:
            return 1 - self.get_conditional(class_letter, 1, context_key)

        class_key = '%s%s' % (class_letter, class_nb)
        assert class_key in self.prior_matrix, 'something got very wrong with the prior matrix (key %s)' % class_key

        # easy case, the prob is already here
        if context_key in self.prior_matrix[class_key]:
            return self.prior_matrix[class_key][context_key]

        # otherwise we should compute the prob
        # TODO: this is a manual output since the only missing prob is when all the values are 1 -> epsilon
        else:
            assert all([int(elt[1]) == 1 for elt in context_key.split('_')])
            return cfg.EPSILON

    def combine(self, sk, pk):
        raise NotImplementedError

    def pick_relabel(self, yk, y_true):
        raise NotImplementedError
