import json

import numpy as np

from data.pascalvoc.preprocessing.utils import load_ids
from model.priors.base import BasePrior

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

    def __init__(self, matrix_path, nb_classes, method='product'):

        class_info = load_ids()
        self.id2superclass = {v['id']: v['superclass'] for v in class_info.values()}

        self.prior_matrix = self.load_matrix(matrix_path)

        self.nb_classes = nb_classes

        self.threshold = 0.5
        self.combination_method = method

    def load_matrix(self, matrix_path):
        with open(matrix_path, 'r') as f_matrix:
            matrix = json.load(f_matrix)
        return matrix

    def compute_pk(self, y_true):
        '''
        y_true: (BS, K)
        output ok: (BS, K, 2)

        For each line (each example, we compute the example pk which is of size K)
        1 - for each value of the pk (one scalar), we compute the 'other key', which is of the form 'a0_i1_v1'
        2 - we retrieve the conditional probability for the value -- like p(p1|a0_i1_v1)

        TODO:
        '''
        y_true = np.asarray(y_true)

        pk = np.zeros(y_true.shape + (2,), dtype=np.float64)
        assert pk.shape == (cfg.BATCH_SIZE, self.nb_classes, 2), 'wrong pk shape %s' % str(pk.shape)

        for i, example in enumerate(y_true):
            assert example.shape == (self.nb_classes,), 'wrong example shape %s' % str(example.shape)
            # onehot_example = self.apply_threshold(example)

            # for each class
            for j in range(len(example)):
                # get the keys
                class_letter, class_nb, context_key = self.get_keys(j, example)

                # retrieve the conditional probability + normalization
                prob0 = self.get_conditional(class_letter, 0, context_key)
                prob1 = self.get_conditional(class_letter, 1, context_key)

                pk[i][j][0] = prob0 / (prob0 + prob1)
                pk[i][j][1] = prob1 / (prob0 + prob1)

        return pk

    def apply_threshold(self, examples):
        '''
        for now just apply a 0.5 threshold
        '''
        return np.where(examples > self.threshold, 1, 0)

    def get_keys(self, index, example):
        '''
        get the key from the other classes of the example (every class except the considered class index)

        example is considered a y_true example with -1, 0 and +1 values
        -1 and +1 values are used to create the context key, not the 0 values
        '''

        class_letter = self.id2superclass[index][0]
        assert class_letter in ['a', 'i', 'p', 'v'], 'wrong superclass letter %s' % class_letter
        class_nb = int(example[index])
        assert class_nb in [-1, 0, 1], 'wrong class value %s' % class_nb

        context = {'a': set(), 'i': set(), 'p': set(), 'v': set()}    # no default dict to ensure we don't have weird letters coming in // python 3 -> ordered
        for i in range(len(example)):
            letter = self.id2superclass[i][0]
            val = int(example[i])

            context[letter].add(val)

        context_parts = list()
        for letter in sorted(context.keys()):
            # we don't take the observed key in the context
            if letter == class_letter:
                continue

            # we don't take missing keys in the context (it is going to be a partial context)
            if context[letter] == {0}:
                continue

            # at least one positive value -> +1 for the context
            if 1 in context[letter]:
                context_parts.append('%s1' % letter)

            # no 1 in the context -> 0 (we know the class is not there, not the absence of information covered by 0s)
            else:
                context_parts.append('%s0' % letter)

        context_key = '_'.join(context_parts)

        return class_letter, class_nb, context_key

    def get_conditional(self, class_letter, class_nb, context_key):
        '''
        '''
        class_key = '%s%s' % (class_letter, class_nb)
        assert class_key in self.prior_matrix, 'something got very wrong with the prior matrix (key %s)' % class_key

        # rare case where we have no information at all
        if context_key not in self.prior_matrix[class_key]:
            return 0.5

        return self.prior_matrix[class_key][context_key]

    def combine(self, sk, pk):
        '''
        produce the outputs weighted by the prior

        Different methods:
        - product: just element wise product of the visual info (sk) and the prior (pk)
        - sigmoid
        - softmax

        pk (BS, K, 2)
        sk (BS, K)

        output: (BS, K) <- RENORMALIZED
        '''
        sk = np.asarray(sk)

        assert sk.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong sk shape %s' % str(sk.shape)
        assert pk.shape == (cfg.BATCH_SIZE, self.nb_classes, 2), 'wrong pk shape %s' % str(pk.shape)

        ones_yk = sk * pk[:, :, 1]
        zeros_yk = (1 - sk) * pk[:, :, 0]
        return ones_yk / (ones_yk + zeros_yk)
