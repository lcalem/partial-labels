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

    def __init__(self, matrix_path, nb_classes=20, method='product'):

        class_info = load_ids()
        self.id2superclass = {v['id']: v['superclass'] for v in class_info.values()}

        self.prior_matrix = self.load_matrix(matrix_path)

        self.nb_classes = 20

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
            onehot_example = self.apply_threshold(example)

            for j in range(len(example)):
                # get the keys
                class_letter, class_nb, context_key = self.get_keys(j, onehot_example)

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
        example is already onehot
        '''

        class_letter = self.id2superclass[index][0]
        assert class_letter in ['a', 'i', 'p', 'v'], 'wrong superclass letter %s' % class_letter
        class_nb = int(example[index])
        assert class_nb in [0, 1], 'wrong class value %s' % class_nb

        context = {'a': 0, 'i': 0, 'p': 0, 'v': 0}    # no default dict to ensure we don't have weird letters coming in // python 3 -> ordered
        for i in range(len(example)):
            letter = self.id2superclass[i][0]
            nb = int(example[i])

            context[letter] += nb

        context_key = '_'.join(sorted(['%s%s' % (k, min(1, v)) for k, v in context.items() if k != class_letter]))

        return class_letter, class_nb, context_key

    def get_conditional(self, class_letter, class_nb, context_key):
        '''
        '''
        class_key = '%s%s' % (class_letter, class_nb)
        assert class_key in self.prior_matrix, 'something got very wrong with the prior matrix (key %s)' % class_key

        # easy case, the prob is already here
        if context_key in self.prior_matrix[class_key]:
            return self.prior_matrix[class_key][context_key]

        # otherwise we should compute the prob
        # TODO: this is a manual output since the only missing prob is when all the values are 1 -> epsilon
        else:
            # assert all([int(elt[1]) == 1 for elt in context_key.split('_')]), 'unknown state is not full 1 %s' % context_key
            return cfg.EPSILON

    def combine(self, sk, pk):
        '''
        produce the outputs weighted by the prior

        Different methods:
        - product: just element wise product of the visual info (sk) and the prior (pk)
        - sigmoid
        - softmax

        pk (BS, K, 2)
        sk (BS, K)

        output: (BS, K)
        '''
        sk = np.asarray(sk)

        assert sk.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong sk shape %s' % str(sk.shape)
        assert pk.shape == (cfg.BATCH_SIZE, self.nb_classes, 2), 'wrong pk shape %s' % str(pk.shape)

        ones_yk = sk * pk[:, :, 1]
        zeros_yk = (1 - sk) * pk[:, :, 0]
        return ones_yk / (ones_yk + zeros_yk)

    def pick_relabel(self, y_pred, yk, y_true):
        '''
        yk: prior-weighted outputs
        y_true: true batch (with zeros where the label is missing)

        1. find the values of yk for which there is a missing label at the same index in the y_true batch
        2. order those values and get the corresponding indexes
        3. take the 'best' 33% of these values and put a 1 in the relabel output at these indexes
        '''
        y_pred = np.asarray(y_pred)

        assert y_pred.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_pred shape %s' % str(y_pred.shape)
        assert yk.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong yk shape %s' % str(yk.shape)
        assert y_true.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_true shape %s' % str(y_true.shape)

        relabel_batch = np.copy(y_true)

        # find and sort the relevant values of the outputs
        relevant_yk = np.where((y_true == 0) & (y_pred > 0.5), yk, 0)   # consider only the values for missing labels and for which the initial prediction is > 0.5
        sorted_indexes = np.argsort(relevant_yk, axis=None)   # the indexes are flattened
        sorted_values = [relevant_yk[i // self.nb_classes][i % self.nb_classes] for i in sorted_indexes]

        # take the best values and find corresponding indexes
        nb_ok_indexes = int((np.count_nonzero(sorted_values)) * 0.33)
        final_indexes = [(i // self.nb_classes, i % self.nb_classes) for i in sorted_indexes[len(sorted_values) - nb_ok_indexes: len(sorted_values)]]

        # put the selected values as 1 in the relabel batch
        xs = [elt[0] for elt in final_indexes]
        ys = [elt[1] for elt in final_indexes]
        relabel_batch[(xs, ys)] = 1

        # sanity check (check that the added values are only ones and that they are only where there was a 0 before)
        check = np.where(y_true != relabel_batch, relabel_batch, 0)
        assert np.count_nonzero(check) == nb_ok_indexes     # check we added the correct number of values
        assert np.all(check[np.where(check != 0)] == 1)     # check we only added ones
        assert np.all(y_true[np.where(check == 1)] == 0)    # check we added values only where the initial batch was 0

        return relabel_batch, nb_ok_indexes
