import json

import numpy as np

from data.pascalvoc.preprocessing.utils import load_ids
from model.priors.base import BasePrior

from pprint import pprint

from config.config import cfg

from collections import defaultdict


class ConditionalPrior(BasePrior):
    '''
    prior based on a matrix counting the co-occurrences of classes
    '''

    def __init__(self,
                 matrix_path,
                 nb_classes,
                 context_length,
                 comb_method='simple',
                 use_superclass=False,
                 alpha=None,
                 mode='train',
                 allow_marginalization=True):
        '''
        context_length: the expected length of the context (nb of classes in the context)

        use_superclass means we use a prior computed on superclasses

        superclasses are:
            - animal
            - indoor
            - person
            - vehicle

        combination methods:
        - simple -> add the logits of y_v and y_p
        - alpha -> weight y_p with alpha first
        '''
        print('loading conditional prior with method %s, superclasses %s, alpha %s, mode %s (context length %s)' % (comb_method, use_superclass, alpha, mode, context_length))
        self.use_superclass = use_superclass

        class_info = load_ids()
        self.id2superclass = {v['id']: v['superclass'] for v in class_info.values()}
        self.id2short = {v['id']: v['short'] for v in class_info.values()}

        self.prior_matrix = self.load_matrix(matrix_path)
        self.nb_classes = nb_classes
        self.context_length = context_length

        self.threshold = 0.5
        self.combination_method = comb_method
        self.mode = mode
        assert self.mode in ['train', 'test']

        if comb_method == 'alpha':
            self.alpha = alpha

        self.allow_marginalization = allow_marginalization

    def load_matrix(self, matrix_path):
        with open(matrix_path, 'r') as f_matrix:
            matrix = json.load(f_matrix)
        return matrix

    def get_logits(self, values):
        '''
        values are (batch_size, nb_classes) size
        ln (p / 1-p)
        '''
        assert values.shape == (cfg.BATCH_SIZE, self.nb_classes)
        return np.log(values / (1 - values))

    def get_sigmoid(self, values):
        return 1.0 / (1.0 + np.exp(-values))

    def compute_pk_prob(self, y_true):
        '''
        y_true: (BS, K)
        output pk: (BS, K)

        For each line (each example, we compute the example pk which is of size K)
        1 - for each value of the pk (one scalar), we compute the 'context_key', which is of the form 'a0_i1_v1'
        2 - we retrieve the conditional probability for the value -- like p(p1|a0_i1_v1)

        In the logit space -> output is the logit of the prob prior
        '''
        y_true = np.asarray(y_true)

        pk = np.zeros(y_true.shape, dtype=np.float64)
        assert pk.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong pk shape %s' % str(pk.shape)

        for i, example in enumerate(y_true):
            assert example.shape == (self.nb_classes,), 'wrong example shape %s' % str(example.shape)
            # onehot_example = self.apply_threshold(example)

            example = self.preprocess_example(example)

            # for each class
            for j in range(len(example)):
                # get the keys
                class_letter, class_nb, context_key = self.get_keys(j, example)

                # retrieve the conditional probability + normalization
                prob0 = self.get_conditional(class_letter, 0, context_key)
                prob1 = self.get_conditional(class_letter, 1, context_key)

                # print('example for class %s' % j)
                # print(example)

                # print('context key')
                # print(context_key)

                # print('prob0 %s, prob1 %s' % (prob0, prob1))

                pk[i][j] = prob1 / (prob0 + prob1)

        return pk

    def preprocess_example(self, example):
        '''
        thresholding, -1 / 0, etc
        '''
        if self.mode == 'test':
            return [-1 if elt == 0 else elt for elt in example]
        return example

    def compute_pk_logits(self, y_true):
        pk = self.compute_pk_prob(y_true)
        return self.get_logits(pk)

    # def apply_threshold(self, examples):
    #     '''
    #     for now just apply a 0.5 threshold
    #     '''
    #     return np.where(examples > self.threshold, 1, 0)

    def get_keys(self, index, example):
        '''
        get the key from the other classes of the example (every class except the considered class index)

        example is considered a y_true example with -1, 0 and +1 values
        -1 and +1 values are used to create the context key, not the 0 values
        '''

        class_letter = self.get_letter(index)
        class_nb = int(example[index])
        assert class_nb in [-1, 0, 1], 'wrong class value %s' % class_nb

        context = self.init_empty_context()
        for i in range(len(example)):
            letter = self.get_letter(i)
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

    def get_letter(self, index):
        '''
        depends on whether we use the superclass or not
        '''
        if self.use_superclass is True:
            return self.id2superclass[index][0]
        else:
            return self.id2short[index]

    def init_empty_context(self):
        '''
        -> default dict with sets
        empty context keys are the superclass letter if full == False or the shorthand for the class if full == True

        - no default dict to ensure we don't have weird letters coming in
        - python 3 -> ordered
        '''
        if self.use_superclass is True:
            keys = self.id2superclass.values()
        else:
            keys = self.id2short.values()

        return {k: set() for k in sorted(keys)}

    def get_conditional(self, class_letter, class_nb, context_key):
        '''
        retrieves the conditional probability p(class_letter = class_nb | context_key) from the matrix
        if the probability is not there, marginalize on the fly
        '''
        assert class_nb in [0, 1], 'wrong class_nb value %s' % class_nb
        class_key = '%s%s' % (class_letter, class_nb)
        assert class_key in self.prior_matrix, 'something got very wrong with the prior matrix (key %s) (all %s)' % (class_key, str(list(self.prior_matrix.keys())))

        prob_value = self.prior_matrix[class_key].get(context_key)

        # we didn't find the value in the matrix
        if prob_value is None:
            if self.allow_marginalization is False:
                raise Exception('marginalization is not allowed. key %s not in matrix' % (context_key))

            # for full contexts we may have to compute the marginal for a partial context
            partial_parts = context_key.split('_')

            # we add all the values of the contexts that match the partial one
            sum_one = sum([val for ctxt, val in self.prior_matrix['%s1' % class_letter].items() if any([part in ctxt for part in partial_parts])])
            sum_zero = sum([val for ctxt, val in self.prior_matrix['%s0' % class_letter].items() if any([part in ctxt for part in partial_parts])])

            if class_nb == 0:
                prob_value = sum_zero / (sum_zero + sum_one)

            elif class_nb == 1:
                prob_value = sum_one / (sum_zero + sum_one)

        return prob_value

    def combine(self, y_v_logits, y_p_logits):
        '''
        --- in the logit space ---
        produce the outputs weighted by the prior

        Different methods:
        - product: just element wise product of the visual info (sk) and the prior (pk)

        pk (BS, K)
        sk (BS, K)

        output: (BS, K) <- RENORMALIZED

        Returns a PROB (from the logits -> sigmoid)
        '''
        y_v_logits = np.asarray(y_v_logits)

        assert y_v_logits.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_v_logits shape %s' % str(y_v_logits.shape)
        assert y_p_logits.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_p_logits shape %s' % str(y_p_logits.shape)

        if self.combination_method == 'simple':
            y_f_logits = y_v_logits + y_p_logits
        elif self.combination_method == 'alpha':
            y_f_logits = y_v_logits + self.alpha * y_p_logits

        return self.get_sigmoid(y_f_logits)


class ConditionalRawPrior(ConditionalPrior):
    '''
    prior based on a matrix counting the co-occurrences of classes
    raw= normalization has not been done so we do it on the fly
    '''
    def __init__(self,
                 matrix_path,
                 nb_classes,
                 context_length,
                 comb_method='simple',
                 use_superclass=False,
                 alpha=None,
                 mode='train',
                 allow_marginalization=True,
                 max_weight=0.6):

        ConditionalPrior.__init__(self,
                                  matrix_path,
                                  nb_classes,
                                  context_length,
                                  comb_method=comb_method,
                                  use_superclass=use_superclass,
                                  alpha=alpha,
                                  mode=mode,
                                  allow_marginalization=allow_marginalization)

        self.max_weight = max_weight
        self.counts = defaultdict(lambda: 0)

    def get_conditional(self, class_letter, class_nb, context_key):
        '''
        retrieves the raw occurence value from the prior matrix
        '''
        assert class_nb in [0, 1], 'wrong class_nb value %s' % class_nb
        class_key = '%s%s' % (class_letter, class_nb)
        class_keybar = '%s%s' % (class_letter, np.abs(class_nb - 1))
        assert class_key in self.prior_matrix, 'something got very wrong with the prior matrix (key %s) (all %s)' % (class_key, str(list(self.prior_matrix.keys())))

        # get the context raw counting values for
        raw_value_key = self.prior_matrix[class_key].get(context_key)
        raw_value_keybar = self.prior_matrix[class_keybar].get(context_key)

        if raw_value_key is None and raw_value_keybar is None:
            self.counts['both_absent'] += 1
            prob_value = 0.5

        elif raw_value_key is None:
            self.counts['key_absent'] += 1
            prob_value = 1 - self.max_weight

        elif raw_value_keybar is None:
            self.counts['keybar_absent'] += 1
            prob_value = self.max_weight

        else:
            self.counts['both_present'] += 1
            prob_value = raw_value_key / (raw_value_key + raw_value_keybar)

        print('for class key %s found raw %s, for keybar %s found %s, final prob %s' % (class_key, raw_value_key, class_keybar, raw_value_keybar, prob_value))

        return prob_value


class ConditionalPartialRawPrior(ConditionalPrior):
    '''
    prior based on a matrix counting the co-occurrences of classes
    raw= normalization has not been done so we do it on the fly

    the raw matrix is always FULL, meaning it contains a full context
    to get the conditional from a partial context we need to marginalize
    '''
    def __init__(self,
                 matrix_path,
                 nb_classes,
                 context_length,
                 comb_method='simple',
                 use_superclass=False,
                 alpha=None,
                 mode='train',
                 allow_marginalization=True,
                 max_weight=0.6):

        ConditionalPrior.__init__(self,
                                  matrix_path,
                                  nb_classes,
                                  context_length,
                                  comb_method=comb_method,
                                  use_superclass=use_superclass,
                                  alpha=alpha,
                                  mode=mode,
                                  allow_marginalization=allow_marginalization)

        self.max_weight = max_weight
        self.counts = defaultdict(lambda: 0)

    def get_conditional(self, class_letter, class_nb, context_key):
        '''
        retrieves the raw occurence value from the prior matrix
        -> marginalize!
        '''
        assert class_nb in [0, 1], 'wrong class_nb value %s' % class_nb
        class_key = '%s%s' % (class_letter, class_nb)
        class_keybar = '%s%s' % (class_letter, np.abs(class_nb - 1))
        assert class_key in self.prior_matrix, 'something got very wrong with the prior matrix (key %s) (all %s)' % (class_key, str(list(self.prior_matrix.keys())))

        # add all the full context values that match the partial context value
        partial_parts = context_key.split('_')
        assert len(partial_parts) in (self.context_length, self.context_length + 1)

        #print("getting conditional for %s %s context %s" % (class_letter, class_nb, context_key))

        # we add all the values of the contexts that match the partial one
        sum_one = 0
        for full_context, val in self.prior_matrix['%s1' % class_letter].items():
            full_parts = full_context.split('_')
            if len(set(partial_parts) - set(full_parts)) == 0:   # partial is included in full
                #print('full context goes in sum_one %s' % full_context)
                #print(full_context)
                sum_one += val

        sum_zero = 0
        for full_context, val in self.prior_matrix['%s0' % class_letter].items():
            full_parts = full_context.split('_')
            if len(set(partial_parts) - set(full_parts)) == 0:   # partial is included in full
                #print('full context goes in sum_one %s' % full_context)
                #print(full_context)
                sum_zero += val

        # logging
        if sum_one == 0 or sum_one == 0:
            self.counts['one_absent'] += 1

        # raw -> probability
        if class_nb == 0:
            prob_value = sum_zero / (sum_zero + sum_one)

        elif class_nb == 1:
            prob_value = sum_one / (sum_zero + sum_one)


        # raise

        # raw_value_key = self.prior_matrix[class_key].get(context_key)
        # raw_value_keybar = self.prior_matrix[class_keybar].get(context_key)

        # if raw_value_key is None and raw_value_keybar is None:
        #     self.counts['both_absent'] += 1
        #     prob_value = 0.5

        # elif raw_value_key is None:
        #     self.counts['key_absent'] += 1
        #     prob_value = 1 - self.max_weight

        # elif raw_value_keybar is None:
        #     self.counts['keybar_absent'] += 1
        #     prob_value = self.max_weight

        # else:
        #     self.counts['both_present'] += 1
        #     prob_value = raw_value_key / (raw_value_key + raw_value_keybar)

        # print('for class key %s found raw %s, for keybar %s found %s, final prob %s' % (class_key, raw_value_key, class_keybar, raw_value_keybar, prob_value))

        return prob_value


