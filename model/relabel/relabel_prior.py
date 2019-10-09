import os

import numpy as np

from model import priors
from model.relabel.base import ClassifRelabelator
from model.utils import log

from config.config import cfg

from pprint import pprint


class PriorRelabeling(ClassifRelabelator):

    def __init__(self, exp_folder, p, nb_classes, selection_params):
        '''
        Selection types:
            - proportion_ones:
                - selects _threshold_ percent of the y_f values that match sk > 0.5 and put them as ones
            - proportion_yf:
                - selection from yf values (prop_ones and prop_zeros)
                - relabel value depends on the chunk of yf selected (highest values -> +1, lowest -> -1)
        '''
        self.exp_folder = exp_folder
        self.p = p
        self.nb_classes = nb_classes
        self.prior = self.load_prior(cfg.RELABEL.PRIOR)

        selection_params = {k.lower(): v for k, v in selection_params.items()}
        self.selection_type = selection_params['type']
        assert self.selection_type in ['proportion_ones', 'proportion_yf']
        self.selection_params = selection_params

    def load_prior(self, name):
        if name == 'conditional':
            prior_path = cfg.RELABEL.PRIOR_PATH
            prior_path = prior_path.replace('$PROP', str(self.p))
            return priors.ConditionalPrior(prior_path, nb_classes=self.nb_classes)

    def relabel(self, x_batch, y_batch, y_pred):
        '''
        y_pred: (1, batch_size, K) -> have to take y_pred[0]
        x_batch is used for image ids

        Steps for relabeling:
        - compute the relevant prior given the ground truth
        - combine the prior with the predictions (-> y_f)
        - Use this y_f and the original predictions y_pred to picj the relabeled examples

        yf: prior-weighted outputs
        y_true: true batch (with zeros where the label is missing)

        1. find the values of yf for which there is a missing label at the same index in the y_true batch
        2. order those values and get the corresponding indexes
        '''

        y_true = y_batch[0]
        y_pred = np.asarray(y_pred[0])  # y_pred gives both the output and the logits, the [0] is to take the predictions

        # print('shape of y_pred %s' % str(y_pred.shape))
        p_k = self.prior.compute_pk(y_true)
        y_f = self.prior.combine(y_pred, p_k)

        # relabeling, nb_added = self.prior.pick_relabel(y_pred, y_k, y_batch[0])  # (BS, K)

        assert y_pred.shape == (cfg.BATCH_SIZE, self.nb_classes),   'wrong y_pred shape %s' % str(y_pred.shape)
        assert y_f.shape == (cfg.BATCH_SIZE, self.nb_classes),      'wrong yk shape %s' % str(y_k.shape)
        assert y_true.shape == (cfg.BATCH_SIZE, self.nb_classes),   'wrong y_true shape %s' % str(y_true.shape)

        relabel_batch = np.copy(y_true)
        nb_added_pos = 0
        nb_added_neg = 0

        # find and sort the relevant values of the outputs
        if self.selection_type == 'proportion_ones':

            relevant_yf = np.where((y_true == 0) & (y_pred > 0.5), y_f, 0)   # consider only the values for missing labels and for which the initial prediction is > 0.5
            sorted_indexes = np.argsort(relevant_yf, axis=None)   # the indexes are flattened
            sorted_values = [relevant_yf[i // self.nb_classes][i % self.nb_classes] for i in sorted_indexes]

            # take the best values and find corresponding indexes
            nb_ok_indexes = int((np.count_nonzero(sorted_values)) * self.selection_params['threshold'])
            final_indexes = [(i // self.nb_classes, i % self.nb_classes) for i in sorted_indexes[len(sorted_values) - nb_ok_indexes: len(sorted_values)]]

            # put the selected values as 1 in the relabel batch
            xs = [elt[0] for elt in final_indexes]
            ys = [elt[1] for elt in final_indexes]
            relabel_batch[(xs, ys)] = 1

            nb_added_pos = nb_ok_indexes

        elif self.selection_type == 'proportion_yf':
            relevant_yf = np.where(y_true == 0, y_f, 0)   # consider only the values for missing labels
            sorted_indexes = np.argsort(relevant_yf, axis=None)   # the indexes are flattened
            sorted_values = [relevant_yf[i // self.nb_classes][i % self.nb_classes] for i in sorted_indexes]

            # highest values -> +1
            nb_positive_indexes = int((np.count_nonzero(sorted_values)) * self.selection_params['prop_pos'])
            positive_indexes = [(i // self.nb_classes, i % self.nb_classes) for i in sorted_indexes[len(sorted_values) - nb_positive_indexes: len(sorted_values)]]

            # put the selected values as 1 in the relabel batch
            xs_pos = [elt[0] for elt in positive_indexes]
            ys_pos = [elt[1] for elt in positive_indexes]
            relabel_batch[(xs_pos, ys_pos)] = 1

            # lowest values -> -1
            nb_zeros = len(sorted_values) - np.count_nonzero(sorted_values)
            nb_negative_indexes = int((np.count_nonzero(sorted_values)) * self.selection_params['prop_neg'])
            negative_indexes = [(i // self.nb_classes, i % self.nb_classes) for i in sorted_indexes[nb_zeros: nb_zeros + nb_negative_indexes]]

            xs_neg = [elt[0] for elt in negative_indexes]
            ys_neg = [elt[1] for elt in negative_indexes]
            relabel_batch[(xs_neg, ys_neg)] = -1

            nb_added_pos = nb_positive_indexes
            nb_added_neg = nb_negative_indexes

        nb_added = nb_added_pos + nb_added_neg

        # sanity check (check that the added values are only ones and that they are only where there was a 0 before)
        check = np.where(y_true != relabel_batch, relabel_batch, 0)
        assert np.count_nonzero(check) == nb_added, 'count %s, added %s' % (np.count_nonzero(check), nb_added)     # check we added the correct number of values
        assert np.all(y_true[np.where(check == 1)] == 0)    # check we added values only where the initial batch was 0

        self.total_added += nb_added
        self.positive_added += nb_added_pos
        self.negative_added += nb_added_neg

        # print(nb_added_pos)
        # print(nb_added_neg)

        # print('y batch')
        # print(y_batch)

        # print('y pred')
        # print(y_pred)

        # print('pk')
        # print(p_k)

        # print('y_k')
        # print(y_k)

        # print('relabeling')
        # print(relabel_batch)

        # raise

        # write batch to relabel csv
        for i in range(len(relabel_batch)):
            parts = relabel_batch[i]
            img_id = x_batch[1][i][0]

            # for last batch we have duplicates to fill the remaining batch size, we don't want to write those
            if img_id not in self.seen_keys:
                relabel_line = '%s,%s,%s\n' % (img_id, str(parts[0]), ','.join([str(elt) for elt in parts[1:]]))
                self.f_relabel.write(relabel_line)

            self.seen_keys.add(img_id)
