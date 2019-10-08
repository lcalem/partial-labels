import os

import numpy as np

from model import priors
from model.relabel.base import Relabelator
from model.utils import log

from config.config import cfg

from pprint import pprint


class PriorRelabeling(Relabelator):

    def __init__(self, exp_folder, p, nb_classes, selection_type='proportion_ones', threshold=0.33):
        '''
        Selection types:
            - proportion_ones: selects _threshold_ percent of the y_k values that match sk > 0.5 and put them as ones
            - proportion: selects _threshold_ / 2 percent of the y_k values and put them as -1 or +1
        '''
        self.exp_folder = exp_folder
        self.p = p
        self.nb_classes = nb_classes
        self.prior = self.load_prior(cfg.RELABEL.PRIOR)

        assert selection_type in ['proportion_ones', 'proportion']
        assert 0.0 <= threshold <= 1.0
        self.selection_type = selection_type
        self.threshold = threshold

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
        - combine the prior with the predictions (-> y_k)
        - Use this y_k and the original predictions y_pred to picj the relabeled examples

        yk: prior-weighted outputs
        y_true: true batch (with zeros where the label is missing)

        1. find the values of yk for which there is a missing label at the same index in the y_true batch
        2. order those values and get the corresponding indexes
        '''

        y_true = y_batch[0]
        y_pred = np.asarray(y_pred[0])  # y_pred gives both the output and the logits, the [0] is to take the predictions

        # print('shape of y_pred %s' % str(y_pred.shape))
        p_k = self.prior.compute_pk(y_true)
        y_k = self.prior.combine(y_pred, p_k)

        # relabeling, nb_added = self.prior.pick_relabel(y_pred, y_k, y_batch[0])  # (BS, K)

        assert y_pred.shape == (cfg.BATCH_SIZE, self.nb_classes),   'wrong y_pred shape %s' % str(y_pred.shape)
        assert y_k.shape == (cfg.BATCH_SIZE, self.nb_classes),      'wrong yk shape %s' % str(y_k.shape)
        assert y_true.shape == (cfg.BATCH_SIZE, self.nb_classes),   'wrong y_true shape %s' % str(y_true.shape)

        relabel_batch = np.copy(y_true)
        nb_added_pos = 0
        nb_added_neg = 0

        # find and sort the relevant values of the outputs
        if self.selection_type == 'proportion_ones':

            relevant_yk = np.where((y_true == 0) & (y_pred > 0.5), y_k, 0)   # consider only the values for missing labels and for which the initial prediction is > 0.5
            sorted_indexes = np.argsort(relevant_yk, axis=None)   # the indexes are flattened
            sorted_values = [relevant_yk[i // self.nb_classes][i % self.nb_classes] for i in sorted_indexes]

            # take the best values and find corresponding indexes
            nb_ok_indexes = int((np.count_nonzero(sorted_values)) * self.threshold)
            final_indexes = [(i // self.nb_classes, i % self.nb_classes) for i in sorted_indexes[len(sorted_values) - nb_ok_indexes: len(sorted_values)]]

            # put the selected values as 1 in the relabel batch
            xs = [elt[0] for elt in final_indexes]
            ys = [elt[1] for elt in final_indexes]
            relabel_batch[(xs, ys)] = 1

            nb_added_pos = nb_ok_indexes

        elif self.selection_type == 'proportion':
            relevant_yk = np.where(y_true == 0, y_k, 0)   # consider only the values for missing labels
            sorted_indexes = np.argsort(relevant_yk, axis=None)   # the indexes are flattened
            sorted_values = [relevant_yk[i // self.nb_classes][i % self.nb_classes] for i in sorted_indexes]

            # highest values -> +1
            nb_positive_indexes = int((np.count_nonzero(sorted_values)) * (self.threshold / 2))
            positive_indexes = [(i // self.nb_classes, i % self.nb_classes) for i in sorted_indexes[len(sorted_values)-nb_positive_indexes: len(sorted_values)]]

            # put the selected values as 1 in the relabel batch
            xs_pos = [elt[0] for elt in positive_indexes]
            ys_pos = [elt[1] for elt in positive_indexes]
            relabel_batch[(xs_pos, ys_pos)] = 1

            # lowest values -> -1
            nb_zeros = len(sorted_values) - np.count_nonzero(sorted_values)
            nb_negative_indexes = int((np.count_nonzero(sorted_values)) * (self.threshold / 2))
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

    def init_step(self, relabel_step_nb):
        # save new targets as file
        self.targets_path = os.path.join(self.exp_folder, 'relabeling', 'relabeling_%s_%sp.csv' % (relabel_step_nb, self.p))
        os.makedirs(os.path.dirname(self.targets_path), exist_ok=True)

        self.total_added = 0
        self.positive_added = 0
        self.negative_added = 0

        self.seen_keys = set()
        self.relabel_step = relabel_step_nb

        self.f_relabel = open(self.targets_path, 'w+')

    def finish_step(self, relabel_step):
        assert relabel_step == self.relabel_step

        relabel_logpath = os.path.join(self.exp_folder, 'relabeling', 'log_relabeling.csv')
        with open(relabel_logpath, 'a') as f_log:
            f_log.write('%s,%s,%s,%s,%s\n' % (self.p, self.relabel_step, self.total_added, self.positive_added, self.negative_added))

        log.printcn(log.OKBLUE, '\tAdded %s labels during relabeling, logging into %s' % (self.total_added, relabel_logpath))
        log.printcn(log.OKBLUE, '\tNew dataset path %s' % (self.targets_path))

        self.f_relabel.close()

    def __del__(self):
        if hasattr(self, 'f_relabel'):
            self.f_relabel.close()

