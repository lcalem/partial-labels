import os

from model.relabel.base import Relabelator
from model.utils import log

from model import priors

from config.config import cfg


class BaselineRelabeling(Relabelator):

    def __init__(self, exp_folder, p):
        self.exp_folder = exp_folder
        self.p = p
        self.prior = self.load_prior(cfg.RELABEL.PRIOR)

    def load_prior(self, name):
        if name == 'conditional':
            prior_path = cfg.RELABEL.PRIOR_PATH
            prior_path = prior_path.replace('$PROP', str(self.p))
            return priors.ConditionalPrior(prior_path)

    def init_step(self, relabel_step_nb):
        # save new targets as file
        self.targets_path = os.path.join(self.exp_folder, 'relabeling', 'relabeling_%s_%sp.csv' % (relabel_step_nb, self.p))
        os.makedirs(os.path.dirname(self.targets_path), exist_ok=True)

        self.total_added = 0
        self.seen_keys = set()
        self.relabel_step = relabel_step_nb

        self.f_relabel = open(self.targets_path, 'w+')

    def finish_step(self, ):
        relabel_logpath = os.path.join(self.exp_folder, 'relabeling', 'log_relabeling.csv')
        with open(relabel_logpath, 'a') as f_log:
            f_log.write('%s,%s,%s\n' % (self.p, self.relabel_step, self.total_added))

        log.printcn(log.OKBLUE, '\tAdded %s labels during relabeling, logging into %s' % (self.total_added, relabel_logpath))
        log.printcn(log.OKBLUE, '\tNew dataset path %s' % (self.targets_path))

        self.f_relabel.close()

    def relabel(self, x_batch, y_batch, y_pred):
        # print('shape of y_pred %s' % str(y_pred.shape))
        p_k = self.prior.compute_pk(y_batch[0])

        y_k = self.prior.combine(y_pred, p_k)
        relabeling, nb_added = self.prior.pick_relabel(y_pred, y_k, y_batch[0])  # (BS, K)
        self.total_added += nb_added

        # print('y batch')
        # print(y_batch)

        # print('y pred')
        # print(y_pred)

        # print('pk')
        # print(p_k)

        # print('y_k')
        # print(y_k)

        # print('relabeling')
        # print(relabeling)

        # write batch to relabel csv
        for i in range(len(relabeling)):
            parts = relabeling[i]
            img_id = x_batch[1][i][0]

            # for last batch we have duplicates to fill the remaining batch size, we don't want to write those
            if img_id not in self.seen_keys:
                relabel_line = '%s,%s,%s\n' % (img_id, str(parts[0]), ','.join([str(elt) for elt in parts[1:]]))
                self.f_relabel.write(relabel_line)

            self.seen_keys.add(img_id)

    def __del__(self):
        self.f_relabel.close()
