import os

from model.utils import log


class Relabelator(object):
    '''
    TODO: change the name for clean version
    '''

    def relabel(self, x_batch, y_pred, y_batch):
        raise NotImplementedError


class ClassifRelabelator(Relabelator):
    '''
    common relabelator functions for the classification relabeling strategies
    '''

    def init_step(self, relabel_step_nb):
        # save new targets as file
        self.targets_path = os.path.join(self.exp_folder, 'relabeling', 'relabeling_%s_%sp.csv' % (relabel_step_nb, self.p))
        os.makedirs(os.path.dirname(self.targets_path), exist_ok=True)

        self.total_added = 0
        self.seen_keys = set()
        self.relabel_step = relabel_step_nb

        self.f_relabel = open(self.targets_path, 'w+')

    def finish_step(self, relabel_step):
        assert relabel_step == self.relabel_step

        relabel_logpath = os.path.join(self.exp_folder, 'relabeling', 'log_relabeling.csv')
        with open(relabel_logpath, 'a') as f_log:
            f_log.write('%s,%s,%s\n' % (self.p, self.relabel_step, self.total_added))

        log.printcn(log.OKBLUE, '\tAdded %s labels during relabeling, logging into %s' % (self.total_added, relabel_logpath))
        log.printcn(log.OKBLUE, '\tNew dataset path %s' % (self.targets_path))

        self.f_relabel.close()

    def __del__(self):
        if hasattr(self, 'f_relabel'):
            self.f_relabel.close()
