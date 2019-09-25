import numpy as np

from model.relabel.base import ClassifRelabelator

from config.config import cfg


class AllSkRelabeling(ClassifRelabelator):

    def __init__(self, exp_folder, p, nb_classes):
        self.exp_folder = exp_folder
        self.p = p
        self.nb_classes = nb_classes

    def relabel(self, x_batch, y_batch, y_pred):
        '''
        This relabeling takes _every_ prediction of the model and simply injects it as ground truth
        '''
        y_pred = np.asarray(y_pred)
        y_batch = y_batch[0]

        assert y_pred.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_pred shape %s' % str(y_pred.shape)
        assert y_batch.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_batch shape %s' % str(y_batch.shape)

        # find and sort the relevant values of the outputs (for the missing labels only)
        one_hot_pred = np.where(y_pred > 0.5, 1, -1)
        relabel_batch = np.where(y_batch == 0, one_hot_pred, y_batch)   # consider only the values for missing labels
        nb_added = np.count_nonzero(y_batch == 0)  # actually counting the number of 0 in the original y_true

        # sanity check (check that the added values are only ones and that they are only where there was a 0 before)
        check = np.where(y_batch != relabel_batch, relabel_batch, 0)
        assert np.count_nonzero(check) == nb_added, 'count %s, added %s' % (np.count_nonzero(check), nb_added)     # check we added the correct number of values
        assert np.all(y_batch[np.where(check != 0)] == 0)    # check we added values only where the initial batch was 0

        self.total_added += nb_added

        # write batch to relabel csv
        for i in range(len(relabel_batch)):
            parts = relabel_batch[i]
            img_id = x_batch[1][i][0]

            # for last batch we have duplicates to fill the remaining batch size, we don't want to write those
            if img_id not in self.seen_keys:
                relabel_line = '%s,%s,%s\n' % (img_id, str(parts[0]), ','.join([str(elt) for elt in parts[1:]]))
                self.f_relabel.write(relabel_line)

            self.seen_keys.add(img_id)
