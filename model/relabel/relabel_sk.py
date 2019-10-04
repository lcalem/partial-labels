import numpy as np

from model.relabel.base import ClassifRelabelator

from config.config import cfg


class SkRelabeling(ClassifRelabelator):

    def __init__(self, exp_folder, p, nb_classes):
        self.exp_folder = exp_folder
        self.p = p
        self.nb_classes = nb_classes

    def relabel(self, x_batch, y_batch, y_pred):
        '''

        '''
        y_pred = np.asarray(y_pred)
        y_batch = y_batch[0]

        assert y_pred.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_pred shape %s' % str(y_pred.shape)
        assert y_batch.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_batch shape %s' % str(y_batch.shape)

        relabel_batch = np.copy(y_batch)

        # find and sort the relevant values of the outputs
        relevant_sk = np.where((y_batch == 0) & (y_pred > 0.5), y_pred, 0)   # consider only the values for missing labels and for which the prediction is > 0.5
        sorted_indexes = np.argsort(relevant_sk, axis=None)   # the indexes are flattened
        sorted_values = [relevant_sk[i // self.nb_classes][i % self.nb_classes] for i in sorted_indexes]

        # take the best values and find corresponding indexes
        nb_ok_indexes = int((np.count_nonzero(sorted_values)) * 0.33)
        final_indexes = [(i // self.nb_classes, i % self.nb_classes) for i in sorted_indexes[len(sorted_values) - nb_ok_indexes: len(sorted_values)]]

        # put the selected values as 1 in the relabel batch
        xs = [elt[0] for elt in final_indexes]
        ys = [elt[1] for elt in final_indexes]
        relabel_batch[(xs, ys)] = 1

        # sanity check (check that the added values are only ones and that they are only where there was a 0 before)
        check = np.where(y_batch != relabel_batch, relabel_batch, 0)
        assert np.count_nonzero(check) == nb_ok_indexes     # check we added the correct number of values
        assert np.all(check[np.where(check != 0)] == 1)     # check we only added ones
        assert np.all(y_batch[np.where(check == 1)] == 0)    # check we added values only where the initial batch was 0

        self.total_added += nb_ok_indexes

        # print('y batch')
        # print(y_batch)

        # print('y pred')
        # print(y_pred)

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
