import numpy as np

from model.relabel.base import ClassifRelabelator

from config.config import cfg


class VisualRelabeling(ClassifRelabelator):

    def __init__(self, exp_folder, p, nb_classes, selection_params):
        self.exp_folder = exp_folder
        self.p = p
        self.nb_classes = nb_classes

        self.selection_params = {k.lower(): v for k, v in selection_params.items()}

    def relabel(self, x_batch, y_batch, y_pred):
        '''
        selection: proportion of positives and negatives
        relabel value: forced +1 or -1 depending on which side of the sorted_values we are (not directly the y_v values even if it should correlate)
        '''
        y_pred = np.asarray(y_pred[0])
        y_batch = y_batch[0]

        assert y_pred.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_pred shape %s' % str(y_pred.shape)
        assert y_batch.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_batch shape %s' % str(y_batch.shape)

        relabel_batch = np.copy(y_batch)

        # select and sort the relevant values of the outputs
        relevant_yv = np.where(y_batch == 0, y_pred, 0)   # consider only the values for missing labels
        sorted_indexes = np.argsort(relevant_yv, axis=None)   # the indexes are flattened
        sorted_values = [relevant_yv[i // self.nb_classes][i % self.nb_classes] for i in sorted_indexes]

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

        # sanity check (check that the added values are only ones and that they are only where there was a 0 before)
        check = np.where(y_batch != relabel_batch, relabel_batch, 0)
        assert np.count_nonzero(check) == nb_positive_indexes + nb_negative_indexes     # check we added the correct number of values
        assert np.all(y_batch[np.where(check == 1)] == 0)    # check we added values only where the initial batch was 0

        self.total_added += nb_positive_indexes + nb_negative_indexes
        self.positive_added += nb_positive_indexes
        self.negative_added += nb_negative_indexes

        print('y batch')
        print(y_batch)

        print('y pred')
        print(y_pred)

        print('relabeling')
        print(relabel_batch)

        raise

        # write batch to relabel csv
        for i in range(len(relabel_batch)):
            parts = relabel_batch[i]
            img_id = x_batch[1][i][0]

            # for last batch we have duplicates to fill the remaining batch size, we don't want to write those
            if img_id not in self.seen_keys:
                relabel_line = '%s,%s,%s\n' % (img_id, str(parts[0]), ','.join([str(elt) for elt in parts[1:]]))
                self.f_relabel.write(relabel_line)

            self.seen_keys.add(img_id)
