import numpy as np

from model.relabel.base import ClassifRelabelator

from config.config import cfg


class BaselineRelabeling(ClassifRelabelator):
    '''
    Relabeling strategies a, b and c from the Durand paper
    All from logits

    a: score threshold
    b: score proportion
    c: positive only (with treshold)
    '''

    def __init__(self, exp_folder, p, nb_classes, selection_type, threshold):
        self.exp_folder = exp_folder
        self.p = p
        self.nb_classes = nb_classes

        # check on options
        assert selection_type in ['threshold', 'proportion', 'positives']
        self.selection_type = selection_type
        self.threshold = threshold

    def relabel(self, x_batch, y_batch, y_pred):
        '''
        applied on logits
        '''
        logits = np.asarray(y_pred[1])   # y_pred gives both the output and the logits, the [1] is to take the logits
        y_batch = y_batch[0]

        assert logits.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong logits shape %s' % str(logits.shape)
        assert y_batch.shape == (cfg.BATCH_SIZE, self.nb_classes), 'wrong y_batch shape %s' % str(y_batch.shape)

        relabel_batch = np.copy(y_batch)

        # find the relevant values of the outputs
        if self.selection_type == 'threshold':
            selection_criterion = (y_batch == 0) & ((logits > self.threshold) | (logits < - self.threshold))
            relabel_batch = np.where(selection_criterion, np.sign(logits), y_batch)

        elif self.selection_type == 'proportion':
            pass

        elif self.selection_type == 'positives':
            selection_criterion = (y_batch == 0) & (logits > self.threshold)
            relabel_batch = np.where(selection_criterion, 1, y_batch)

        nb_added = np.sum(selection_criterion)

        # sanity checks (check that the added values are only ones and that they are only where there was a 0 before)
        check = np.where(y_batch != relabel_batch, relabel_batch, 0)
        assert np.count_nonzero(check) == nb_added, 'count %s, added %s' % (np.count_nonzero(check), nb_added)     # check we added the correct number of values
        assert np.all(y_batch[np.where(check != 0)] == 0)    # check we added values only where the initial batch was 0

        self.total_added += nb_added
        relabel_batch = relabel_batch.astype(np.int32)

        # print('y batch')
        # print(y_batch)

        # print('logits')
        # print(logits)

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
