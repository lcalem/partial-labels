import numpy as np
import tensorflow as tf

from sklearn import metrics

import numpy as np

from model.metrics import BaseMetric


class MAP(BaseMetric):
    name = 'map'

    def compute_metric(self, y_true, y_pred):
        '''
        returns unweighted mean of the AP for each class
        '''
        print('y_true is a %s' % (type(y_true)))
        print('y_pred is a %s' % (type(y_pred)))

        return metrics.average_precision_score(y_true, y_pred)

    def compute_separated(self, y_true, y_pred):
        '''
        computes the AP for each class and returns it as is
        '''
        # print('type true %s (%s), type pred %s (%s)' % (type(y_true), np.array(y_true).shape, type(y_pred), y_pred.shape))
        print('there is %s nans' % tf.count_nonzero(tf.is_nan(y_pred)))
        return metrics.average_precision_score(y_true[0], y_pred, average=None)

