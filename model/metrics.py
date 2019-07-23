import numpy as np
import tensorflow as tf

from sklearn import metrics


class BaseMetric(object):

    def __init__(self):
        self.__name__ = self.name

    def __call__(self, y_true, y_pred, **kwargs):
        return self.compute_metric(y_true, y_pred, **kwargs)

    def compute_metric(self, y_true, y_pred):
        raise NotImplementedError


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
        print('type true %s, type pred %s' % (type(y_true), type(y_pred)))
        return metrics.average_precision_score(y_true, y_pred, average=None)

