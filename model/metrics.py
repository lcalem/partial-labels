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


def calculate_map(y_true, y_pred):
    num_classes = y_true.shape[1]
    average_precisions = []

    for index in range(num_classes):
        pred = y_pred[:, index]
        label = y_true[:, index]

        sorted_indices = np.argsort(-pred)
        sorted_pred = pred[sorted_indices]
        sorted_label = label[sorted_indices]

        tp = (sorted_label == 1)
        fp = (sorted_label == 0)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        npos = np.sum(sorted_label)

        recall = tp * 1.0 / npos

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp * 1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        average_precisions.append(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    print(average_precisions)
    mAP = np.mean(average_precisions)

    return mAP
