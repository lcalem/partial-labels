

class BaseMetric(object):

    def __init__(self):
        self.__name__ = self.name

    def __call__(self, y_true, y_pred, **kwargs):
        return self.compute_metric(y_true, y_pred, **kwargs)

    def compute_metric(self, y_true, y_pred):
        raise NotImplementedError
