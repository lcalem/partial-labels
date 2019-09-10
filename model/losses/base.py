

class BaseLoss(object):

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return self.compute_loss(y_true, y_pred)

    def compute_loss(self, y_true, y_pred):
        raise NotImplementedError
