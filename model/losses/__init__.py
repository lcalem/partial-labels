import tensorflow as tf

from model.losses.bce import BCE
from model.losses.partialbce import PartialBCE

from model.losses.base import BaseLoss


class NoopLoss(BaseLoss):

    def compute_loss(self, y_true, y_pred, trace=False):
        '''
        returns 0 no matter what
        '''
        return tf.math.square(0.0)


LOSSES = {
    'noop': NoopLoss,
    'bce': BCE,
    'partialbce': PartialBCE
}


def get_losses(loss_names):
    '''
    Return instanciated losses
    '''
    return [get_loss(name) for name in loss_names]


def get_loss(loss_name, params=None):
    '''
    ugly
    '''
    if params is None:
        params = dict()

    if loss_name == 'bce':
        return BCE()
    elif loss_name == 'partialbce':
        return PartialBCE(**params)

