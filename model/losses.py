import tensorflow as tf


class BaseLoss(object):

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return self.compute_loss(y_true, y_pred)

    def compute_loss(self, y_true, y_pred):
        raise NotImplementedError


class BCE(BaseLoss):

    def compute_loss(self, y_true, y_pred, trace=False):
        '''
        y_true is a (batch_size, n_classes) tensor containing -1 (no), 0 (unknown) and 1 (yes)
        '''

        weights = tf.where(tf.equal(y_true, 0), tf.zeros_like(y_true), tf.ones_like(y_true))
        y_true_01 = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_true), y_true)
        y_true_01 = tf.cast(y_true_01, tf.float32)

        bce = tf.keras.backend.binary_crossentropy(y_true_01, y_pred)
        weighted_bce = tf.multiply(bce, tf.cast(weights, tf.float32))
        final_bce = tf.reduce_sum(weighted_bce)

        if trace:
            return {
                'final_bce': final_bce,
                'weighted_bce': weighted_bce,
                'bce': bce,
                'weights': weights,
                'y_true_01': y_true_01
            }

        else:
            return final_bce


LOSSES = {
    'bce': BCE
}


def get_losses(self, loss_names):
    '''
    Return instanciated losses
    '''
    return [get_loss(name) for name in loss_names]


def get_loss(self, loss_name):
    '''
    '''
    return LOSSES[loss_name]()
