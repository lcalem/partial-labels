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
        y_pred is always 0/1
        '''

        weights = tf.where(tf.equal(y_true, 0), tf.zeros_like(y_true), tf.ones_like(y_true))
        y_true_01 = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_true), y_true)
        y_true_01 = tf.cast(y_true_01, tf.float32)

        bce = tf.keras.backend.binary_crossentropy(y_true_01, y_pred)
        weighted_bce = tf.multiply(bce, tf.cast(weights, tf.float32))

        # total = bce.shape.as_list()[-1]
        nonzeros = tf.math.count_nonzero(weighted_bce, -1)
        batch_sum = tf.math.reduce_sum(weighted_bce, axis=-1)
        # final_bce = batch_sum / total
        final_bce = batch_sum / tf.cast(nonzeros, tf.float32)

        if trace:
            return {
                'final_bce': final_bce,
                'weighted_bce': weighted_bce,
                'bce': bce,
                'weights': weights,
                'y_true_01': y_true_01,
                'nonzeros': nonzeros
            }

        else:
            return final_bce


class PartialBCE(BaseLoss):

    def __init__(self, prop):

        alpha = -4.45
        beta = 5.45
        gamma = 1

        self.norm_factor = alpha * prop + beta   # TODO: actual pow but for now gamma is one so osef

    def compute_loss(self, y_true, y_pred, trace=False):
        '''
        y_true is a (batch_size, n_classes) tensor containing -1 (no), 0 (unknown) and 1 (yes)
        y_pred is always 0/1
        '''

        weights = tf.where(tf.equal(y_true, 0), tf.zeros_like(y_true), tf.ones_like(y_true))
        y_true_01 = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_true), y_true)
        y_true_01 = tf.cast(y_true_01, tf.float32)

        bce = tf.keras.backend.binary_crossentropy(y_true_01, y_pred)
        weighted_bce = tf.multiply(bce, tf.cast(weights, tf.float32))

        # total = bce.shape.as_list()[-1]
        nonzeros = tf.math.count_nonzero(weighted_bce, -1)
        batch_sum = tf.math.reduce_sum(weighted_bce, axis=-1)

        final_bce = self.norm_factor * (batch_sum / tf.cast(nonzeros, tf.float32))

        if trace:
            return {
                'final_bce': final_bce,
                'weighted_bce': weighted_bce,
                'bce': bce,
                'weights': weights,
                'y_true_01': y_true_01,
                'nonzeros': nonzeros
            }

        else:
            return final_bce


LOSSES = {
    'bce': BCE,
    'partialbce': PartialBCE
}


def get_losses(loss_names):
    '''
    Return instanciated losses
    '''
    return [get_loss(name) for name in loss_names]


def get_loss(loss_name, params):
    '''
    ugly
    '''
    if loss_name == 'bce':
        return BCE()
    elif loss_name == 'partialbce':
        return PartialBCE(**params)

