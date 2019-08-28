import os

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

import tensorflow as tf


class PriorCallback(Callback):

    def __init__(self):
        super(PriorCallback, self).__init__()

        # the shape of these 2 variables will change according to batch shape
        # `validate_shape=False` to handle last bactch size
        self.var_y_true = tf.Variable(0., validate_shape=False)

    def on_batch_end(self, batch):
        # evaluate the variables and save them into lists
        # self.targets.append(K.eval(self.var_y_true))
        self.model.update_prior(K.eval(self.var_y_true))
