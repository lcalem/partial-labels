import os

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

import tensorflow as tf

from config.config import cfg


class RelabelCallback(Callback):

    def __init__(self, dataset_train, exp_folder, relabel_method, prop):
        self.exp_folder = exp_folder
        self.method = relabel_method
        self.prop = prop
        self.trigger_modulo = cfg.RELABEL.TRIGGER_EPOCH

        self.dataset_train = dataset_train
        Callback.__init__(self)

    def on_epoch_end(self, epoch):

        if epoch % self.trigger_modulo == 0:

            # evaluate
            predictions = self.model.predict(self.xy.x_train, batch_size=p.bath_size)

            # create new targets

            # save new targets as file
            targets_path = os.path.join(self.exp_folder, 'relabeling', 'relabeling_e%s_%s_%sp.csv' % (epoch, self.method, self.prop))
            os.makedirs(os.path.dirname(targets_path), exist_ok=True)

            # update dataset
            self.dataset_train.update_targets(targets_path)
