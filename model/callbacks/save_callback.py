import os

from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import Callback


class SaveModel(Callback):

    def __init__(self, exp_folder, prop, relabel_step=None, verbose=True):
        '''
        prop = proportion of known labels for experiment
        '''

        self.exp_folder = exp_folder
        self.prop = prop
        self.verbose = verbose
        self.relabel_step = relabel_step

    def on_epoch_end(self, epoch, logs=None):

        suffix = '{prop:02d}_{epoch:03d}'.format(prop=self.prop, epoch=epoch + 1)
        if self.relabel_step is not None:
            suffix += '_{relabel:02d}'.format(relabel=self.relabel_step)

        save_file = os.path.join(self.exp_folder, 'model_%s.h5' % suffix)
        try:
            if self.verbose:
                print('\nTrying to save model @epoch=%03d to %s' % (epoch + 1, save_file))

            save_model(self.model, save_file)
        except Exception as e:
            save_file = os.path.join(self.exp_folder, 'weights_%s.h5' % suffix)
            print("Couldn't save model, saving weights instead at %s" % save_file)
            self.model.save_weights(save_file)
