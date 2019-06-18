
from tensorflow.keras.models import load_model

from model.utils import log


class BaseModel(object):
    '''
    Base class for different models
    '''
    def __init__(self):
        pass

    def log(self, msg):
        if self.verbose:
            log.printcn(log.HEADER, msg)

    def load(self, checkpoint_path, custom_objects=None):
        self.model = load_model(checkpoint_path, custom_objects=custom_objects)

    def load_weights(self, weights_path, by_name=False):
        self.build()
        self.model.load_weights(weights_path, by_name=by_name)

    def build(self):
        raise NotImplementedError

    def train(self, data_tr, steps_per_epoch, model_folder, n_epochs, cb_list, n_workers=2):

        print("Training with %s callbacks" % len(cb_list))

        self.model.fit_generator(data_tr,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=n_epochs,
                                 callbacks=cb_list,
                                 use_multiprocessing=False,
                                 max_queue_size=10,
                                 workers=n_workers,
                                 initial_epoch=0)

    def predict(self, data):
        return self.model.predict(data)
