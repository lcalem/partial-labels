
from tensorflow.keras.models import load_model

from model.utils import log
from model.utils.config import cfg


class BaseModel(object):
    '''
    Base class for different models
    '''
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.input_shape = (cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, cfg.IMAGE.N_CHANNELS)
        self.verbose = cfg.VERBOSE

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

    def train(self, data_tr, steps_per_epoch, cb_list, dataset_val=None):

        print("Training with %s callbacks" % len(cb_list))

        kwargs = {
            'steps_per_epoch': steps_per_epoch,
            'epochs': cfg.TRAINING.N_EPOCHS,
            'callbacks': cb_list,
            'use_multiprocessing': cfg.MULTIP.USE_MULTIPROCESS,
            'max_queue_size': cfg.MULTIP.MAX_QUEUE_SIZE,
            'workers': cfg.MULTIP.N_WORKERS,
            'shuffle': cfg.DATASET.SHUFFLE,
            'initial_epoch': 0
        }

        if dataset_val:
            kwargs['validation_data'] = dataset_val

        self.model.fit_generator(data_tr, **kwargs)

    def predict(self, data):
        return self.model.predict(data)
