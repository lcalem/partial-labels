import os
import yaml

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from model.utils import log
from config.config import cfg

from config import config_utils

from pprint import pprint


class BaseModel(object):
    '''
    Base class for different models
    '''

    def log(self, msg):
        if self.verbose:
            log.printcn(log.HEADER, msg)

    def load(self, checkpoint_path, custom_objects=None):
        self.model = load_model(checkpoint_path, custom_objects=custom_objects)

    def load_weights(self, weights_path, by_name=False, load_config=True):
        if load_config:
            folder = os.path.abspath(os.path.dirname(weights_path))
            self.load_config(os.path.join(folder, 'config.yaml'))

        self.build()
        self.model.load_weights(weights_path, by_name=by_name)

    def load_config(self, config_path):
        print("Loading options")
        # options = parse_options_file(config_file)
        with open(config_path, 'r') as f_in:
            loaded_conf = yaml.load(f_in)

        config_utils.update_config(loaded_conf)
        pprint(cfg)

    def build(self):
        raise NotImplementedError

    def train(self, data_tr, steps_per_epoch, cb_list, n_epochs, dataset_val=None):

        print("Training with %s callbacks" % len(cb_list))
        kwargs = {
            'steps_per_epoch': steps_per_epoch,
            'epochs': n_epochs,
            'callbacks': cb_list,
            'shuffle': cfg.DATASET.SHUFFLE,
            'initial_epoch': 0
        }

        if cfg.MULTIP.USE_MULTIPROCESS:
            kwargs['use_multiprocessing'] = cfg.MULTIP.USE_MULTIPROCESS
            kwargs['max_queue_size'] = cfg.MULTIP.MAX_QUEUE_SIZE
            kwargs['workers'] = cfg.MULTIP.N_WORKERS

        if dataset_val and not cfg.TRAINING.SKIP_VAL:
            kwargs['validation_data'] = dataset_val

        self.model.fit_generator(data_tr, **kwargs)

    def predict(self, data):
        return self.model.predict(data)

    def build_optimizer(self):
        '''
        TODO: something better than an ugly switch <3
        '''
        if cfg.TRAINING.OPTIMIZER == 'rmsprop':
            return RMSprop(lr=cfg.TRAINING.START_LR)
        elif cfg.TRAINING.OPTIMIZER == 'adam':
            return Adam(lr=cfg.TRAINING.START_LR)
        elif cfg.TRAINING.OPTIMIZER == 'sgd':
            return SGD(lr=cfg.TRAINING.START_LR)

        raise Exception('Unknown optimizer %s' % cfg.TRAINING.OPTIMIZER)

