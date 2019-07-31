
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from model.utils import log
from config.config import cfg

from experiments.launch import parse_options_file
from config import config_utils


class BaseModel(object):
    '''
    Base class for different models
    '''
    def __init__(self, exp_folder, n_classes):
        self.exp_folder = exp_folder

        self.n_classes = n_classes
        self.input_shape = (cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, cfg.IMAGE.N_CHANNELS)
        self.verbose = cfg.VERBOSE

        print("Init input_shape %s" % str(self.input_shape))

    def log(self, msg):
        if self.verbose:
            log.printcn(log.HEADER, msg)

    def load(self, checkpoint_path, custom_objects=None):
        self.model = load_model(checkpoint_path, custom_objects=custom_objects)

    def load_weights(self, weights_path, by_name=False, build_args=None, config_file=None):
        if build_args is None:
            build_args = dict()

        if config_file:
            print("Loading options")
            options = parse_options_file(config_file)
            config_utils.update_config(options)

        self.build(**build_args)
        self.model.load_weights(weights_path, by_name=by_name)

    def build(self):
        raise NotImplementedError

    def train(self, data_tr, steps_per_epoch, cb_list, dataset_val=None):

        print("Training with %s callbacks" % len(cb_list))
        kwargs = {
            'steps_per_epoch': steps_per_epoch,
            'epochs': cfg.TRAINING.N_EPOCHS,
            'callbacks': cb_list,
            'shuffle': cfg.DATASET.SHUFFLE,
            'initial_epoch': 0
        }

        if cfg.MULTIP.USE_MULTIPROCESS:
            kwargs['use_multiprocessing'] = cfg.MULTIP.USE_MULTIPROCESS
            kwargs['max_queue_size'] = cfg.MULTIP.MAX_QUEUE_SIZE
            kwargs['workers'] = cfg.MULTIP.N_WORKERS

        if dataset_val:
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

