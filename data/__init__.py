
import math

import numpy as np

from tensorflow.keras.utils import Sequence

from config.config import cfg


class Dataset(Sequence):

    # override this to add more keys
    supported_keys = ()
    supported_modes = ()

    def __init__(self,
                 dataset_path,
                 batch_size,
                 mode,
                 x_keys,
                 y_keys,
                 p=None):
        '''
        Only multilabel for now
        - dataset_path: for example blabla/VOCdevkit/VOC2007/
        - mode: train / val / trainval / test -> file to be loaded
        - p: known labels proportion -> will be used to open the correct partial dataset file (only for training)


        Note: shuffle is managed in the fit_generator, not at all here
        '''
        assert mode in self.supported_modes, 'Unknown subset %s' % str(mode)
        if p is not None and not mode.startswith('train'):
            raise Exception('prop only for training')

        self.dataset_path = dataset_path
        self.mode = mode
        self.p = p

        self.x_keys = x_keys
        self.y_keys = y_keys
        self.all_keys = set(x_keys + y_keys)
        assert all([k in self.supported_keys for k in self.all_keys]), 'Unexpected key in %s' % str(self.all_keys)

        self.img_size = (cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, cfg.IMAGE.NB_CHANNELS)

        # loading samples
        self.sample_ids = self.load_samples()
        self.nb_samples = len(self.sample_ids)

        self.batch_size = self.nb_samples if batch_size == 'all' else batch_size

        # prior loading
        self.init_cooc()

        Sequence.__init__(self)

    def __len__(self):
        '''
        Should give off the number of batches
        '''
        return math.ceil(self.nb_samples / self.batch_size)

    def __getitem__(self, batch_idx):
        '''
        required by the Sequence api
        return a batch, composed of x_batch and y_batch
        TODO: data augmentation should take place here
        '''
        batch_data = self._load_data(batch_idx)

        # Convert the dictionary of samples to a list for x and y
        x_batch = []
        for key in self.x_keys:
            x_batch.append(batch_data[key])

        y_batch = []
        for key in self.y_keys:
            y_batch.append(batch_data[key])

        return x_batch, y_batch

    def _load_data(self, batch_idx):
        '''
        Used by Dataset.__getitem__ during batching
        '''
        data_dict = {}
        for key in self.all_keys:
            data_dict[key] = np.empty((self.batch_size,) + self.get_key_shape(key))

        # get ids to load
        batch_idxs = list()
        for i in range(self.batch_size):
            sample_idx = batch_idx * self.batch_size + i
            if sample_idx >= self.nb_samples:
                sample_idx -= self.nb_samples
            batch_idxs.append(sample_idx)

        # get data
        data = self.get_data_dict(batch_idxs)
        for key in self.all_keys:
            data_dict[key] = data[key]

        return data_dict

    def load_samples(self):
        '''
        returns a list of sample ids
        optional: can load the targets if possible
        '''
        raise NotImplementedError

    def init_cooc(self):
        '''
        loading the Co-occurrence matrix from disk in the case where it is static
        '''
        raise NotImplementedError

    def get_key_shape(self, key):
        '''
        shape for each key in supported_keys (without the batch size)

        returns: tuple representing the shape
        '''
        raise NotImplementedError

    def get_data_dict(self, sample_idxs):
        '''
        Creation of the actual batch

        returns: a dict, with one key per supported_keys element
        Each key contains the batch representation for this key
        '''
        raise NotImplementedError

    def update_targets(self, new_path):
        '''
        Called during relabeling to tell the dataset to batch from the new targets instead
        '''
        raise NotImplementedError
