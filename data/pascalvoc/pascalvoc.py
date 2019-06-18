'''
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit
'''
import os

import numpy as np

from data import Dataset
from data.pascalvoc.preprocessing import utils

from model.utils.config import cfg


class PascalVOC(Dataset):

    supported_keys = ('image', 'multilabel')

    def __init__(self,
                 dataset_path,
                 mode,
                 x_keys,
                 y_keys):
        '''
        Only multilabel for now
        TODO: locking for shuffle?
        '''
        assert mode in ['train', 'val', 'trainval', 'test'], 'Unknown mode %s' % str(mode)

        self.dataset_path = dataset_path
        self.mode = mode
        self.x_keys = x_keys
        self.y_keys = y_keys
        self.all_keys = set(x_keys + y_keys)
        assert all([k in self.supported_keys for k in self.all_keys]), 'Unexpected key in %s' % str(self.all_keys)

        self.batch_size = cfg.BATCH_SIZE
        self.shuffle = cfg.DATASET.SHUFFLE

        self.class_info = utils.load_ids()
        self.n_classes = len(self.class_info)

        self.load_annotations(os.path.join(dataset_path, 'VOCdevkit/VOC2007/Annotations/annotations_multilabel_%s.csv' % mode))

        Dataset.__init__(self)

    def __len__(self):
        raise NotImplementedError

    def load_annotations(self, annotations_path):
        pass

    def __getitem__(self, batch_idx):
        data_dict = self.get_data(batch_idx)

        # Convert the dictionary of samples to a list for x and y
        x_batch = []
        for key in self.x_keys:
            x_batch.append(data_dict[key])

        y_batch = []
        for key in self.y_keys:
            y_batch.append(data_dict[key])

        return x_batch, y_batch

    def get_data(self, batch_idx):
        '''
        Used by Dataset.__getitem__ during batching
        '''
        data_dict = {}
        for key in self.all_keys:
            data_dict[key] = np.empty((self.batch_size,) + self.get_key_shape(key))

        i_batch = 0
        for i in range(self.batch_size):
            if self.shuffle:
                key = self.get_shuffled_key()
            else:
                key = idx * self.batch_size + i
                if key >= self.get_length(mode):
                    key -= self.get_length(mode)

            data = self.get_data_dict(key, mode)
            for dkey in self.allkeys:
                data_dict[dkey][batch_cnt, :] = data[dkey]

            batch_cnt += 1

        return data_dict

    def get_data_dict(self, idx):
        output = {}

        output['frame'] = frames
        output['multilabel'] = multilabel

        return output
