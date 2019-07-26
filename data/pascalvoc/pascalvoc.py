'''
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit
'''
import math
import os

from collections import defaultdict

import numpy as np
import tensorflow as tf

from PIL import Image

from data import Dataset
from data.pascalvoc.preprocessing import utils

from model.utils.config import cfg

NB_CLASSES = 20


class PascalVOC(Dataset):

    supported_keys = ('image', 'multilabel')

    def __init__(self,
                 dataset_path,
                 batch_size,
                 subset,
                 x_keys,
                 y_keys,
                 p=None):
        '''
        Only multilabel for now
        - dataset_path: folder where VOCdevkit/VOC2007/ is contained
        - subset: train / val / trainval / test -> file to be loaded
        - p: known labels proportion -> will be used to open the correct partial dataset file (only for training)


        Note: shuffle is managed in the fit_generator, not at all here
        '''
        assert subset in ['train', 'val', 'trainval', 'test'], 'Unknown subset %s' % str(subset)

        self.dataset_path = dataset_path
        self.subset = subset
        self.p = p

        self.x_keys = x_keys
        self.y_keys = y_keys
        self.all_keys = set(x_keys + y_keys)
        assert all([k in self.supported_keys for k in self.all_keys]), 'Unexpected key in %s' % str(self.all_keys)

        self.batch_size = batch_size

        self.class_info = utils.load_ids()
        self.nb_classes = len(self.class_info)
        self.img_size = (cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, cfg.IMAGE.N_CHANNELS)

        self.samples = defaultdict(dict)
        annotations_file = self.get_annot_file(p)
        self.load_annotations(annotations_file)

        Dataset.__init__(self)

    def __len__(self):
        '''
        Should give off the number of batches
        '''
        return math.floor(self.nb_samples / self.batch_size)

    def get_annot_file(self, p):
        if p is not None:
            assert self.mode.startswith('train'), 'partial labels datasets only available for training'
            name = '%s_partial_%s_%s' % (self.mode, p, cfg.RANDOM_SEED)
        else:
            name = self.mode

        return os.path.join(self.dataset_path, 'VOCdevkit/VOC2007/Annotations/annotations_multilabel_%s.csv' % name)

    def load_annotations(self, annotations_path):
        '''
        for now we only load the multilabel classification annotations.

        example line :
        000131,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1
        '''

        samples = defaultdict(dict)

        # multilabel annotations
        with open(annotations_path, 'r') as f_in:
            for line in f_in:
                parts = line.strip().split(',')
                ground_truth = [int(val) for val in parts[1:]]
                assert all([gt in [-1, 0, 1] for gt in ground_truth])

                if self.mode in ['val', 'test']:
                    ground_truth = self.convert_multilabel_to_binary(ground_truth)

                samples[parts[0]]['multilabel'] = ground_truth

        # we sort it by # sample to make sure it's always the same order
        self.sample_ids = sorted(samples.keys())
        self.samples = {k: samples[k] for k in self.sample_ids}
        self.nb_samples = len(self.samples)

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

        for i in range(self.batch_size):
            sample_idx = batch_idx * self.batch_size + i
            if sample_idx >= self.nb_samples:
                sample_idx -= self.nb_samples

            data = self.get_data_dict(sample_idx)
            for key in self.all_keys:
                data_dict[key][i, :] = data[key]

        return data_dict

    def get_key_shape(self, key):
        if key == 'image':
            return self.img_size
        elif key == 'multilabel':
            return (self.nb_classes, )
        else:
            raise Exception('Unknown key %s' % key)

    def get_data_dict(self, sample_idx):
        output = {}
        sample_id = self.sample_ids[sample_idx]

        output['image'] = self.get_image(sample_id)
        output['multilabel'] = self.samples[sample_id]['multilabel']

        return output

    def convert_multilabel_to_binary(self, multilabel_truth):
        '''
        originally the dataset is -1 / 0 / 1 (resp. false / Unknown / true),
        in this function -1 will be converted to 0 (useful for val and test)

        Useful for val and test sets but not for train because the unknown distinction is used in the loss directly
        '''
        return np.where(np.equal(multilabel_truth, -1), np.zeros_like(multilabel_truth), multilabel_truth).tolist()

    def get_image(self, img_id):
        '''
        open the image with given id (like 000131)
        TODO: we want it to fail if an image is not found, but we should make it fail gracefully
        '''
        image_path = os.path.join(self.dataset_path, 'VOCdevkit/VOC2007/JPEGImages/%s.jpg' % img_id)
        img = Image.open(image_path)
        img = img.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)
