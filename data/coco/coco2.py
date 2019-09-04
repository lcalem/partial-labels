import os

from collections import defaultdict

import numpy as np

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from data import Dataset


NB_CLASSES = 80


class CocoGenerator(Dataset):

    supported_keys = ('image', 'multilabel')
    supported_modes = ('train', 'val')
    nb_classes = NB_CLASSES

    def __init__(self,
                 dataset_path,
                 batch_size,
                 mode,
                 x_keys,
                 y_keys,
                 year='2014',
                 p=None):

        assert year in ['2014', '2017'], 'Invalid Coco year %s (accepted years 2014 and 2017)' % year
        self.year = year

        self.images_path = os.path.join(dataset_path, '%s%s' % (mode, self.year))

        Dataset.__init__(self, dataset_path, batch_size, mode, x_keys, y_keys, p)

    def load_samples(self):
        annotations_file = self.get_annot_file(self.p)
        samples = self.load_annotations(annotations_file)

        self.targets = samples
        return samples.keys()

    def get_annot_file(self, p):
        if self.mode == 'val':
            dataset_path = os.path.join(self.dataset_path, 'annotations', 'multilabel_val%s.csv' % self.year)
        elif self.mode == 'train':
            dataset_path = os.path.join(self.dataset_path, 'annotations', 'multilabel_train%s_partial_%s_1.csv' % (self.year, self.p))    # TODO: seed

        return dataset_path

    def init_cooc(self):
        '''
        loading the Co-occurrence matrix from disk in the case where it is static
        '''
        pass

    def load_annotations(self, annotations_path):
        '''
        '''
        samples = defaultdict(dict)

        print('loading dataset from %s' % annotations_path)
        with open(annotations_path, 'r') as f_in:
            for line in f_in:
                parts = line.strip().split(',')
                image_id = parts[0]
                labels = [int(l) for l in parts[1:]]
                samples[image_id]['multilabel'] = labels

        return samples

    def get_key_shape(self, key):
        if key == 'image':
            return self.img_size
        elif key == 'multilabel':
            return (self.nb_classes, )
        else:
            raise Exception('Unknown key %s' % key)

    def get_data_dict(self, sample_idxs):
        '''
        Creation of the actual batch
        '''
        output = {}
        sample_ids = [self.sample_ids[i] for i in sample_idxs]

        img_batch = []
        target_batch = []

        for img_id in sample_ids:
            img = image.load_img(self.get_img_path(int(img_id)), grayscale=False, target_size=(self.img_size[0], self.img_size[1]))
            img_arr = image.img_to_array(img, data_format='channels_last')
            img_batch.append(img_arr)

            target_batch.append(self.get_labels(img_id))

        img_batch = np.reshape(img_batch, (-1, self.img_size[0], self.img_size[1], 3))  # TODO: figure out why this line is necessary
        img_batch = preprocess_input(img_batch, data_format='channels_last')

        target_batch = np.reshape(target_batch, (-1, self.nb_classes))

        output['image'] = img_batch
        output['multilabel'] = target_batch

        return output

    def get_img_path(self, img_id):
        prefix = ''
        if self.year == '2014':
            prefix = 'COCO_%s%s_' % (self.mode, self.year)
        return os.path.join(self.images_path, '%s%012d.jpg' % (prefix, int(img_id)))

    def get_labels(self, image_id):
        '''
        for tests
        negative labels:
        -1 for train
        0 for test or val
        '''
        labels = self.targets[image_id]['multilabel']
        if self.mode in ['val', 'test']:
            labels = [0 if l == -1 else l for l in labels]

        return labels
