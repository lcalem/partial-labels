'''
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit
'''
import os

from collections import defaultdict

import numpy as np

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from data import Dataset

from config.config import cfg

NB_CLASSES = 20


class PascalVOC(Dataset):

    supported_keys = ('image', 'multilabel', 'cooc_matrix')
    supported_modes = ('train', 'val', 'trainval', 'test')
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
        if p is not None:
            assert self.mode.startswith('train'), 'partial labels datasets only available for training'
            name = '%s_partial_%s_%s' % (self.mode, p, cfg.RANDOM_SEED)
        else:
            name = self.mode

        return os.path.join(self.dataset_path, 'Annotations/annotations_multilabel_%s.csv' % name)

    def init_cooc(self):
        '''
        loading the Co-occurrence matrix from disk in the case where it is static
        OR
        only load a zeroed out matrix (not now)
        '''
        matrix_filename = 'cooc_matrix_trainval_partial_100_1.npy'
        matrix_filepath = os.path.join(self.dataset_path, 'Annotations', matrix_filename)
        self.cooc_matrix = np.load(matrix_filepath)
        # print(self.cooc_matrix)

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

        return samples

    def get_key_shape(self, key):
        if key == 'image':
            return self.img_size
        elif key == 'multilabel':
            return (self.nb_classes, )
        elif key == 'cooc_matrix':
            return (self.nb_classes, self.nb_classes)
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
            img_path = os.path.join(self.dataset_path, 'JPEGImages/%s.jpg' % img_id)
            img = image.load_img(img_path, grayscale=False, target_size=(self.img_size[0], self.img_size[1]))
            img_arr = image.img_to_array(img, data_format='channels_last')
            img_batch.append(img_arr)

            target_batch.append(self.get_labels(img_id))

        img_batch = np.reshape(img_batch, (-1, self.img_size[0], self.img_size[1], 3))  # TODO: figure out why this line is necessary
        img_batch = preprocess_input(img_batch, data_format='channels_last')

        target_batch = np.reshape(target_batch, (-1, self.nb_classes))

        output['image'] = img_batch
        output['cooc_matrix'] = np.repeat(self.cooc_matrix[None, ...], len(sample_idxs), 0)
        output['multilabel'] = target_batch

        return output

    def convert_multilabel_to_binary(self, multilabel_truth):
        '''
        originally the dataset is -1 / 0 / 1 (resp. false / Unknown / true),
        in this function -1 will be converted to 0 (useful for val and test)

        Useful for val and test sets but not for train because the unknown distinction is used in the loss directly
        '''
        return np.where(np.equal(multilabel_truth, -1), np.zeros_like(multilabel_truth), multilabel_truth).tolist()

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
