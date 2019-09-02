import json
import os
import random

import numpy as np

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from collections import defaultdict

from data import Dataset

from config.config import cfg

NB_CLASSES = 80


class CocoGenerator(Dataset):

    def __init__(self, subset, data_path, year='2014', prop=None):
        '''
        data path is base dir: /share/DEEPLEARNING/datasets/mscoco
        '''
        assert year in ['2014', '2017']
        if prop is not None and subset != 'train':
            raise Exception('prop only for training')

        self.subset = subset
        self.year = year
        self.prop = prop or 100
        self.nb_classes = NB_CLASSES

        self.data_path = data_path
        self.images_path = os.path.join(self.data_path, '%s%s' % (self.subset, self.year))

        # key: image id // value: image label as one-hot
        self.id_to_label = {}

        self.load_data()
        self.image_ids_in_subset = list(self.id_to_label.keys())

    def load_data(self):
        if self.subset == 'val':
            dataset_path = os.path.join(self.data_path, 'annotations', 'multilabel_val%s.csv' % self.year)
        elif self.subset == 'train':
            dataset_path = os.path.join(self.data_path, 'annotations', 'multilabel_train%s_partial_%s_1.csv' % (self.year, self.prop))    # TODO: seed

        print('loading dataset from %s' % dataset_path)
        with open(dataset_path, 'r') as f_in:
            for line in f_in:
                parts = line.split(',')
                image_id = parts[0]
                labels = [int(l) for l in parts[1:]]
                self.id_to_label[image_id] = labels

    def get_labels(self, image_id):
        '''
        for tests
        negative labels:
        -1 for train
        0 for test or val
        '''
        labels = self.id_to_label[image_id]
        # if self.subset.startswith('train'):
            # labels = [-1 if l == 0 else l for l in labels]
        if self.subset in ['val', 'test']:
            labels = [0 if l == -1 else l for l in labels]

        return labels

    def get_img_path(self, img_id):
        prefix = ''
        if self.year == '2014':
            prefix = 'COCO_%s%s_' % (self.subset, self.year)
        return os.path.join(self.images_path, '%s%012d.jpg' % (prefix, int(img_id)))

    def flow(self, batch_size=32):
        """
        When calling next python build in function, it returns a batch with a given size
        with a X_batch of size (None, IMG_HEIGHT, IMG_WIDTH, 3)
        and a Y_batch of size (None, nb_classes)
        """
        nb_batches = int(len(self.image_ids_in_subset) / batch_size) + 1
        while True:
            # Before each epoch we shuffle the images' ids
            random.shuffle(self.image_ids_in_subset)

            for i in range(nb_batches):
                # We first get all the image ids for the next batch
                current_bach = self.image_ids_in_subset[i*batch_size:(i+1)*batch_size]
                X_batch = []
                Y_batch = []

                for image_id in current_bach:
                    # Load the image and resize it. We get a PIL Image object
                    img = image.load_img(self.get_img_path(int(image_id)), grayscale=False, target_size=(cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE))
                    # Cast the Image object to a numpy array and put the channel has the last dimension
                    img_arr = image.img_to_array(img, data_format='channels_last')
                    X_batch.append(img_arr)
                    # Y_batch.append(self.id_to_label[image_id])
                    Y_batch.append(self.get_labels(image_id))

                # resize X_batch in (batch_size, IMG_HEIGHT, IMG_WIDTH, 3)
                X_batch = np.reshape(X_batch, (-1, cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, 3))
                # resize Y_batch in (None, nb_classes)
                Y_batch = np.reshape(Y_batch, (-1, self.nb_classes))

                # substract mean values from imagenet
                X_batch = preprocess_input(X_batch, data_format='channels_last')
                yield(X_batch, Y_batch)

    def load_test(self):
        X_batch = []
        Y_batch = []
        count = 0

        for image_id in self.id_to_label:
            if count % 1000 == 0:
                print("loaded %s images" % count)
            count += 1

            # Load the image and resize it. We get a PIL Image object
            img = image.load_img(self.get_img_path(int(image_id)), grayscale=False, target_size=(cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE))
            # Cast the Image object to a numpy array and put the channel has the last dimension
            img_arr = image.img_to_array(img, data_format='channels_last')
            X_batch.append(img_arr)
            # Y_batch.append(self.id_to_label[image_id])
            Y_batch.append(self.get_labels(image_id))

        # resize X_batch in (batch_size, IMG_HEIGHT, IMG_WIDTH, 3)
        X_batch = np.reshape(X_batch, (-1, cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, 3))
        # resize Y_batch in (None, nb_classes)
        Y_batch = np.reshape(Y_batch, (-1, self.nb_classes))

        # substract mean values from imagenet
        X_batch = preprocess_input(X_batch, data_format='channels_last')
        return (X_batch, Y_batch)

    def __len__(self):
        '''
        total number of images
        '''
        return len(self.id_to_label)
