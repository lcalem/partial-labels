import json
import os
import random

import numpy as np

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from collections import defaultdict

from data import Dataset

IMG_HEIGHT = 224
IMG_WIDTH = 224


class CocoGenerator(Dataset):

    def __init__(self, subset, data_path, prop=None):
        '''
        data path is base dir: /share/DEEPLEARNING/datasets/mscoco
        '''

        assert subset in ['train', 'val']
        self.subset = subset
        self.prop = prop

        self.data_path = data_path
        self.images_path = os.path.join(self.data_path, '%s2014' % self.subset)
        self.labels_path = os.path.join(self.data_path, 'annotations', 'instances_%s2014.json' % self.subset)

        self.json_data = self.load_json_data(self.labels_path)

        # key: image id // value: image label as one-hot
        self.id_to_label = {}

        # key: coco_id // value: {name: str, norm_id: int}
        self.cats = self.load_categories()
        self.nb_classes = len(self.cats)    # 80 classes for MSCoco

        # get all the image ids for the given subset
        self.image_ids_in_subset = self._get_image_ids()

        self.load_data()

    def load_json_data(self, annotations_file):
        '''
        open the annotations file only once
        '''
        with open(annotations_file, 'r') as f_in:
            json_data = json.load(f_in)
        return json_data

    def load_categories(self):
        '''
        categories from json_data
        '''
        cats = dict()
        for i, category in enumerate(self.json_data['categories']):
            cats[category['id']] = {'name': category['name'], 'norm_id': i}

        assert len(cats) == 80
        return cats

    def _get_image_ids(self):
        img_ids = list()
        for img in self.json_data['images']:
            img_ids.append(img['id'])

        return img_ids

    def load_data(self):
        self._initialize_id_to_label_dict()
        self._fill_id_to_label_dict_with_classes()

    def _initialize_id_to_label_dict(self):
        for image_id in self.image_ids_in_subset:
            self.id_to_label[image_id] = np.zeros(self.nb_classes)

    def _fill_id_to_label_dict_with_classes(self):
        stats = defaultdict(int)

        for annot in self.json_data['annotations']:
            cat = annot['category_id']
            norm_cat = self.cats[cat]['norm_id']
            stats[cat] += 1

            img_id = annot['image_id']

            self.id_to_label[img_id][norm_cat] = 1

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
                    img = image.load_img(os.path.join(self.images_path, 'COCO_%s2014_%012d.jpg' % (self.subset, image_id)), grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    # Cast the Image object to a numpy array and put the channel has the last dimension
                    img_arr = image.img_to_array(img, data_format='channels_last')
                    X_batch.append(img_arr)
                    # Y_batch.append(self.id_to_label[image_id])
                    Y_batch.append(self.get_labels(image_id))

                # resize X_batch in (batch_size, IMG_HEIGHT, IMG_WIDTH, 3)
                X_batch = np.reshape(X_batch, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
                # resize Y_batch in (None, nb_classes)
                Y_batch = np.reshape(Y_batch, (-1, self.nb_classes))

                # substract mean values from imagenet
                X_batch = preprocess_input(X_batch, data_format='channels_last')
                yield(X_batch, Y_batch)

    def __len__(self):
        return int(len(self.id_to_label) / self.batch_size) + 1