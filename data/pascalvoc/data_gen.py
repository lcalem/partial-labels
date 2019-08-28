"""
import os
import random
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

random.seed(2506)

IMG_HEIGHT = 448
IMG_WIDTH = 448

LABELS = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


class PascalVOCDataGenerator(object):
    '''
    prop: proportion of known labels
    '''

    def __init__(self, subset, data_path, prop=None, force_old=False):

        assert subset in ['train', 'val', 'trainval', 'test']
        self.subset = subset
        self.prop = prop

        self.data_path = data_path
        self.images_path = os.path.join(self.data_path, 'JPEGImages')
        self.labels_path = os.path.join(self.data_path, 'ImageSets', 'Main')

        # key: image id // value: image label as one-hot
        self.id_to_label = {}

        self.labels = LABELS
        self.nb_classes = len(self.labels)  # 20 classes for PascalVOC

        # Get all the images' ids for the given subset
        self.images_ids_in_subset = self._get_images_ids_from_subset(self.subset)

        if not force_old:
            self.load_csv_data()
        else:
            self.load_data()

    def load_data(self):
        '''
        aka: the old way (the working way)
        '''
        self._initialize_id_to_label_dict()

        self._fill_id_to_label_dict_with_classes()

    def load_csv_data(self):
        '''
        the new way
        /!\ loads with -1 instead of 0 for false labels
        0 are for unknown labels
        '''
        if self.subset in ['val', 'test']:
            csv_path = os.path.join(self.data_path, 'Annotations', 'annotations_multilabel_%s.csv' % self.subset)
        else:
            csv_path = os.path.join(self.data_path, 'Annotations', 'annotations_multilabel_%s_partial_%s_1.csv' % (self.subset, self.prop))

        print('loading dataset from %s' % csv_path)
        with open(csv_path, 'r') as f_csv:
            for line in f_csv:
                parts = line.split(',')
                image_id = parts[0]
                labels = [int(l) for l in parts[1:]]
                self.id_to_label[image_id] = labels

    def _initialize_id_to_label_dict(self):
        for image_id in self.images_ids_in_subset:
            self.id_to_label[image_id] = np.zeros(self.nb_classes)

    def _fill_id_to_label_dict_with_classes(self):
        '''_fill_id_to_label_dict_with_classes
        For each class, the <class>_<subset>.txt file contain the presence information
        of this class in the image
        '''
        for i in range(self.nb_classes):
            label = self.labels[i]
            # Open the <class>_<subset>.txt file
            with open(os.path.join(self.labels_path, "%s_%s.txt" % (label, self.subset)), 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    splited_line = line.split()
                    image_id = splited_line[0]
                    is_present = int(splited_line[1])
                    if is_present == 1:
                        self.id_to_label[image_id][i] = 1

    def _get_images_ids_from_subset(self, subset):
        '''
        _get_images_ids_from_subset
        The images' ids are found in the <subset>.txt file in ImageSets/Main
        '''
        with open(os.path.join(self.labels_path, subset + '.txt'), 'r') as f:
            images_ids = f.read().splitlines()
        return images_ids

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

    def flow(self, batch_size):
        '''
        X_batch of size (None, IMG_HEIGHT, IMG_WIDTH, 3)
        Y_batch of size (None, nb_classes)
        '''
        nb_batches = int(len(self.images_ids_in_subset) / batch_size) + 1
        while True:
            random.shuffle(self.images_ids_in_subset)

            for i in range(nb_batches):
                # image ids
                current_bach = self.images_ids_in_subset[i * batch_size:(i + 1) * batch_size]
                X_batch = []
                Y_batch = []
                for image_id in current_bach:

                    img = image.load_img(os.path.join(self.images_path, image_id + '.jpg'), grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    img_arr = image.img_to_array(img, data_format='channels_last')
                    X_batch.append(img_arr)
                    # Y_batch.append(self.id_to_label[image_id])
                    Y_batch.append(self.get_labels(image_id))

                X_batch = np.reshape(X_batch, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
                Y_batch = np.reshape(Y_batch, (-1, self.nb_classes))

                X_batch = preprocess_input(X_batch, data_format='channels_last')
                yield(X_batch, Y_batch)

    def __len__(self):
        return len(self.id_to_label)

"""
