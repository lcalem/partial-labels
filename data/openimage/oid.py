'''
'''
import os

from collections import defaultdict

import numpy as np

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from data import Dataset

from config.config import cfg

NB_CLASSES = 500


class OIDGenerator(Dataset):

    supported_keys = ('image', 'multilabel', 'bbox', 'image_id')
    supported_modes = ('train', 'val', 'test')
    nb_classes = NB_CLASSES

    def __init__(self,
                 dataset_path,
                 batch_size,
                 mode,
                 x_keys,
                 y_keys,
                 p=None):

        self.images_path = os.path.join(dataset_path, 'images')

        Dataset.__init__(self, dataset_path, batch_size, mode, x_keys, y_keys, p)

    def load_samples(self, filepath=None):
        if filepath is None:
            annotations_file = self.get_annot_file(self.p)
        else:
            annotations_file = filepath

        samples = self.load_annotations(annotations_file)

        self.targets = samples
        return sorted(samples.keys())  # we sort it by # sample to make sure it's always the same order

    def get_annot_file(self, p):

        return os.path.join(self.dataset_path, 'annotations', 'challenge-2018-%s.csv' % self.mode)

    def init_cooc(self):
        '''
        loading the Co-occurrence matrix from disk in the case where it is static
        OR
        only load a zeroed out matrix (not now)
        '''
        pass

    def load_annotations(self, annotations_path):
        '''
        one annotation csv line is like:
        img_id,folder,class_id,xmin,ymin,xmax,ymax

        38a420b38cd3c350,train_3,92,0.325781,0.450781,0.447917,0.668750

        ---

        load annotations:
        - samples[img_id][multilabel] = list(one_hot)   # (N, nb_classes) list of variable size depending on the number of bounding boxes that are GT
        - samples[img_id][bboxes] = list(bbox)          # (N, 4) list of variable size of 4 int tuples representing x1, y1, x2, y2

        example:
        samples['5c015f7e9bbd728a']['multilabel'][0] = [0, 1, ....] (length 500)
        samples['5c015f7e9bbd728a']['bboxes'][0] = (x1, y1, x2, y2)
        '''

        samples = defaultdict(lambda: defaultdict(list))

        # multilabel annotations
        with open(annotations_path, 'r') as f_in:
            for line in f_in:
                parts = line.strip().split(',')
                img_id = parts[0]
                ground_truth_cls = self.one_hotify_gt(parts[2])

                samples[img_id]['multilabel'].append(ground_truth_cls)
                samples[img_id]['bboxes'].append((parts[3], parts[4], parts[5], parts[6]))

        return samples

    def one_hotify_gt(self, numeric_gt):
        return np.array([int(i == numeric_gt) for i in range(self.nb_classes)])

    def get_key_shape(self, key):
        if key == 'image':
            return self.img_size
        elif key == 'multilabel':
            return (self.nb_classes, )
        elif key == 'cooc_matrix':
            return (self.nb_classes, self.nb_classes)
        elif key == 'image_id':
            return (1, )
        else:
            raise Exception('Unknown key %s' % key)

    def get_data_dict(self, sample_idxs):
        '''
        Creation of the actual batch
        '''
        output = {}
        sample_ids = [self.sample_ids[i] for i in sample_idxs]

        img_batch = []
        cls_batch = []
        regr_batch = []

        debug_img_batch = []
        ids_batch = []

        for img_id in sample_ids:
            debug = dict()

            img_path = os.path.join(self.dataset_path, 'images', 'train', 'train_%s' % img_id[0], '%s.jpg' % img_id)
            img = image.load_img(img_path, grayscale=False, target_size=(self.img_size[0], self.img_size[1]))
            img_arr = image.img_to_array(img, data_format='channels_last')
            img_batch.append(img_arr)

            cls_batch.append(self.get_labels(img_id))
            ids_batch.append(img_id)

            debug_img_batch.append(debug)

        img_batch = np.reshape(img_batch, (-1, self.img_size[0], self.img_size[1], 3))  # TODO: figure out why this line is necessary
        img_batch = preprocess_input(img_batch, data_format='channels_last')

        target_batch = np.reshape(target_batch, (-1, self.nb_classes))
        ids_batch = np.reshape(np.array(ids_batch), (-1, 1))

        output['image'] = img_batch
        output['rpn_cls'] = cls_batch
        output['rpn_regr'] = regr_batch
        output['debug_img_data'] = debug_img_batch
        output['image_id'] = ids_batch

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

    def update_targets(self, new_path):
        '''
        Update annotations directly
        '''
        self.load_samples(filepath=new_path)
