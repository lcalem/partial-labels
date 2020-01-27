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

    supported_keys = ('image', 'multilabel', 'cooc_matrix', 'image_id')
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

    def load_samples(self, filepath=None):
        if filepath is None:
            annotations_file = self.get_annot_file(self.p)
        else:
            annotations_file = filepath

        samples = self.load_annotations(annotations_file)

        self.targets = samples
        return sorted(samples.keys())  # we sort it by # sample to make sure it's always the same order

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
        target_batch = []
        ids_batch = []

        for img_id in sample_ids:
            img_path = os.path.join(self.dataset_path, 'JPEGImages/%s.jpg' % img_id)
            img = image.load_img(img_path, grayscale=False, target_size=(self.img_size[0], self.img_size[1]))
            img_arr = image.img_to_array(img, data_format='channels_last')
            img_batch.append(img_arr)

            target_batch.append(self.get_labels(img_id))
            ids_batch.append(img_id)

        img_batch = np.reshape(img_batch, (-1, self.img_size[0], self.img_size[1], 3))  # TODO: figure out why this line is necessary
        img_batch = preprocess_input(img_batch, data_format='channels_last')

        target_batch = np.reshape(target_batch, (-1, self.nb_classes))
        ids_batch = np.reshape(np.array(ids_batch), (-1, 1))

        output['image'] = img_batch
        output['cooc_matrix'] = np.repeat(self.cooc_matrix[None, ...], len(sample_idxs), 0)
        output['multilabel'] = target_batch
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

import time
import os

import numpy as np

from collections import defaultdict

from keras.preprocessing import image as k_image
from keras.applications.imagenet_utils import preprocess_input

from model.mrcnn import dataset


class PascalVOCDataset(dataset.Dataset):
    '''
    PascalVOC dataset generator version for frcnn
    '''
    supported_modes = ('train', 'val', 'test', 'trainval')

    def load_pascal(self,
                    dataset_path,
                    batch_size,
                    mode,
                    cfg,
                    p=None):

        assert mode in self.supported_modes, 'Unknown subset %s' % str(mode)
        if p is not None and not mode.startswith('train'):
            raise Exception('prop only for training')

        self.dataset_path = dataset_path
        self.mode = mode
        self.p = p
        self.nb_classes = cfg.NB_CLASSES

        self.img_size = (cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.NB_CHANNELS)

        # class data
        self.class_data = self.load_class_data()
        for i, data in self.class_data.items():
            self.add_class('pascalvoc', i, data["name"])

        # loading samples
        self.sample_ids = self.load_samples()
        self.nb_samples = len(self.sample_ids)

        self.batch_size = self.nb_samples if batch_size == 'all' else batch_size

        # registering step
        t0 = time.time()

        for img_id, data in self.targets.items():
            self.add_image(
                    'pascalvoc',
                    image_id=img_id,
                    path=None,
                    width=data['size'][0],
                    height=data['size'][1])
        t1 = time.time()
        total = t1 - t0
        print('Registered images in %s' % total)

    def load_class_data(self):
        class_data = dict()
        class_data_path = os.path.join(self.dataset_path, 'annotations', 'challenge-2018-classes.csv')

        with open(class_data_path, 'r') as f_in:
            for line in f_in:
                parts = line.strip().split(',')
                class_id = int(parts[0])
                class_name = parts[1]
                class_oid = parts[2]

                class_data[class_id] = {'id': class_id, 'name': class_name, 'oid': class_oid}

        return class_data

    def load_samples(self, filepath=None):
        if filepath is None:
            annotations_file = self.get_annot_file(self.p)
        else:
            annotations_file = filepath

        t0 = time.time()
        samples = self.load_annotations(annotations_file)
        t1 = time.time()

        total = t1 - t0
        print('loaded annotations in %s' % total)

        self.targets = samples
        return sorted(samples.keys())  # we sort it by # sample to make sure it's always the same order

    def get_annot_file(self, p):

        return os.path.join(self.dataset_path, 'annotations', 'challenge-2018-%s.csv' % self.mode)

    def one_hotify_gt(self, numeric_gt):
        return np.array([int(i == numeric_gt) for i in range(self.nb_classes)])

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
                class_id = int(parts[2])
                ground_truth_cls = self.one_hotify_gt(class_id)

                samples[img_id]['multilabel'].append(ground_truth_cls)
                samples[img_id]['bboxes'].append((float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])))
                samples[img_id]['classes'].append((self.class_data[class_id]['id'], self.class_data[class_id]['name']))

                size = (parts[7], parts[8])

                if samples[img_id]['size'] == list():  # default
                    samples[img_id]['size'] = size
                else:
                    assert samples[img_id]['size'] == size    # it should always be the same

        return samples

    def get_img_id(self, img_index):
        return self.sample_ids[img_index]

    def load_image(self, img_index):
        '''
        Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        '''
        img_id = self.get_img_id(img_index)
        img_path = os.path.join(self.dataset_path, 'images', 'train', 'train_%s' % img_id[0], '%s.jpg' % img_id)
        img = k_image.load_img(img_path, grayscale=False, target_size=(self.img_size[0], self.img_size[1]))
        img_arr = k_image.img_to_array(img, data_format='channels_last')

        return img_arr

    def load_bboxes(self, img_index):
        '''
        Generate instance GT bboxes for shapes of the given image ID.
        bboxes in targets are between 0 and 1 so we put them at pixel level here

        bboxes are always

        WARNING:
        In the dataset and in self.targets the boxes are [x1, y1, x2, y2]
        However as the code requires a different format the output of load_bboxes is [y1, x1, y2, x2]
        '''
        img_id = self.get_img_id(img_index)
        img_data = self.targets[img_id]

        # print(type(img_data['bboxes']))
        gt_bboxes = np.array(img_data['bboxes'])
        gen_bboxes = np.empty_like(img_data['bboxes'])

        classes = np.array([c[0] for c in img_data['classes']])

        width, height, _ = self.img_size
        # print(height)
        # print(width)

        for i, bbox in enumerate(gt_bboxes):
            # bbox (input) is [x1, y1, x2, y2]
            # bboxes (output) is [y1, x1, y2, x2]
            # print('gt box (x-y) %s' % str(bbox))
            gen_bboxes[i][0] = bbox[1] * height
            gen_bboxes[i][1] = bbox[0] * width
            gen_bboxes[i][2] = bbox[3] * height
            gen_bboxes[i][3] = bbox[2] * width

            # print('gt box 2 (x-y) %s' % str(bbox))
            # print('final box (y-x) %s' % str(gen_bboxes[i]))

        return gen_bboxes, classes
