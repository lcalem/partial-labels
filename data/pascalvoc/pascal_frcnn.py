'''
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit
'''
import time
import os

import numpy as np

from collections import defaultdict

from keras.preprocessing import image as k_image
from keras.applications.imagenet_utils import preprocess_input

from model.mrcnn import dataset

from data.pascalvoc.preprocessing.utils import load_ids_frcnn


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
        class_data = load_ids_frcnn()

        id2data = {int(elt['id']): elt for elt in class_data.values()}
        return id2data

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

        return os.path.join(self.dataset_path, 'Annotations', 'frcnn_%s.csv' % self.mode)

    def one_hotify_gt(self, numeric_gt):
        return np.array([int(i == numeric_gt) for i in range(self.nb_classes)])

    def load_annotations(self, annotations_path):
        '''
        one annotation csv line is like:
        img_id,cls_id,tag_id,is_part,parent_id,xmin,ymin,xmax,ymax,width,height,depth

        ---

        load annotations:
        - samples[img_id][multilabel] = list(one_hot)   # (N, nb_classes) list of variable size depending on the number of bounding boxes that are GT
        - samples[img_id][bboxes] = list(bbox)          # (N, 4) list of variable size of 4 int tuples representing x1, y1, x2, y2

        example:
        samples['5c015f7e9bbd728a']['multilabel'][0] = [0, 1, ....] (length 20)
        samples['5c015f7e9bbd728a']['bboxes'][0] = (x1, y1, x2, y2) here they are in pixels
        '''

        samples = defaultdict(lambda: defaultdict(list))

        # multilabel annotations
        with open(annotations_path, 'r') as f_in:
            for line in f_in:
                parts = line.strip().split(',')
                is_part = int(parts[3])
                if is_part:
                    continue

                img_id = parts[0]
                class_id = int(parts[1])
                ground_truth_cls = self.one_hotify_gt(class_id)

                width = int(parts[9])
                height = int(parts[10])
                size = (width, height)

                # convert bbox to % (they are originally in pixels)
                bbox = (float(parts[5]) / width , float(parts[6]) / height, float(parts[7]) / width, float(parts[8]) / height)

                samples[img_id]['multilabel'].append(ground_truth_cls)
                samples[img_id]['bboxes'].append(bbox)
                samples[img_id]['classes'].append((self.class_data[class_id]['id'], self.class_data[class_id]['name']))

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
        img_path = os.path.join(self.dataset_path, 'JPEGImages', '%s.jpg' % img_id)
        img = k_image.load_img(img_path, grayscale=False, target_size=(self.img_size[0], self.img_size[1]))
        img_arr = k_image.img_to_array(img, data_format='channels_last')

        return img_arr

    def load_bboxes(self, img_index):
        '''
        Generate instance GT bboxes for shapes of the given image ID.
        bboxes in targets are between 0 and 1 so we put them at pixel level here

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
