import os
import sys

import numpy as np

from sklearn import metrics
from collections import defaultdict

from data.pascalvoc.preprocessing.utils import load_ids
from data.pascalvoc.pascalvoc import PascalVOC

from model import priors
from model.networks.baseline_logits import BaselineLogits

from pprint import pprint

from experiments import launch_utils as utils
from config import config_utils


class PriorMapTest(object):

    def __init__(self):
        self.batch_size = 16
        self.nb_classes = 20
        self.prop = 10

        class_info = load_ids()
        self.id2name = {int(info['id']): info['name'] for info in class_info.values()}

        # options = utils.parse_options_file('pv_baseline')
        # config_utils.update_config(options)

        self.data_dir = '/home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/'
        self.weights_path = '/home/caleml/partial_experiments/exp_20191028_1501_baseline_logits_pv_baseline/weights_100_001.h5'
        # self.weights_path = '/home/caleml/partial_experiments/exp_20190725_1516_baseline/weights_100_020.h5'
        self.gt_path = os.path.join(self.data_dir, 'Annotations', 'annotations_multilabel_trainval_partial_100_1.csv')
        self.test_path = os.path.join(self.data_dir, 'Annotations', 'annotations_multilabel_test.csv')

        # 1. load model
        self.model = BaselineLogits('%s/partial_experiments/' % os.environ['HOME'], self.nb_classes, self.prop)
        self.model.load_weights(self.weights_path)

        # load prior
        self.load_prior()

        # load GT
        self.gt = self.load_annotations(self.test_path)

    def compute_prior_map(self):
        '''
        1. Load model trained without relabeling
        2. for each example of the train set:
            - get the visual y_v
            - compute the full prior y_p from the test ground truth (YES THIS IS JUST A TEST)
            - merge that to get y_f
        3. per class we gather the ground truth y_gt
        4. for each of y_v, y_p, y_f, compute a mAP between y_gt and the y
        '''

        # 2. train dataset
        dataset_test = PascalVOC(self.data_dir, self.batch_size, 'test', x_keys=['image', 'image_id'], y_keys=['multilabel'])

        y_true_k = defaultdict(list)
        y_v_k = defaultdict(list)
        y_f_k = defaultdict(list)
        y_p_k = defaultdict(list)

        for i_batch in range(len(dataset_test)):
            if i_batch % 100 == 0:
                print('doing batch %s' % i_batch)

            x_batch, y_batch = dataset_test[i_batch]
            y_true = y_batch[0]
            bs, K = y_true.shape
            assert bs == self.batch_size
            assert K == self.nb_classes

            y_pred = self.model.predict(x_batch)   # y_pred[0] -> y_v, y_pred[1] -> logits of y_v
            y_v_logits = np.asarray(y_pred[1])

            p_k_logits = self.prior.compute_pk_logits(y_true)
            y_f = self.prior.combine(y_v_logits, p_k_logits)

            batch_img_ids = np.array(x_batch[1])[:, 0]

            # add values for each class where y_batch is 0
            for i_class in range(self.nb_classes):

                # 3. get ground truth values matching the zeros from the 10% dataset
                batch_gt_k = [self.gt[img_id][i_class] for img_id in batch_img_ids]  # all GT for the batch
                batch_gt_k_class = [elt for i, elt in enumerate(batch_gt_k)]
                y_true_k[i_class].append(batch_gt_k_class)

                # add y_v for the zeros
                y_v_class = [elt[i_class] for i, elt in enumerate(y_v_logits)]
                y_v_k[i_class].append(y_v_class)

                # add y_f for the zeros
                y_f_class = [elt[i_class] for i, elt in enumerate(y_f)]
                y_f_k[i_class].append(y_f_class)

                # add y_p for the zeros
                assert p_k_logits.shape == (self.batch_size, self.nb_classes)
                y_p_class = [elt[i_class] for i, elt in enumerate(p_k_logits)]
                y_p_k[i_class].append(y_p_class)

            # print('y_true')
            # print(y_true)

            # print('y_v size %s' % str(y_v.shape))
            # print('y_f size %s' % str(y_f.shape))

            # print('y_true_k')
            # pprint(y_true_k)

            # print('y_v_k')
            # pprint(y_v_k)

            # print('y_f_k')
            # pprint(y_f_k)

            # raise

        # 4. compute map
        all_map_v = list()
        all_map_f = list()
        all_map_p = list()
        print('class,map_visual,map_yf,map_prior')

        for i_class in range(self.nb_classes):

            # print(y_true_k[i_class])

            y_gt_val = np.concatenate(y_true_k[i_class]).astype(np.float64)
            y_v_val = self.get_sigmoid(np.concatenate(y_v_k[i_class]).astype(np.float64))
            y_f_val = np.concatenate(y_f_k[i_class]).astype(np.float64)
            y_p_val = self.get_sigmoid(np.concatenate(y_p_k[i_class]).astype(np.float64))

            # print(y_gt_val[0])
            # print(y_v_val[0])
            # print(y_f_val[0])
            # print(y_p_val[0])

            # print(y_gt_val.shape)
            # print(y_v_val.shape)

            # print(type(y_gt_val))
            # print(y_gt_val[0])

            map_visual = metrics.average_precision_score(y_gt_val, y_v_val)
            map_yf = metrics.average_precision_score(y_gt_val, y_f_val)
            map_prior = metrics.average_precision_score(y_gt_val, y_p_val)

            print('%s (%s),%s,%s,%s' % (i_class, self.id2name[i_class], map_visual, map_yf, map_prior))

            all_map_v.append(map_visual)
            all_map_f.append(map_yf)
            all_map_p.append(map_prior)

        print('average test map visual, %s' % (sum(all_map_v) / len(all_map_v)))
        print('average test map combined, %s' % (sum(all_map_f) / len(all_map_f)))
        print('average test map prior only, %s' % (sum(all_map_p) / len(all_map_p)))

    def load_annotations(self, annotations_path):
        '''
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

                samples[parts[0]] = [0 if elt == -1 else elt for elt in ground_truth]

        return samples

    def get_sigmoid(self, values):
        return 1.0 / (1.0 + np.exp(-values))


class PriorMapTestFullPascal(PriorMapTest):
    '''
    use the full prior computed on pascal 100%
    '''

    def load_prior(self):
        prior_path = '/home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_partial_raw_100_1.json'
        kwargs = {
            'nb_classes': self.nb_classes,
            'max_weight': 0.6,
            'mode': 'test'
        }

        self.prior = priors.ConditionalRawPrior(prior_path, **kwargs)


class PriorMapTestCoco(PriorMapTest):
    '''
    Using partial prior computed on coco
    '''

    def load_prior(self):
        prior_path = '/home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_coco14_partial_100_1.json'
        kwargs = {
            'nb_classes': self.nb_classes,
            'prior_type': 'partial',
            'mode': 'test'
        }

        self.prior = priors.ConditionalPrior(prior_path, **kwargs)


# python3 prior_map_test.py 2 fullpascal
# python3 prior_map_test.py 0 coco
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    prior_type = sys.argv[2]

    if prior_type == 'fullpascal':
        mapper = PriorMapTestFullPascal()
        print('prior lookup counts')
        pprint(mapper.prior.counts)
    elif prior_type == 'coco':
        mapper = PriorMapTestCoco()

    mapper.compute_prior_map()
