import argparse
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


class PriorMap(object):

    def __init__(self, prior_path, weights_path, comb_method, prop, mode, use_superclass, allow_marginalization):
        self.batch_size = 16
        self.nb_classes = 20
        self.prop = prop
        self.context_length = (self.prop * self.nb_classes / 100) - 1

        self.mode = mode

        class_info = load_ids()
        self.id2name = {int(info['id']): info['name'] for info in class_info.values()}

        self.data_dir = '/home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/'
        self.weights_path = weights_path
        self.gt_path = os.path.join(self.data_dir, 'Annotations', 'annotations_multilabel_trainval_partial_100_1.csv')

        # 1. load model
        self.model = BaselineLogits('%s/partial_experiments/' % os.environ['HOME'], self.nb_classes, self.prop)
        self.model.load_weights(self.weights_path)

        self.prior = self.load_prior(prior_path, comb_method, use_superclass, allow_marginalization)

        # load GT
        self.gt = self.load_annotations(self.gt_path)

    def load_prior(self, prior_path, comb_method, use_superclass, allow_marginalization):
        kwargs = {
            'matrix_path': prior_path,
            'nb_classes': self.nb_classes,
            'context_length': self.context_length,
            'comb_method': comb_method,
            'use_superclass': use_superclass,
            'mode': self.mode,
            'allow_marginalization': allow_marginalization
        }

        return priors.ConditionalPrior(**kwargs)

    def compute_prior_map(self):
        '''
        1. Load model trained without relabeling
        2. for the train dataset, get the y_v and the y_f from the model and prior
        3. per class, take the ground truth values for each zero value of the 10% known label dataset
        4. for each of these values, compute a mAP between y_gt and y_v AND the mAP between y_gt and y_f
        '''

        # 2. train dataset
        dataset_train = PascalVOC(self.data_dir, self.batch_size, 'trainval', x_keys=['image', 'image_id'], y_keys=['multilabel'], p=self.prop)

        y_true_k = defaultdict(list)
        y_v_k = defaultdict(list)
        y_f_k = defaultdict(list)
        y_p_k = defaultdict(list)

        for i_batch in range(len(dataset_train)):
            if i_batch % 100 == 0:
                print('doing batch %s' % i_batch)

            x_batch, y_batch = dataset_train[i_batch]
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
                indexes_zeros = [i for i in range(bs) if y_true[i][i_class] == 0]

                # 3. get ground truth values matching the zeros from the 10% dataset
                batch_gt_k = [self.gt[img_id][i_class] for img_id in batch_img_ids]  # all GT for the batch
                batch_gt_k_zeros = [elt for i, elt in enumerate(batch_gt_k) if i in indexes_zeros]

                y_true_k[i_class].append(batch_gt_k_zeros)

                # add y_v for the zeros
                y_v_zeros = [elt[i_class] for i, elt in enumerate(y_v_logits) if i in indexes_zeros]
                y_v_k[i_class].append(y_v_zeros)

                # add y_f for the zeros
                y_f_zeros = [elt[i_class] for i, elt in enumerate(y_f) if i in indexes_zeros]
                y_f_k[i_class].append(y_f_zeros)

                # add y_p for the zeros
                assert p_k_logits.shape == (self.batch_size, self.nb_classes)
                y_p_zeros = [elt[i_class] for i, elt in enumerate(p_k_logits) if i in indexes_zeros]
                y_p_k[i_class].append(y_p_zeros)

            # print('y_true')
            # print(y_true)

            # print('pk logits')
            # print(p_k_logits)

            # print('y_v size %s' % str(y_v_logits.shape))
            # print('y_f size %s' % str(y_f.shape))

            # print('y_true_k')
            # pprint(y_true_k)

            # print('y_v_k')
            # pprint(y_v_k)

            # print('y_f_k')
            # pprint(y_f_k)

            # print('y_p_k')
            # pprint(y_p_k)

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

        print('average map visual, %s' % (sum(all_map_v) / len(all_map_v)))
        print('average map combined, %s' % (sum(all_map_f) / len(all_map_f)))
        print('average map prior only, %s' % (sum(all_map_p) / len(all_map_p)))

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


class PriorMapAlpha(PriorMap):
    def __init__(self, matrix_path, weights_path, comb_method, prop, mode, use_superclass, allow_marginalization, alpha):
        self.alpha = alpha

        PriorMap.__init__(self, matrix_path, weights_path, comb_method, prop, mode, use_superclass, allow_marginalization)

    def load_prior(self, prior_path, comb_method, use_superclass, allow_marginalization):
        # load prior
        kwargs = {
            'matrix_path': prior_path,
            'nb_classes': self.nb_classes,
            'context_length': self.context_length,
            'comb_method': comb_method,
            'alpha': self.alpha,
            'use_superclass': use_superclass,
            'mode': self.mode,
            'allow_marginalization': allow_marginalization,
            'max_weight': 0.6,
        }

        return priors.ConditionalPrior(prior_path, **kwargs)


class RawPriorMap(PriorMap):
    def load_prior(self, prior_path, comb_method, use_superclass, allow_marginalization):
        kwargs = {
            'matrix_path': prior_path,
            'nb_classes': self.nb_classes,
            'context_length': self.context_length,
            'comb_method': comb_method,
            'use_superclass': use_superclass,
            'mode': self.mode,
            'allow_marginalization': allow_marginalization,
            'max_weight': 0.6,
        }

        return priors.ConditionalRawPrior(**kwargs)


class PartialRawPriorMap(PriorMap):
    def load_prior(self, prior_path, comb_method, use_superclass, allow_marginalization):
        kwargs = {
            'matrix_path': prior_path,
            'nb_classes': self.nb_classes,
            'context_length': self.context_length,
            'comb_method': comb_method,
            'use_superclass': use_superclass,
            'mode': self.mode,
            'allow_marginalization': allow_marginalization,
            'max_weight': 0.6,
        }

        return priors.ConditionalPartialRawPrior(**kwargs)



# python3 prior_map.py 0 -w exp_20191009_1546_baseline_logits/weights_10_015_00.h5
# python3 prior_map.py 0 /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix4_trainval_partial_10_1.json simple super
# python3 prior_map.py 0 /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_full_trainval_partial_10_1.json simple full
# python3 prior_map.py 0 /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_coco14_partial_100_1.json simple partial
# python3 prior_map.py 0 /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_coco14_partial_100_1.json alpha partial 0
# python3 prior_map.py 0 /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_partial_100_1.json simple partial
# python3 prior_map.py -m prior_matrix_partial_raw_90_1.json -w exp_20191028_1522_baseline_logits_pv_baseline/weights_90_010.h5 -pr raw -p 90 -g 1
# python3 prior_map.py -m prior_matrix_coco14_raw_100_1.json -w exp_20191028_1522_baseline_logits_pv_baseline/weights_90_010.h5 -pr raw_partial -p 90 -g 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', required=True, help='# of the gpu device')
    parser.add_argument('--matrix_path', '-m', required=True)
    parser.add_argument('--weights_path', '-w', required=True)
    parser.add_argument('--prop', '-p', help='the specific percentage of known labels', type=int)

    parser.add_argument('--prior', '-pr', help='raw prior or normal prior')

    parser.add_argument('--comb_method', '-c', default='simple', help='either alpha or simple for now')
    parser.add_argument('--alpha', '-a', help='must be set if comb method is alpha', type=float)
    parser.add_argument('--super', '-s', default=False, help='superclass prior', type=bool)
    parser.add_argument('--mode', '-mo', default='train', help='train or test')
    parser.add_argument('--allow_marginalization', '-ma', default=False, type=bool)

    # options management
    args = parser.parse_args()

    matrix_path = args.matrix_path
    if not matrix_path.startswith('/'):
        matrix_path = '/home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/' + matrix_path

    weights_path = args.weights_path
    if not weights_path.startswith('/'):
        weights_path = '/home/caleml/partial_experiments/' + weights_path

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.comb_method == 'alpha':
        alpha = args.alpha
        mapper = PriorMapAlpha(matrix_path, weights_path, args.comb_method, args.prop, args.mode, args.super, args.allow_marginalization, alpha)
    elif args.prior == 'raw':
        mapper = RawPriorMap(matrix_path, weights_path, args.comb_method, args.prop, args.mode, args.super, args.allow_marginalization)
    elif args.prior == 'raw_partial':
        mapper = PartialRawPriorMap(matrix_path, weights_path, args.comb_method, args.prop, args.mode, args.super, args.allow_marginalization)
    else:
        mapper = PriorMap(matrix_path, weights_path, args.comb_method, args.prop, args.mode, args.super, args.allow_marginalization)

    mapper.compute_prior_map()
