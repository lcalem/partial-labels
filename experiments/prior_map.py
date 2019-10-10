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


def compute_prior_map():
    '''
    1. Load model trained without relabeling
    2. for the train dataset, get the y_v and the y_f from the model and prior
    3. per class, take the ground truth values for each zero value of the 10% known label dataset
    4. for each of these values, compute a mAP between y_gt and y_v AND the mAP between y_gt and y_f
    '''
    batch_size = 16
    nb_classes = 20
    prop = 10

    class_info = load_ids()
    id2name = {int(info['id']): info['name'] for info in class_info.values()}

    data_dir = '/home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/'
    weights_path = '/home/caleml/partial_experiments/exp_20191009_1546_baseline_logits/weights_10_015_00.h5'
    gt_path = os.path.join(data_dir, 'Annotations', 'annotations_multilabel_trainval_partial_100_1.csv')
    prior_path = '/home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix3_trainval_partial_10_1.json'

    # 1. load model
    model = BaselineLogits('%s/partial_experiments/' % os.environ['HOME'], nb_classes, prop)
    model.load_weights(weights_path)

    # load prior
    prior = priors.ConditionalPrior(prior_path, nb_classes=nb_classes)

    # load GT
    gt = load_annotations(gt_path)

    # 2. train dataset
    dataset_train = PascalVOC(data_dir, batch_size, 'trainval', x_keys=['image', 'image_id'], y_keys=['multilabel'], p=prop)

    y_true_k = defaultdict(list)
    y_v_k = defaultdict(list)
    y_f_k = defaultdict(list)

    for i_batch in range(len(dataset_train)):
        x_batch, y_batch = dataset_train[i_batch]
        y_true = y_batch[0]
        bs, K = y_true.shape
        assert bs == batch_size
        assert K == nb_classes

        y_pred = model.predict(x_batch)   # y_pred[0] -> y_v, y_pred[1] -> logits of y_v
        y_v = np.asarray(y_pred[0])

        p_k = prior.compute_pk(y_true)
        y_f = prior.combine(y_v, p_k)

        batch_img_ids = np.array(x_batch[1])[:, 0]

        # add values for each class where y_batch is 0
        for i_class in range(nb_classes):
            indexes_zeros = [i for i in range(bs) if y_true[i][i_class] == 0]

            # 3. get ground truth values matching the zeros from the 10% dataset
            batch_gt_k = [gt[img_id][i_class] for img_id in batch_img_ids]  # all GT for the batch
            batch_gt_k_zeros = [elt for i, elt in enumerate(batch_gt_k) if i in indexes_zeros]

            y_true_k[i_class].append(batch_gt_k_zeros)

            # add y_v for the zeros
            y_v_zeros = [elt[i_class] for i, elt in enumerate(y_v) if i in indexes_zeros]
            y_v_k[i_class].append(y_v_zeros)

            # add y_f for the zeros
            y_f_zeros = [elt[i_class] for i, elt in enumerate(y_f) if i in indexes_zeros]
            y_f_k[i_class].append(y_f_zeros)

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
    all_map_p = list()

    for i_class in range(nb_classes):

        # print(y_true_k[i_class])

        y_gt_val = np.concatenate(y_true_k[i_class]).astype(np.float64)
        y_v_val = np.concatenate(y_v_k[i_class]).astype(np.float64)
        y_f_val = np.concatenate(y_f_k[i_class]).astype(np.float64)

        # print(y_gt_val.shape)
        # print(y_v_val.shape)

        # print(type(y_gt_val))
        # print(y_gt_val[0])

        map_visual = metrics.average_precision_score(y_gt_val, y_v_val)
        map_prior = metrics.average_precision_score(y_gt_val, y_f_val)

        print('for class %s (%s), map_visual %s, map_prior %s' % (i_class, id2name[i_class], map_visual, map_prior))

        all_map_v.append(map_visual)
        all_map_p.append(map_prior)

    print('average map visual %s' % (sum(all_map_v) / len(all_map_v)))
    print('average map prior %s' % (sum(all_map_p) / len(all_map_p)))


def load_annotations(annotations_path):
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


# python3 prior_map.py 1
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    compute_prior_map()
