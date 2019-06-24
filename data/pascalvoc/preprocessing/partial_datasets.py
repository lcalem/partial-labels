import copy
import os
import sys

import numpy as np

from data.pascalvoc.pascalvoc import NB_CLASSES
from model.utils.config import cfg


def partal_datasets(annotations_path):
    '''
    will create partial datasets from the train dataset with keeping X% of labels
    X in [10, 20 ... 100] (10 = 90% of the labels are removed, 100 = original dataset)
    '''
    seed = cfg.RANDOM_SEED
    np.random.seed(seed)
    print('creating partial datasets for random seed %s' % seed)

    kept_proportions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    nb_dropped_indexes = [int((100 - prop) * NB_CLASSES / 100) for prop in kept_proportions]

    base_dataset = os.path.join(annotations_path, 'annotations_multilabel_train.csv')
    partial_paths = [os.path.join(annotations_path, 'annotations_multilabel_train_partial_%s_%s.csv' % (p, seed)) for p in kept_proportions]

    # maybe that's ugly
    with open(base_dataset, 'r') as f_in, \
         open(partial_paths[0], 'w+') as f_10, \
         open(partial_paths[1], 'w+') as f_20, \
         open(partial_paths[2], 'w+') as f_30, \
         open(partial_paths[3], 'w+') as f_40, \
         open(partial_paths[4], 'w+') as f_50, \
         open(partial_paths[5], 'w+') as f_60, \
         open(partial_paths[6], 'w+') as f_70, \
         open(partial_paths[7], 'w+') as f_80, \
         open(partial_paths[8], 'w+') as f_90, \
         open(partial_paths[9], 'w+') as f_100:

         for line in f_in:
            parts = line.strip().split(',')
            img_idx = parts[0]
            ground_truths = np.array([int(v) for v in parts[1:]])

            # computing indexes to drop for each proportion
            for i, prop in enumerate(kept_proportions):
                nb_indexes = nb_dropped_indexes[i]
                dropped_indexes = np.random.choice(len(ground_truths), nb_indexes, replace=False)

                gt_copy = copy.copy(ground_truths)
                gt_copy[dropped_indexes] = 0

                partial_labels_line = img_idx + ',' + ','.join([str(elt) for elt in gt_copy]) + '\n'

                # dynamically accessing the file handler... I know it's bad sorry it's just a one-off script
                locals()['f_%d' % prop].write(partial_labels_line)


# python3 partial_datasets.py /share/DEEPLEARNING/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations
if __name__ == '__main__':
    partal_datasets(sys.argv[1])
