import sys

import numpy as np


def count_ones(dataset_path):
    count_ones = 0
    total = 0

    with open(dataset_path, 'r') as f_in:
        for line in f_in:
            ground_truths = np.array([int(v) for v in line.strip().split(',')[1:]])
            total += len(ground_truths)
            count_ones += np.count_nonzero(ground_truths == 1)   # actually counts zeros

    print('Dataset %s has %s ones (%2f %%)' % (dataset_path, count_ones, count_ones * 100 / total))


# python3 count_ones.py /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_train_partial_100_1.csv
if __name__ == '__main__':
    count_ones(sys.argv[1])
