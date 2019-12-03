import sys

import numpy as np


def count_zeros(dataset_path):
    count_zeros = 0
    total = 0

    with open(dataset_path, 'r') as f_in:
        for line in f_in:
            ground_truths = np.array([int(v) for v in line.strip().split(',')[1:]])
            total += len(ground_truths)
            count_zeros += np.count_nonzero(ground_truths == 0)   # actually counts zeros

    print('Dataset %s has %s zeros (%2f %%)' % (dataset_path, count_zeros, count_zeros * 100 / total))


# python3 count_zeros.py /share/DEEPLEARNING/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_train.csv
# python3 count_zeros.py /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_100_1.csv
# python3 count_zeros.py /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_10_1.csv
if __name__ == '__main__':
    count_zeros(sys.argv[1])
