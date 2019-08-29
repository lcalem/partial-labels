import sys

import numpy as np


def create_cooc(filepath):
    cooc = np.zeros((20, 20), dtype=np.float32)
    count_ones = 0

    with open(filepath, 'r') as f_in:
        for line in f_in:
            parts = [int(x) for x in line.strip().split(',')]

            ones_idx = [i - 1 for i in range(len(parts[1:]) + 1) if parts[i] == 1]
            count_ones += len(ones_idx)

            for index in ones_idx:
                # cooc[index][index] += 1
                for other_idx in ones_idx:
                    if other_idx == index:
                        continue
                    cooc[index][other_idx] += 1

    # normalization per line
    total_ones = 0
    for i in range(len(cooc)):
        # print(sum(cooc[i]))
        print(cooc[i])
        total_ones += sum(cooc[i])
        cooc[i] /= sum(cooc[i])

    print(cooc)

    print("total", total_ones)
    # saving
    save_file = filepath.replace('annotations_multilabel', 'cooc_matrix').replace('.csv', '')
    np.save(save_file, cooc)


# python3 cooc_matrix.py /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_100_1.csv
if __name__ == '__main__':
    filepath = sys.argv[1]
    create_cooc(filepath)
