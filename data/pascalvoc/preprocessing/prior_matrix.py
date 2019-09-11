import json
import sys

import numpy as np

from collections import defaultdict

from data.pascalvoc.preprocessing.utils import load_ids

from pprint import pprint


def create_prior(dataset_path):
    '''
    order of the keys:

    matrix['a1']['i1_p0_v1'] gives the probability of the class animal given the presence of the classes indoor and vehicle and the absence of the classes person in the image
    '''
    classes_order = ['animal', 'indoor', 'person', 'vehicle']
    cooc_superclass = defaultdict(lambda: defaultdict(lambda: 0))

    class_info = load_ids()
    id2superclass = {info['id']: info['superclass'] for info in class_info.values()}

    with open(dataset_path, 'r') as f_in:
        for line in f_in:
            parts = [int(x) for x in line.strip().split(',')]

            ones_idx = [i - 1 for i in range(len(parts[1:]) + 1) if parts[i] == 1]
            ones_superclasses = set([id2superclass[idx] for idx in ones_idx])

            for superclass in ones_superclasses:
                other_classes = '_'.join(['%s%s' % (sc[0], (1 if sc in ones_superclasses else 0)) for sc in classes_order if sc != superclass])
                cooc_superclass['%s1' % superclass[0]][other_classes] += 1

    pprint(cooc_superclass)

    # normalization
    for key, other in cooc_superclass.items():
        sum_marginal = 0
        for other_name, other_count in other.items():
            sum_marginal += other_count

        for other_name, other_count in other.items():
            other[other_name] /= sum_marginal

    pprint(cooc_superclass)

    # saving
    save_file = filepath.replace('annotations_multilabel', 'prior_matrix').replace('.csv', '.json')
    with open(save_file, 'w+') as f_json:
        json.dump(cooc_superclass, f_json)


# python3 prior_matrix.py /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_50_1.csv
if __name__ == '__main__':
    filepath = sys.argv[1]
    create_prior(filepath)
