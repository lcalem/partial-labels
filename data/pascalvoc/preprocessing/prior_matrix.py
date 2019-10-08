import copy
import itertools
import json
import sys

import numpy as np

from collections import defaultdict

from data.pascalvoc.preprocessing.utils import load_ids

from pprint import pprint

from config.config import cfg

ALL_LETTERS = ['a', 'i', 'p', 'v']


def viz_prior(matrix_path):
    data = json.load(matrix_path)
    pprint(data)


def get_all_contexts(letter):
    '''
    for context a: gives the 8 full contexts i0_p0_v0, i0_p0_v1, etc
    also gives the list of all 15 partial contexts i0, p0, v0, i0_p0, etc
    '''
    context_letters = copy.copy(ALL_LETTERS)
    context_letters.remove(letter)

    full_contexts = list()
    partial_contexts = list()

    # creation of full context
    for combination in itertools.product([0, 1], repeat=len(context_letters)):
        full_contexts.append('_'.join(['%s%s' % (context_letters[i], combination[i]) for i in range(len(context_letters))]))

    # partial ones
    partial_contexts = list()
    for i in range(len(context_letters) - 1):
        i += 1
        for partial in itertools.combinations(context_letters, r=i):
            for combination in itertools.product([0, 1], repeat=i):
                context = '_'.join(['%s%s' % (partial[j], combination[j]) for j in range(i)])
                partial_contexts.append(context)

    return full_contexts, partial_contexts


def create_prior(dataset_path):
    '''
    order of the keys:

    matrix['a1']['i1_p0_v1'] gives the probability of the class animal given the presence of the classes indoor and vehicle and the absence of the classes person in the image
    matrix['a1']['i0'] gives the marginalization over all complete-context values with i=0
    '''
    classes_order = ['animal', 'indoor', 'person', 'vehicle']
    cooc_superclass = defaultdict(lambda: defaultdict(lambda: 0.0))

    class_info = load_ids()
    id2superclass = {info['id']: info['superclass'] for info in class_info.values()}

    with open(dataset_path, 'r') as f_in:
        for line in f_in:
            parts = [int(x) for x in line.strip().split(',')]

            # ones
            ones_idx = [i - 1 for i in range(len(parts[1:]) + 1) if parts[i] == 1]
            ones_superclasses = set([id2superclass[idx] for idx in ones_idx])

            for superclass in ones_superclasses:
                other_classes = '_'.join(['%s%s' % (sc[0], (1 if sc in ones_superclasses else 0)) for sc in classes_order if sc != superclass])
                cooc_superclass['%s1' % superclass[0]][other_classes] += 1

            # zeros
            zeros_idx = [i - 1 for i in range(len(parts[1:]) + 1) if parts[i] == 0]
            zeros_superclasses = set([id2superclass[idx] for idx in zeros_idx])

            for superclass in zeros_superclasses:
                other_classes = '_'.join(['%s%s' % (sc[0], (1 if sc in zeros_superclasses else 0)) for sc in classes_order if sc != superclass])
                cooc_superclass['%s0' % superclass[0]][other_classes] += 1

    pprint(cooc_superclass)

    # fill missing values, normalize and compute incomplete contexts
    for letter in ALL_LETTERS:

        full_contexts, partial_contexts = get_all_contexts(letter)
        print(full_contexts)
        print(partial_contexts)

        # normalization: sum over the same context for each version 0 and 1: matrix['a1']['i1_p0_v1'] = matrix['a1']['i1_p0_v1'] / matrix['a1']['i1_p0_v1'] + matrix['a0']['i1_p0_v1']
        for context in full_contexts:
            print('doing context %s' % context)
            one_context = cooc_superclass['%s1' % letter][context]
            zero_context = cooc_superclass['%s0' % letter][context]

            # both are None: 0.5 for each
            if (one_context == 0.0) and (zero_context == 0.0):
                cooc_superclass['%s1' % letter][context] = 0.5
                cooc_superclass['%s0' % letter][context] = 0.5

            # only one is None: epsilon
            elif one_context == 0.0:
                cooc_superclass['%s1' % letter][context] = cfg.EPSILON

            elif zero_context == 0.0:
                cooc_superclass['%s0' % letter][context] = cfg.EPSILON

            # updated values if needed
            one_context = cooc_superclass['%s1' % letter][context]
            zero_context = cooc_superclass['%s0' % letter][context]
            print('updated one context %s' % one_context)
            print('updated zero context %s' % zero_context)

            total = one_context + zero_context

            cooc_superclass['%s0' % letter][context] /= total
            cooc_superclass['%s1' % letter][context] /= total

        # precompute the incomplete part
        for partial in partial_contexts:
            partial_parts = partial.split('_')

            # we add all the values of the contexts that match the partial one
            sum_one = sum([val for ctxt, val in cooc_superclass['%s1' % letter].items() if any([part in ctxt for part in partial_parts])])
            sum_zero = sum([val for ctxt, val in cooc_superclass['%s0' % letter].items() if any([part in ctxt for part in partial_parts])])

            cooc_superclass['%s0' % letter][partial] = sum_zero / (sum_zero + sum_one)
            cooc_superclass['%s1' % letter][partial] = sum_one / (sum_zero + sum_one)

    print('\nFinal matrix')
    pprint(cooc_superclass)

    # saving
    save_file = filepath.replace('annotations_multilabel', 'prior_matrix3').replace('.csv', '.json')
    with open(save_file, 'w+') as f_json:
        json.dump(cooc_superclass, f_json)


# python3 prior_matrix.py create /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_10_1.csv
# python3 prior_matrix.py viz /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix3_trainval_partial_10_1.json
if __name__ == '__main__':
    action = sys.argv[1]
    filepath = sys.argv[2]

    if action == 'create':
        create_prior(filepath)
    elif action == 'viz':
        viz_prior(filepath)
    else:
        raise Exception('AYAYAYAYAY')
