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
    with open(matrix_path, 'r') as f_in:
        data = json.load(f_in)
    pprint(data)


def get_full_contexts(context_letters):
    full_contexts = list()

    for combination in itertools.product([0, 1], repeat=len(context_letters)):
        full_contexts.append('_'.join(['%s%s' % (context_letters[i], combination[i]) for i in range(len(context_letters))]))

    print('found %s contexts for letters %s' % (len(full_contexts), str(context_letters)))
    return full_contexts


def get_all_contexts(letter):
    '''
    for context a: gives the 8 full contexts i0_p0_v0, i0_p0_v1, etc
    also gives the list of all 15 partial contexts i0, p0, v0, i0_p0, etc
    '''
    context_letters = copy.copy(ALL_LETTERS)
    context_letters.remove(letter)

    # creation of full context
    full_contexts = get_full_contexts(context_letters)

    # partial contexts
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
    create the prior with superclass clustering

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
                context = '_'.join(['%s%s' % (sc[0], (1 if sc in ones_superclasses else 0)) for sc in classes_order if sc != superclass])
                cooc_superclass['%s1' % superclass[0]][context] += 1

            # zeros
            zeros_idx = [i - 1 for i in range(len(parts[1:]) + 1) if parts[i] == -1]
            zeros_superclasses = set([id2superclass[idx] for idx in zeros_idx])

            for superclass in zeros_superclasses:
                context = '_'.join(['%s%s' % (sc[0], (1 if sc in ones_superclasses else 0)) for sc in classes_order if sc != superclass])
                cooc_superclass['%s0' % superclass[0]][context] += 1

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
    save_file = filepath.replace('annotations_multilabel', 'prior_matrix4').replace('.csv', '.json')
    with open(save_file, 'w+') as f_json:
        json.dump(cooc_superclass, f_json)


def create_full_prior(dataset_path):
    '''
    create the full matrix (19 class context)
    '''

    cooc_matrix = defaultdict(lambda: defaultdict(lambda: 0.0))

    class_info = load_ids()
    id2letter = {info['id']: info['short'] for info in class_info.values()}
    all_letters = list(id2letter.values())

    # counting of the dataset
    with open(dataset_path, 'r') as f_in:
        for line in f_in:
            parts = [int(x) for x in line.strip().split(',')]
            ground_truths = parts[1:]

            for i, gt in enumerate(ground_truths):

                letter = id2letter[i]
                value = (1 if gt == 1 else 0)
                context = list()

                # '_'.join(['%s%s' % (id2letter[cid], (1 if cid in ones_idx else 0)) for cid in range(len(ground_truths)) if cid != class_id])
                for context_j, context_gt in enumerate(ground_truths):
                    if context_j == i:
                        continue

                    context_letter = id2letter[context_j]
                    context_value = (1 if context_gt == 1 else 0)
                    context.append('%s%s' % (context_letter, context_value))

                    full_context = '_'.join(context)
                    cooc_matrix['%s%s' % (letter, value)][full_context] += 1

    # normalization and filling missing full contexts
    # for letter in all_letters:
    #     context_letters = copy.copy(all_letters)
    #     context_letters.remove(letter)

    #     full_contexts = get_full_contexts(context_letters)

    #     # normalization: sum over the same context for each version 0 and 1: matrix['ae1']['context1'] = matrix['ae1']['context1'] / matrix['a1']['context1'] + matrix['a0']['context1']
    #     for context in full_contexts:
    #         one_context = cooc_matrix['%s1' % letter][context]
    #         zero_context = cooc_matrix['%s0' % letter][context]

    #         # both are None: 0.5 for each
    #         if (one_context == 0.0) and (zero_context == 0.0):
    #             cooc_matrix['%s1' % letter][context] = 0.5
    #             cooc_matrix['%s0' % letter][context] = 0.5

    #         # missing one value: 0.4
    #         elif one_context == 0.0:
    #             cooc_matrix['%s1' % letter][context] = 0.4

    #         elif zero_context == 0.0:
    #             cooc_matrix['%s0' % letter][context] = 0.4

    #         # updated values if needed
    #         one_context = cooc_matrix['%s1' % letter][context]
    #         zero_context = cooc_matrix['%s0' % letter][context]

    #         # actual normalization
    #         total = one_context + zero_context

    #         cooc_matrix['%s0' % letter][context] /= total
    #         cooc_matrix['%s1' % letter][context] /= total

    # saving
    save_file = filepath.replace('annotations_multilabel', 'prior_matrix_full0').replace('.csv', '.json')
    with open(save_file, 'w+') as f_json:
        json.dump(cooc_matrix, f_json)

    print('saved matrix at %s' % save_file)


def check_full(matrix_path):
    '''
    will count the number of contexts containing a value
    '''
    with open(matrix_path, 'r') as f_in:
        data = json.load(f_in)

    count = 0

    for key, contexts in data.items():
        count += len(contexts)

    print(count)


# python3 prior_matrix.py create /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_10_1.csv
# python3 prior_matrix.py full /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_10_1.csv
# python3 prior_matrix.py viz /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix4_trainval_partial_10_1.json
# python3 prior_matrix.py check_full /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_full_trainval_partial_10_1.json
if __name__ == '__main__':
    action = sys.argv[1]
    filepath = sys.argv[2]

    if action == 'create':
        create_prior(filepath)
    elif action == 'full':
        create_full_prior(filepath)
    elif action == 'check_full':
        check_full(filepath)
    elif action == 'viz':
        viz_prior(filepath)
    else:
        raise Exception('AYAYAYAYAY')
