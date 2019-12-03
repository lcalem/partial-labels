import copy
import itertools
import json
import math
import os
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
    for letter in all_letters:
        context_letters = copy.copy(all_letters)
        context_letters.remove(letter)

        full_contexts = get_full_contexts(context_letters)

        # normalization: sum over the same context for each version 0 and 1: matrix['ae1']['context1'] = matrix['ae1']['context1'] / matrix['a1']['context1'] + matrix['a0']['context1']
        for context in full_contexts:
            one_context = cooc_matrix['%s1' % letter][context]
            zero_context = cooc_matrix['%s0' % letter][context]

            # both are None: 0.5 for each
            if (one_context == 0.0) and (zero_context == 0.0):
                cooc_matrix['%s1' % letter][context] = 0.5
                cooc_matrix['%s0' % letter][context] = 0.5

            # missing one value: 0.4
            elif one_context == 0.0:
                cooc_matrix['%s1' % letter][context] = 0.4

            elif zero_context == 0.0:
                cooc_matrix['%s0' % letter][context] = 0.4

            # updated values if needed
            one_context = cooc_matrix['%s1' % letter][context]
            zero_context = cooc_matrix['%s0' % letter][context]

            # actual normalization
            total = one_context + zero_context

            cooc_matrix['%s0' % letter][context] /= total
            cooc_matrix['%s1' % letter][context] /= total

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


class PartialPriorCreator(object):

    def __init__(self):
        self.class_info = load_ids()
        self.nb_classes = len(self.class_info)
        assert self.nb_classes == 20

        self.id2short = {data['id']: data['short'] for data in self.class_info.values()}

    def get_partial_contexts(self, current_short, nb=2):
        '''
        '''
        partial_contexts = list()
        shorts = copy.copy(list(self.id2short.values()))
        shorts.remove(current_short)

        for partial_combination in itertools.combinations(shorts, r=nb):
            for nb_combination in itertools.product([0, 1], repeat=nb):
                partial_contexts.append('%s%s_%s%s' % (partial_combination[0], nb_combination[0], partial_combination[1], nb_combination[1]))

        assert len(partial_contexts) == math.factorial(self.nb_classes - 1) / (math.factorial(nb) * math.factorial(self.nb_classes - 1 - nb)) * len(list(itertools.product([0, 1], repeat=nb)))
        return partial_contexts

    def create_partial_from_full(self, full_matrix):
        partial_prior = defaultdict(lambda: defaultdict(lambda: 0))

        count_total = 0
        count_full = 0
        count_half = 0
        count_none = 0

        # for all the classes (20)
        for pascal_id, pascal_short in self.id2short.items():

            # for each possible context (171 * 4)
            for partial_context in self.get_partial_contexts(pascal_short):

                count_total += 1
                if count_total % 1000 == 0:
                    print('done %s partial contexts' % count_total)

                partial_parts = partial_context.split('_')

                # retrieve relevant values from the raw matrix
                one_value = sum([val for fc, val in full_matrix['%s1' % pascal_short].items() if any([fc_part in partial_parts for fc_part in fc.split('_')])])
                zero_value = sum([val for fc, val in full_matrix['%s0' % pascal_short].items() if any([fc_part in partial_parts for fc_part in fc.split('_')])])

                # normalization + filling missing values
                if one_value == 0.0 and zero_value == 0.0:
                    count_none += 1
                    one_value = 0.5
                    zero_value = 0.5

                elif one_value == 0.0:
                    count_half += 1
                    one_value = 0.4
                    zero_value = 0.6

                elif zero_value == 0.0:
                    count_half += 1
                    zero_value = 0.4
                    one_value = 0.6

                else:
                    count_full += 1

                partial_prior['%s0' % pascal_short][partial_context] = zero_value / (one_value + zero_value)
                partial_prior['%s1' % pascal_short][partial_context] = one_value / (one_value + zero_value)

        print('created %s context entries, %s had full information, %s half information and %s no information' % (count_total, count_full, count_half, count_none))
        return partial_prior


class CocoPrior(PartialPriorCreator):

    def __init__(self):
        PartialPriorCreator.__init__(self)
        self.coco2voc = {data['coco_id']: data['id'] for data in self.class_info.values()}

    def get_context_key(self, example_gt, current_id):
        '''
        will get the pascal context (length 19) out of the coco example gt (length 80) for the
        current coco id (that is included in pascal)
        '''
        context_parts = defaultdict(lambda: 0)

        for i, val in enumerate(example_gt):
            if (i not in self.coco2voc) or (i == current_id) or (val == 0):
                continue

            assert val in [-1, 1]

            pascal_id = self.coco2voc[i]
            context_short = self.id2short[pascal_id]
            context_val = 0 if val == -1 else 1

            context_parts[context_short] += context_val  # we add zeros for -1 classes so the resulting is +1 if it has been +1 al least once and zero if its always -1

        return '_'.join(['%s%s' % (context_key, 1 if context_parts[context_key] > 0 else 0) for context_key in sorted(context_parts)])

    def create_coco_prior_full(self, dataset_path, known_prop=10):
        '''
        create the full prior matrix (full = contexts contain 19 classes) from the coco dataset
        this will create two matrices:
            - the raw counting matrix, without any normalization
            - the normalized partial matrix, containing only partial contexts

            the number of partial contexts depends on the known labels proportion, eg. 10% on pascal is 2 classes so
            each partial context is 2 classes
        '''
        prior_matrix = defaultdict(lambda: defaultdict(lambda: 0))
        count = 0
        combinations = set()

        # 1. raw counting matrix
        with open(dataset_path, 'r') as f_in:
            for line in f_in:
                count += 1
                if count % 1000 == 0:
                    print('done %s lines' % count)

                parts = [int(x) for x in line.strip().split(',')]
                gt = parts[1:]  # -1 / 0 / +1

                # only iterate on the coco indices that are also in the pascal dataset
                for coco_id in self.coco2voc.keys():
                    pascal_id = self.coco2voc[coco_id]
                    class_val = gt[coco_id]

                    # we don't take information if we don't know the value for the current class
                    if class_val == 0:
                        continue

                    class_key = '%s%s' % (self.id2short[pascal_id], 0 if class_val == -1 else 1)
                    context = self.get_context_key(gt, coco_id)

                    prior_matrix[class_key][context] += 1
                    combinations.add((class_key, context))

        # saving
        save_file = '%s/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_coco14_raw_%s_1.json' % (known_prop, os.environ['HOME'])
        with open(save_file, 'w+') as f_json:
            json.dump(prior_matrix, f_json)

        print('raw counting matrix from coco contains %s combinations, saved at %s' % (len(combinations), save_file))

        # 2. normalized partial matrix
        partial_prior = self.create_partial_from_full(prior_matrix)

        save_file_partial = '%s/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_coco14_partial_%s_1.json' % (known_prop, os.environ['HOME'])
        with open(save_file_partial, 'w+') as f_json:
            json.dump(partial_prior, f_json)
        print('saved at %s' % save_file_partial)


class PartialPrior(PartialPriorCreator):
    '''
    only 2-class contexts
    from the pascalvoc 100 % dataset
    '''

    def get_context_key(self, example_gt, current_id):
        '''
        will get the pascal context (length 19) example gt (length 20) for the current class id
        '''
        context_parts = defaultdict(lambda: 0)

        for i, val in enumerate(example_gt):
            if (i == current_id) or (val == 0):
                continue

            assert val in [-1, 1]

            context_short = self.id2short[i]
            context_val = 0 if val == -1 else 1

            context_parts[context_short] += context_val  # we add zeros for -1 classes so the resulting is +1 if it has been +1 al least once and zero if its always -1

        return '_'.join(['%s%s' % (context_key, 1 if context_parts[context_key] > 0 else 0) for context_key in sorted(context_parts)])

    def create_partial_prior(self, dataset_path, known_prop=10):
        '''
        create the full prior matrix (full = contexts contain 19 classes) from the coco dataset
        this will create two matrices:
            - the raw counting matrix, without any normalization
            - the normalized partial matrix, containing only partial contexts

            the number of partial contexts depends on the known labels proportion, eg. 10% on pascal is 2 classes so
            each partial context is 2 classes
        '''
        prior_matrix = defaultdict(lambda: defaultdict(lambda: 0))
        count = 0
        combinations = set()

        # 1. raw counting matrix
        with open(dataset_path, 'r') as f_in:
            for line in f_in:
                count += 1
                if count % 1000 == 0:
                    print('done %s lines' % count)

                parts = [int(x) for x in line.strip().split(',')]
                gt = parts[1:]  # -1 / 0 / +1

                for cid, cval in enumerate(gt):
                    assert cval in [-1, 1]

                    class_key = '%s%s' % (self.id2short[cid], 0 if cval == -1 else 1)
                    context = self.get_context_key(gt, cid)

                    prior_matrix[class_key][context] += 1
                    combinations.add((class_key, context))

        # saving
        save_file = '%s/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_partial_raw_100_1.json' % os.environ['HOME']
        with open(save_file, 'w+') as f_json:
            json.dump(prior_matrix, f_json)

        print('raw counting matrix from pascal 100%% contains %s combinations, saved at %s' % (len(combinations), save_file))

        # 2. normalized partial matrix
        partial_prior = self.create_partial_from_full(prior_matrix)

        save_file_partial = '%s/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_partial_100_1.json' % os.environ['HOME']
        with open(save_file_partial, 'w+') as f_json:
            json.dump(partial_prior, f_json)
        print('saved at %s' % save_file_partial)


class PartialPrior(PartialPriorCreator):
    '''
    only 2-class contexts
    from the pascalvoc 100 % dataset
    '''

    def get_context_key(self, example_gt, current_id):
        '''
        will get the pascal context (length 19) example gt (length 20) for the current class id
        '''
        context_parts = defaultdict(lambda: 0)

        for i, val in enumerate(example_gt):
            if (i == current_id) or (val == 0):
                continue

            assert val in [-1, 1]

            context_short = self.id2short[i]
            context_val = 0 if val == -1 else 1

            context_parts[context_short] += context_val  # we add zeros for -1 classes so the resulting is +1 if it has been +1 al least once and zero if its always -1

        return '_'.join(['%s%s' % (context_key, 1 if context_parts[context_key] > 0 else 0) for context_key in sorted(context_parts)])

    def create_partial_prior(self, dataset_path, known_prop=10):
        '''
        create the full prior matrix (full = contexts contain 19 classes) from the coco dataset
        this will create two matrices:
            - the raw counting matrix, without any normalization
            - the normalized partial matrix, containing only partial contexts

            the number of partial contexts depends on the known labels proportion, eg. 10% on pascal is 2 classes so
            each partial context is 2 classes
        '''
        prior_matrix = defaultdict(lambda: defaultdict(lambda: 0))
        count = 0
        combinations = set()

        # 1. raw counting matrix
        with open(dataset_path, 'r') as f_in:
            for line in f_in:
                count += 1
                if count % 1000 == 0:
                    print('done %s lines' % count)

                parts = [int(x) for x in line.strip().split(',')]
                gt = parts[1:]  # -1 / 0 / +1

                for cid, cval in enumerate(gt):
                    assert cval in [-1, 1]

                    class_key = '%s%s' % (self.id2short[cid], 0 if cval == -1 else 1)
                    context = self.get_context_key(gt, cid)

                    prior_matrix[class_key][context] += 1
                    combinations.add((class_key, context))

        # saving
        save_file = '%s/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_partial_raw_100_1.json' % os.environ['HOME']
        with open(save_file, 'w+') as f_json:
            json.dump(prior_matrix, f_json)

        print('raw counting matrix from pascal 100%% contains %s combinations, saved at %s' % (len(combinations), save_file))

        # 2. normalized partial matrix
        partial_prior = self.create_partial_from_full(prior_matrix)

        save_file_partial = '%s/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_partial_100_1.json' % os.environ['HOME']
        with open(save_file_partial, 'w+') as f_json:
            json.dump(partial_prior, f_json)
        print('saved at %s' % save_file_partial)



class RawPartialPrior(object):
    '''
    From a PARTIAL training set of a given know labels %:
    - compute and store the raw counting matrix for each partial context
    '''
    def __init__(self):
        self.class_info = load_ids()
        self.nb_classes = len(self.class_info)
        assert self.nb_classes == 20

        self.id2short = {data['id']: data['short'] for data in self.class_info.values()}

    def get_context_key(self, example_gt, current_id, known_prop):
        '''
        length N depends on known labels % (100% = 20, 90% = 18, etc)

        will get the pascal partial context (length N-1)
        from the example gt (length N) for the current class id
        '''
        context_parts = defaultdict(lambda: 0)

        for i, val in enumerate(example_gt):
            if (i == current_id) or (val == 0):
                continue

            assert val in [-1, 1]

            context_short = self.id2short[i]
            context_val = 0 if val == -1 else 1

            context_parts[context_short] += context_val  # we add zeros for -1 classes so the resulting is +1 if it has been +1 al least once and zero if its always -1

        if len(context_parts) != (known_prop * self.nb_classes / 100) - 1:
            print(example_gt)
            print(current_id)
            pprint(context_parts)
            raise

        return '_'.join(['%s%s' % (context_key, 1 if context_parts[context_key] > 0 else 0) for context_key in sorted(context_parts)])

    def create_partial_prior(self, dataset_path):
        '''
        '''
        raw_matrix = defaultdict(lambda: defaultdict(lambda: 0))
        count = 0
        combinations = set()

        known_prop = int(dataset_path.split('_')[-2])
        assert known_prop in [10, 20, 30, 40, 50, 60, 70, 80, 90]

        # 1. raw counting matrix
        with open(dataset_path, 'r') as f_in:
            for line in f_in:
                count += 1
                if count % 1000 == 0:
                    print('done %s lines' % count)

                parts = [int(x) for x in line.strip().split(',')]
                gt = parts[1:]  # -1 / 0 / +1

                for cid, cval in enumerate(gt):
                    # we don't take information if we don't know the value for the current class
                    if cval == 0:
                        continue

                    assert cval in [-1, 1]

                    class_key = '%s%s' % (self.id2short[cid], 0 if cval == -1 else 1)
                    context = self.get_context_key(gt, cid, known_prop)

                    raw_matrix[class_key][context] += 1
                    combinations.add((class_key, context))

        # saving
        save_file = '%s/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_partial_raw_%s_1.json' % (os.environ['HOME'], known_prop)
        with open(save_file, 'w+') as f_json:
            json.dump(raw_matrix, f_json)

        print('raw counting matrix from pascal %s%% contains %s combinations, saved at %s' % (known_prop, len(combinations), save_file))


# python3 prior_matrix.py create /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_10_1.csv
# python3 prior_matrix.py full /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_10_1.csv
# python3 prior_matrix.py viz /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix4_trainval_partial_10_1.json
# python3 prior_matrix.py check_full /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/prior_matrix_full_trainval_partial_10_1.json
# python3 prior_matrix.py coco /home/caleml/datasets/mscoco/annotations/multilabel_train2014_partial_100_1.csv
# python3 prior_matrix.py partial /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_100_1.csv
# python3 prior_matrix.py raw_partial /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_90_1.csv
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
    elif action == 'coco':
        prior = CocoPrior()
        prior.create_coco_prior_full(filepath)
    elif action == 'partial':
        prior = PartialPrior()
        prior.create_partial_prior(filepath)
    elif action == 'raw_partial':
        prior = RawPartialPrior()
        prior.create_partial_prior(filepath)
    else:
        raise Exception('AYAYAYAYAY')
