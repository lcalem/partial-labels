import json
import sys

import numpy as np

from collections import defaultdict

from data.coco.utils import load_ids

from pprint import pprint


def viz_prior(prior_path, print_context='ac0_an0_ap0_el0_fo0_in0_ki0_ob1_pe1_ro1_sp0_ve1'):
    with open(prior_path, 'r') as f_in:
        data = json.load(f_in)

    for classprob, context_data in data.items():
        for context, context_val in context_data.items():
            if context == print_context:
                print('p(%s | %s) = %s' % (classprob, context, context_val))


def get_context_key(example, class_index, class_info):
    '''
    superclass context key!
    explicit computation of the context for the class
    one-liner: context = '_'.join(['%s%s' % (sc[0], (1 if sc in ones_superclasses else 0)) for sc in classes_order if sc != superclass])

    CAUTION: if there is one context where we don't have any information, the resulting context will be a PARTIAL one!
    '''
    context_parts = defaultdict(lambda: 0)

    for i, val in enumerate(example):
        if i == class_index or val == 0:
            continue
        assert val in [-1, 1]

        context_short = class_info[i]['superclass'][:2]
        context_val = 0 if val == -1 else 1

        context_parts[context_short] += context_val  # we add zeros for -1 classes so the resulting is +1 if it has been +1 al least once and zero if its always -1

    return '_'.join(['%s%s' % (context_key, 1 if context_parts[context_key] > 0 else 0) for context_key in sorted(context_parts)])


def create_prior(dataset_path):
    '''
    SUEPRCLASS PRIOR ONLY

    prior_matrix['ae1']['context'] = p(aeroplane = 1 | context)

    TODO: for now it doesn't work if we compute the prior from a partial dataset... although it doesn't explicitly fail here
    '''
    class_info = load_ids()
    superclasses = {v['superclass'] for v in class_info.values()}
    print('using %s superclasses' % (len(superclasses)))

    prior_matrix = defaultdict(lambda: defaultdict(lambda: 0))
    count = 0

    with open(dataset_path, 'r') as f_in:
        for line in f_in:
            count += 1
            if count % 1000 == 0:
                print('done %s lines' % count)

            parts = [int(x) for x in line.strip().split(',')]
            gt = parts[1:]  # -1 / 0 / +1

            # for each position in the example we compute the context and increment the corresponding prior matrix prob
            for id_class, class_gt in enumerate(gt):

                # we don't add any information from unknown class value
                if class_gt == 0:
                    continue

                assert class_gt in [-1, 1], 'found wrong gt value %s for example %s' % (class_gt, parts[0])

                current_superclass = class_info[id_class]['superclass']
                superclass_short = current_superclass[:2]

                context = get_context_key(gt, id_class, class_info)
                key = '%s%s' % (superclass_short, 0 if class_gt == -1 else 1)

                prior_matrix[key][context] += 1

    print('matrix before normalization')
    pprint(prior_matrix)

    # normalization
    for superclass in superclasses:

        supershort = superclass[:2]
        seen_contexts = set()

        # first we iterate on the contexts that are available for the superclass = 1 prob
        for context_1, c1_val in prior_matrix['%s1' % supershort].items():
            seen_contexts.add(context_1)

            # if context is available in both 1 and 0 superclass keys: normalization (c1_val = c1_val / (c1_val+c0_val))
            if context_1 in prior_matrix['%s0' % supershort]:
                c0_val = prior_matrix['%s0' % supershort][context_1]
                total = c1_val + c0_val

                prior_matrix['%s1' % supershort][context_1] = c1_val / total
                prior_matrix['%s0' % supershort][context_1] = c0_val / total

            # if it's not, we put the c1 at 0.6 and c0 at 0.4
            else:
                prior_matrix['%s1' % supershort][context_1] = 0.6
                prior_matrix['%s0' % supershort][context_1] = 0.4

        # then we do the same for the contexts of the superclass = 0 prob (that we didn't see in the first sweep)
        for context_0, c0_val in prior_matrix['%s0' % supershort].items():
            if context_0 in seen_contexts:
                continue

            # if we are here it means context_0 is available for superclass = 0 but not for superclass = 1: we apply our 0.6 / 0.4 balance
            assert context_0 not in prior_matrix['%s1' % supershort]
            prior_matrix['%s1' % supershort][context_0] = 0.4
            prior_matrix['%s0' % supershort][context_0] = 0.6

    # pprint(prior_matrix)

    # saving
    save_file = filepath.replace('multilabel', 'prior_matrix1').replace('.csv', '.json')
    with open(save_file, 'w+') as f_json:
        json.dump(prior_matrix, f_json)


# python3 prior_matrix.py create /home/caleml/datasets/mscoco/annotations/multilabel_train2014_partial_100_1.csv
# python3 prior_matrix.py viz /home/caleml/datasets/mscoco/annotations/prior_matrix1_train2014_partial_100_1.json
if __name__ == '__main__':
    filepath = sys.argv[2]
    action = sys.argv[1]

    if action == 'create':
        create_prior(filepath)
    elif action == 'viz':
        viz_prior(filepath)
