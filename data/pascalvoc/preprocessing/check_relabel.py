import sys

import numpy as np


def check_relabel(original_file, new_file):
    '''
    '''

    base_values = dict()
    new_keys = list()

    with open(original_file, 'r') as f_old:
        for line in f_old:
            parts = line.strip().split(',')
            base_values[parts[0]] = parts[1:]

    with open(new_file, 'r') as f_new:
        for line in f_new:
            parts = line.strip().split(',')

            new_val = np.array(parts[1:]).astype(np.int64)
            old_val = np.array(base_values[parts[0]]).astype(np.int64)

            diff = np.where(new_val != old_val, new_val, -2)

            try:
                assert np.all(diff[np.where(diff != -2)] == 1)      # check we only added ones
                assert np.all(old_val[np.where(diff == 1)] == 0)    # check we added values only where the initial batch was 0
            except:
                print('image id %s' % parts[0])
                print('new val %s' % str(new_val))
                print('old val %s' % str(old_val))
                print('diff %s' % str(diff))
                raise

            new_keys.append(parts[0])

    assert len(new_keys) == len(base_values), 'count new %s, len base %s, diff n/o %s, diff o/n %s' % (len(new_keys), len(base_values), set(new_keys) - set(base_values.keys()), set(base_values.keys()) - set(new_keys))
    print('all good')


# python3 check_relabel.py /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_50_1.csv /home/caleml/partial_experiments/exp_20190911_1557_baseline/relabeling/relabeling_0_50p.csv
if __name__ == '__main__':
    check_relabel(sys.argv[1], sys.argv[2])
