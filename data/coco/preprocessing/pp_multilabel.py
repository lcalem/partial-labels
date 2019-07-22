'''
Take the raw MSCOCO annotation files (like instances_train2014.json) and creates a csv where each lines is an example:
12746,0,1,0,0,0,...1,0
(12746 is the image id and the rest is the ground truth of each class)
'''
import json
import os
import sys

import numpy as np

from collections import defaultdict
from pprint import pprint


def categories(json_data):
    cats = dict()
    for i, category in enumerate(sorted(json_data['categories'], key=lambda x: x['id'])):
        cats[category['id']] = {'name': category['name'], 'norm_id': i}

    assert len(cats) == 80
    return cats


def write_categories(root_dir, cats):
    '''
    normalized_id,name,original_id
    '''
    categories_file = os.path.join(root_dir, 'annotations', 'categories.csv')

    with open(categories_file, 'w+') as f_out:
        for key in sorted(cats, key=lambda x: cats[x]['norm_id']):
            line = '%s,%s,%s\n' % (cats[key]['norm_id'], cats[key]['name'], key)
            f_out.write(line)


def write_dataset(root_dir, original_name, id_to_label):
    dataset_file = os.path.join(root_dir, 'annotations', 'multilabel_%s.csv' % original_name)
    with open(dataset_file, 'w+') as f_out:

        for img_id in sorted(id_to_label):
            ground_truth = [str(int(gt)) for gt in id_to_label[img_id]]
            line = str(img_id) + ',' + ','.join(ground_truth) + '\n'
            f_out.write(line)


def create_csv(root_dir, original_name):

    annotations_file = os.path.join(root_dir, 'annotations', 'instances_%s.json' % original_name)
    with open(annotations_file, 'r') as f_in:
        json_data = json.load(f_in)

    # create categories file
    cats = categories(json_data)
    nb_classes = len(cats)

    # init image_ids
    id_to_label = dict()
    for img in json_data['images']:
        img_id = img['id']
        id_to_label[img_id] = np.zeros(nb_classes)

    # fill with data
    stats = defaultdict(int)
    for annot in json_data['annotations']:
        cat = annot['category_id']
        norm_cat = cats[cat]['norm_id']
        stats[cat] += 1

        img_id = annot['image_id']

        id_to_label[img_id][norm_cat] = 1
    pprint(stats)

    # write categories file
    write_categories(root_dir, cats)

    # write data csv file
    write_dataset(root_dir, original_name, id_to_label)


# python3 pp_multilabel.py /share/DEEPLEARNING/datasets/mscoco train2014
if __name__ == '__main__':
    create_csv(sys.argv[1], sys.argv[2])
