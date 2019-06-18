'''
Preprocessing of the ground truth classification files in <root_dataset>/VOCdevkit/VOC2007/ImageSets/Main

-> Creates a unique .csv containing the multilabel classification info
'''
import os
import sys

from collections import defaultdict

from utils import load_ids


def preprocess(dataset_folder, dataset_type):

    output_file = os.path.join(dataset_folder, 'VOCdevkit/VOC2007/Annotations', 'annotations_multilabel_%s.csv' % dataset_type)
    classif_annot_folder = os.path.join(dataset_folder, 'VOCdevkit/VOC2007/ImageSets/Main')

    class_ids = load_ids()
    data = defaultdict(lambda: [None] * len(class_ids))

    # read
    for filename in os.listdir(classif_annot_folder):
        if not filename.endswith('_%s.txt' % dataset_type):
            continue

        class_name = filename.split('_')[0]
        class_id = class_ids[class_name]['id']
        print('doing class %s (%s)' % (class_name, class_id))

        with open(os.path.join(classif_annot_folder, filename), 'r') as f_in:
            for line in f_in:
                img_id, class_gt = line.strip().split()
                data[img_id][class_id] = int(class_gt)

    # write data in output csv
    with open(output_file, 'w+') as f_out:
        for img_id, img_gt in data.items():
            assert all([x is not None] for x in img_gt), 'missing class value for img_id %s (%s)' % (img_id, str(img_gt))
            write_line = img_id + ',' + ','.join([str(x) for x in img_gt]) + '\n'
            f_out.write(write_line)

    print('multilabel annotations compiled in %s' % output_file)


# python3 pp_multilabel.py /share/DEEPLEARNING/datasets/pascalvoc/ trainval
if __name__ == '__main__':
    dataset_folder = sys.argv[1]
    dataset_type = sys.argv[2]
    assert dataset_type in ['train', 'val', 'trainval', 'test'], 'unrecognized dataset type %s' % dataset_type

    preprocess(dataset_folder, dataset_type)
