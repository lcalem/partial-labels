import os
import sys

import imagesize


def create_train_val_csv(data_path):
    '''
    output format
    img_id,subdir,cls_id,xmin,ymin,xmax,ymax, width, height
    '''

    classes_path = os.path.join(data_path, 'annotations', 'challenge-2018-class-descriptions-500.csv')
    val_ids_path = os.path.join(data_path, 'annotations', 'challenge-2018-image-ids-valset-od.csv')
    annotations_path = os.path.join(data_path, 'annotations', 'challenge-2018-train-annotations-bbox.csv')

    output_train_path = os.path.join(data_path, 'annotations', 'challenge-2018-train.csv')
    output_val_path = os.path.join(data_path, 'annotations', 'challenge-2018-val.csv')

    classes = dict()
    with open(classes_path, 'r') as f_cls:
        for i, line in enumerate(f_cls):
            parts = line.strip().split(',')
            classes[parts[0]] = {'id': i, 'name': parts[1]}

    val_ids = list()
    with open(val_ids_path, 'r') as f_valids:
        for line in f_valids:
            val_ids.append(line.strip())

    # walk the image subdirs to get the proper subdir
    img_subdirs = dict()
    for img_folder in os.listdir(os.path.join(data_path, 'images', 'train')):
        print(img_folder)
        folder_path = os.path.join(data_path, 'images', 'train', img_folder)
        if os.path.isdir(folder_path) and img_folder.startswith('train_'):
            for img_name in os.listdir(folder_path):
                img_subdirs[img_name.split('.')[0]] = img_folder

    # sanity check
    print('subdir values')
    print(set(img_subdirs.values()))
    assert len(set(img_subdirs.values())) > 0

    # generate output on the fly from reading the raw annotation file
    count_skipped = 0
    count_val = 0
    count_train = 0
    count_total = 0

    first = True
    with open(annotations_path, 'r') as f_in, open(output_train_path, 'w+') as f_train, open(output_val_path, 'w+') as f_val:
        for line in f_in:
            if first:
                first = False
                continue

            count_total += 1
            if count_total % 10000 == 0:
                print('done %s' % count_total)

            parts = line.strip().split(',')

            img_id = parts[0]

            if img_id not in img_subdirs:
                count_skipped += 1
                continue

            img_subdir = img_subdirs[img_id]
            cls_id = classes[parts[2]]['id']
            xmin = parts[4]
            xmax = parts[5]
            ymin = parts[6]
            ymax = parts[7]

            image_path = os.path.join(data_path, 'images', 'train', img_subdir, img_id + '.jpg')
            width, height = imagesize.get(image_path)

            line = ','.join([img_id, img_subdir, str(cls_id), xmin, ymin, xmax, ymax, str(width), str(height)]) + '\n'

            if img_id in val_ids:
                f_val.write(line)
                count_val += 1
            else:
                f_train.write(line)
                count_train += 1

    print('found %s train examples, %s val examples, %s have been skipped, %s total', (count_train, count_val, count_skipped, count_total))


# python3 create_csv /local/DEEPLEARNING/oid/
if __name__ == '__main__':
    data_path = sys.argv[1]
    create_train_val_csv(data_path)
