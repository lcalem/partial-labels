import os


def clean_base():
    annotations_path = '/share/DEEPLEARNING/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations'
    old_path = os.path.join(annotations_path, 'annotations_multilabel_val.csv')
    new_path = os.path.join(annotations_path, 'annotations_multilabel_val_new.csv')

    with open(old_path, 'r') as f_in, open(new_path, 'w+') as f_out:
        for line in f_in:
            parts = line.strip().split(',')
            new_line = parts[0] + ',' + ','.join(['-1' if elt == '0' else elt for elt in parts[1:]])

            f_out.write(new_line + '\n')


if __name__ == '__main__':
    clean_base()
