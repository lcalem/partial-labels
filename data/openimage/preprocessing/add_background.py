import os


def add_background_class():
    '''
    1. add background in challenge-2018-classes.csv
    2. add +1 to every class id in challenge-2018-train.csv and challenge-2018-val.csv
    '''

    dataset_path = '/local/DEEPLEARNING/oid'

    # 1. add new class
    classes_filepath = os.path.join(dataset_path, 'annotations', 'challenge-2018-classes.csv')
    old_classes = classes_filepath.replace('.csv', '_old2.csv')
    os.rename(classes_filepath, old_classes)

    with open(old_classes, 'r') as f_in, open(classes_filepath, 'w+') as f_out:
        f_out.write('0,background,_\n')

        for line in f_in:
            parts = line.strip().split(',')
            parts[0] = str(int(parts[0]) + 1)
            newline = ','.join(parts) + '\n'
            f_out.write(newline)

    # 2. update datasets
    for dataset in ['train', 'val']:
        dataset_filepath = os.path.join(dataset_path, 'annotations', 'challenge-2018-%s.csv' % dataset)
        old_path = dataset_filepath.replace('.csv', '_old2.csv')
        os.rename(dataset_filepath, old_path)

        with open(old_path, 'r') as f_in, open(dataset_filepath, 'w+') as f_out:
            for line in f_in:
                parts = line.strip().split(',')
                parts[2] = str(int(parts[2]) + 1)
                newline = ','.join(parts) + '\n'
                f_out.write(newline)


if __name__ == '__main__':
    add_background_class()
