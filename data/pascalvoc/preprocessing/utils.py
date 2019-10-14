import csv
import os


def load_ids():
    '''
    Example:

    'car': {
        'name': 'car',
        'id': 06,
        'train_img': 376,
        'train_obj': 625,
        'val_img': 337,
        'val_obj': 625,
        'superclass': 'vehicle',
        'short': 'cr'
    }

    '''
    # most of the fields in class_id.csv are integers
    def int_or_not(key, val): return (val if key in ['name', 'superclass', 'short'] else int(val))

    here = os.path.abspath(os.path.dirname(__file__))

    name2info = dict()
    with open(os.path.join(here, 'class_id.csv'), 'r') as f_in:
        reader = csv.DictReader(f_in, delimiter=';')
        for row in reader:
            name2info[row['name']] = {k: int_or_not(k, v) for k, v in row.items()}

    return name2info
