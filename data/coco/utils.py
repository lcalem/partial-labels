import csv
import os


def load_ids():
    '''
    Example:

    06: {
        'name': 'car',
        'id': 06,
        'superclass': 'vehicle',
        'oid': 07
    }

    '''
    here = os.path.abspath(os.path.dirname(__file__))

    id2info = dict()
    with open(os.path.join(here, 'class_ids.csv'), 'r') as f_in:
        reader = csv.DictReader(f_in, delimiter=',')
        for row in reader:
            id2info[int(row['id'])] = {k: v for k, v in row.items()}

    return id2info
