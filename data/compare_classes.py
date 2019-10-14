import os


def check_oid_coco():
    '''
    check if coco classes are in oid classes
    '''

    here = os.path.abspath(os.path.dirname(__file__))
    coco_path = os.path.join(here, 'coco', 'class_ids.csv')
    oid_path = os.path.join(here, 'openimage', 'class_ids.csv')

    coco_names = list()
    oid_names = list()

    with open(coco_path, 'r') as f_coco:
        for line in f_coco:
            parts = line.split(',')
            coco_names.append(parts[1].lower().strip())

    with open(oid_path, 'r') as f_oid:
        for line in f_oid:
            parts = line.split(',')
            oid_names.append(parts[1].lower().strip())

    print('found %s coco names' % len(coco_names))
    print('found %s OID names' % len(oid_names))

    missing_names = list()

    for name in coco_names:
        if name not in oid_names:
            missing_names.append(name)

    print('out of %s categories, %s were found in OID' % (len(coco_names), len(coco_names) - len(missing_names)))
    print('missing names:')
    for name in missing_names:
        print('missing %s' % name)


if __name__ == '__main__':
    check_oid_coco()


'''
found 80 coco names
found 602 OID names
out of 80 categories, 66 were found exactly in OID

missing cow                 -> 'cattle'
missing frisbee             -> 'flying disc'
missing skis                -> 'ski'
missing sports ball         -> 'ball', 'football', 'cricket ball', 'golf ball', 'rugby ball', 'tennis ball', 'volleyball (ball)'
missing cup                 -> 'coffee cup'
missing donut               -> 'doughnut'
missing potted plant        -> 'houseplant'
missing dining table        -> 'coffee table', 'kitchen & dining room table', 'table'
missing tv                  -> 'television'
missing remote              -> 'remote control'
missing keyboard            -> 'computer keyboard'
missing cell phone          -> 'mobile phone'
missing microwave           -> 'microwave oven'
missing hair drier          -> 'hair dryer'

'''