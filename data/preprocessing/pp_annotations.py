import csv
import os
import sys

import xml.etree.ElementTree as ET

from utils import load_ids

from pprint import pprint


def parse_xml_annot(annot_path):
    '''
    TODO: complete that once it's really useful
    '''

    annotation = dict()
    root = ET.parse(annot_path).getroot()

    for node in root:
        if node.tag == 'filename':
            annotation['img_name'] = node.text.split('.jpg')[0]

        print(node.tag, node.text)

    # size = root.findall('annotation')
    # print("size node %s" % size)
    # annotation['shape'] = [int(size.find('width').text), int(size.find('height').text), int(size.find('depth').text)]
    # pprint(annotation)

    return annotation


def preprocess_annotations(annot_folder):
    '''
    Preprocessing of the raw .xml annotation data in <root_path>/VOCdevkit/VOC2007/Annotations

    Creates a single csv file containing all ground truth data
    csv columns:
    - img_name (000012)
    - class_id (see class_id.csv for matching)
    - shape ([width, height, depth])
    - bounding_box ([xmin, ymin, xmax, ymax])
    - difficult (0 or 1)
    - truncated (0 or 1)
    - segmented (0 or 1)

    TODO: segmentation mask not loaded for now
    '''

    output_path = os.path.join(annot_folder, 'annotations.csv')
    class_ids = load_ids('class_id.csv')

    count_xml = 0
    count_ignored = 0

    with open(output_path, 'w+') as f_out:

        for filename in os.listdir(annot_folder):

            # ignore non-xml file
            if not filename.endswith('.xml'):
                print('Ignoring file %s' % filename)
                count_ignored += 1
                continue

            count_xml += 1
            parsed = parse_xml_annot(os.path.join(annot_folder, filename))

            # yes we want it to fail if the key is not here
            row = ';'.join([parsed[key] for key in csv_keys]) + '\n'
            f_out.write(row)

            break


# python3 pp_annotations.py /share/DEEPLEARNING/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations
if __name__ == '__main__':
    preprocess_annotations(sys.argv[1])
