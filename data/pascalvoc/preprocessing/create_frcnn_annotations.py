import os
import sys

from utils import load_ids_frcnn

import xml.etree.ElementTree as et

from collections import defaultdict
from pprint import pprint


def extract_name_and_bbox(tree_elem):
    obj = defaultdict(list)

    for el in tree_elem:
        if el.tag == 'name':
            obj['name'] = el.text
        if el.tag == 'bndbox':
            bbox = [None, None, None, None]
            for subel in el.iter():
                if subel.tag == 'xmin':
                    bbox[0] = subel.text
                elif subel.tag == 'ymin':
                    bbox[1] = subel.text
                elif subel.tag == 'xmax':
                    bbox[2] = subel.text
                elif subel.tag == 'ymax':
                    bbox[3] = subel.text

            obj['bbox'] = bbox

    return obj


def create_annotations(data_path):
    '''
    output format
    img_id,cls_id,tag_id,is_part,parent_id,xmin,ymin,xmax,ymax,width,height,depth
        - tag_id is just img_id + int representing the counter of tags in the image
        - is_part indicates whether the tag is a 'part' of another tag
        - parent_id is to indicate the parent of the part (only for parts)
    '''

    subsets = ('train', 'val', 'trainval', 'test')

    class_info = load_ids_frcnn()

    for subset in subsets:
        ids_path = os.path.join(data_path, 'ImageSets/Main', '%s.txt' % subset)
        output_path = os.path.join(data_path, 'Annotations', 'frcnn_%s.csv' % subset)

        # get image ids for the subset
        image_ids = list()
        with open(ids_path, 'r') as f_ids:
            for line in f_ids:
                image_ids.append(line.strip())

        count_images = 0

        with open(output_path, 'w+') as f_out:

            for image_id in image_ids:
                # print('doing %s' % image_id)
                count_images += 1

                # recover bbox data for each image
                xml_image_data = os.path.join(data_path, 'Annotations', '%s.xml' % image_id)

                size_data = [None, None, None]
                object_data = list()

                with open(xml_image_data, 'r') as f_xml:

                    xmltext = f_xml.read()
                    tree = et.fromstring(xmltext)

                    # size data
                    for size_elt in tree.iterfind('size'):
                        for el in size_elt:
                            if el.tag == 'width':
                                size_data[0] = el.text
                            elif el.tag == 'height':
                                size_data[1] = el.text
                            elif el.tag == 'depth':
                                size_data[2] = el.text

                    # object data
                    for obj_elt in tree.iterfind('object'):
                        obj = extract_name_and_bbox(obj_elt)

                        for parts_elt in obj_elt.iterfind('part'):
                            part_obj = extract_name_and_bbox(parts_elt)
                            obj['parts'].append(part_obj)

                        object_data.append(obj)

                # write annotation line for each bbox
                for i, obj in enumerate(object_data):
                    # img_id,cls_id,tag_id,is_part,parent_id,xmin,ymin,xmax,ymax,width,height,depth
                    class_id = class_info[obj['name']]['id']
                    tag_id = '%s_%s' % (image_id, str(i))
                    xmin, ymin, xmax, ymax = obj['bbox']   # xmin, ymin, xmax, ymax
                    width, height, depth = size_data

                    annot_line = [image_id, str(class_id), tag_id, '0', '0', xmin, ymin, xmax, ymax, width, height, depth]
                    f_out.write(','.join(annot_line) + '\n')

                    # write parts (we put the class name instead of the class_id because they don't have any ids)
                    for j, part in enumerate(obj['parts']):
                        pxmin, pymin, pxmax, pymax = part['bbox']
                        part_id = '%s_%s_%s' % (image_id, str(i), str(j))
                        part_line = [image_id, part['name'], part_id, '1', tag_id, pxmin, pymin, pxmax, pymax, width, height, depth]
                        f_out.write(','.join(part_line) + '\n')

        print('found %s examples, for subset %s', (count_images, subset))


# python3 create_frcnn_annotations.py /local/DEEPLEARNING/pascalvoc/VOCdevkit/VOC2007
if __name__ == '__main__':
    data_path = sys.argv[1]
    create_annotations(data_path)
