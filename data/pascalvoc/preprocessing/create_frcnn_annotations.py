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


class Annotations(object):
    def __init__(self, data_path, extended=False):
        self.data_path = data_path
        self.extended = extended
        self.class_info = load_ids_frcnn(extended=extended)

    def extract_xml_data(self, image_id):

        # recover bbox data for each image
        xml_image_data = os.path.join(self.data_path, 'Annotations', '%s.xml' % image_id)

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

        return object_data, size_data

    def create_annot_line(self, obj, size_data, image_id):
        # img_id,cls_id,tag_id,is_part,parent_id,xmin,ymin,xmax,ymax,width,height,depth
        class_id = self.class_info[obj['name']]['id']
        xmin, ymin, xmax, ymax = obj['bbox']   # xmin, ymin, xmax, ymax
        width, height, depth = size_data

        annot_line = [image_id, str(class_id), xmin, ymin, xmax, ymax, width, height, depth]
        return annot_line

    def create_annotations(self):
        '''
        output format
        img_id,cls_id,xmin,ymin,xmax,ymax,width,height,depth
            - tag_id is just img_id + int representing the counter of tags in the image
            - is_part indicates whether the tag is a 'part' of another tag
            - parent_id is to indicate the parent of the part (only for parts)
        extended: includes person parts ()
        '''

        subsets = ('train', 'val', 'trainval', 'test')

        for subset in subsets:
            ids_path = os.path.join(data_path, 'ImageSets/Main', '%s.txt' % subset)
            output_path = os.path.join(data_path, 'Annotations', 'frcnn_%s%s.csv' % (subset, '_ext' if self.extended else ''))

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

                    object_data, size_data = self.extract_xml_data(image_id)

                    # write annotation line for each bbox
                    for i, obj in enumerate(object_data):
                        annot_line = self.create_annot_line(obj, size_data, image_id)
                        f_out.write(','.join(annot_line) + '\n')

                        if self.extended:
                            # write parts
                            for j, part in enumerate(obj['parts']):
                                part_line = self.create_annot_line(part, size_data, image_id)
                                f_out.write(','.join(part_line) + '\n')

            print('found %s examples, for subset %s' % (count_images, subset))


# python3 create_frcnn_annotations.py /local/DEEPLEARNING/pascalvoc/VOCdevkit/VOC2007 ext
if __name__ == '__main__':
    data_path = sys.argv[1]
    extended = False if (len(sys.argv) == 2) else True   # ugly
    anot = Annotations(data_path, extended)
    anot.create_annotations()
