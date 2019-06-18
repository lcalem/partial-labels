'''
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit
'''

from data import Dataset


class PascalVOC(Dataset):

    def __init__(self, dataset_path, dataset_type):
        '''
        Only multilabel for now
        '''
        assert dataset_type in ['train', 'val', 'trainval', 'test']

        self.dataset_path = dataset_path
        self.load_annotations(os.path.join(dataset_path, 'VOCdevkit/VOC2007/Annotations/annotations_multilabel_%s.csv' % dataset_type))

    def load_annotations(self, annotations_path):
        pass
