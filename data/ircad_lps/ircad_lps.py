import os
import numpy as np
import glob

from data import Dataset
from config.config import cfg

import tensorflow as tf

NB_CLASSES = 4 # Background, Liver, Pancreas, Stomach


class IrcadLPS(Dataset):

    supported_keys = ('image', 'segmentation', 'ambiguity')
    supported_modes = ('train', 'valid')
    nb_classes = NB_CLASSES

    def __init__(self,
                 dataset_path,
                 batch_size,
                 mode,
                 x_keys,
                 y_keys,
                 split_name,
                 valid_split_number=0,
                 p=None):
        
        # Setting the random generator seed
        np.random.seed(cfg.RANDOM_SEED)

        self.images_path = os.path.join(dataset_path, 'images')
        self.annotations_path = os.path.join(dataset_path, 'annotations')
        self.missing_annotations_path = os.path.join(dataset_path, 'missing_organs')
        
        self.valid_split_number = valid_split_number
        self.split_name = split_name
        
        Dataset.__init__(self, dataset_path, batch_size, mode, x_keys, y_keys, p)
    

    def load_samples(self):
        if self.mode == 'valid':
            annotations_file = os.path.join(self.dataset_path, 'splits', '{}.{}'.format(self.split_name, self.valid_split_number))
            samples = self._read_split_file(annotations_file)
        elif self.mode == 'train':
            all_splits_files = glob.glob(os.path.join(self.dataset_path, 'splits', '{}.*'.format(self.split_name)))
            all_splits_files = [x for x in all_splits_files if not x.endswith(str(self.valid_split_number))]
            samples = []
            for split_file in all_splits_files:
                samples += self._read_split_file(split_file)
        else:
            raise Exception('Unknown mode {}'.format(self.mode))

        samples = sorted(samples)
        np.random.shuffle(samples)
        return samples

    
    def _read_split_file(self, filename):
        with open(filename, 'r') as f:
            line = f.read()
        patient_ids = [int(x) for x in line.split(';') if x != '']
        
        samples = []
        for pid in patient_ids:
            if self.p is None:
                p = 100
            else:
                p = self.p
            samples += [str(pid)+'/'+x for x in os.listdir(os.path.join(self.annotations_path, str(p), str(pid)))]
        
        return samples


    def get_key_shape(self, key):
        if key == 'image':
            return self.img_size
        elif key == 'segmentation':
            return self.img_size
        elif key == 'ambiguity':
            return self.img_size
        else:
            raise Exception('Unknown key {}'.format(key))

    def get_data_dict(self, sample_idxs):
        '''
        Creation of the actual batch
        '''
        output = {}
        sample_ids = [self.sample_ids[i] for i in sample_idxs]

        img_batch = []
        ambiguity_batch = []
        target_batch = []

        for img_id in sample_ids:

            missing_organs = np.array([1, 1, 1])
            if self.mode == 'train':
                target_path = os.path.join(self.annotations_path, str(self.p), img_id)
                missing_organs = np.load(os.path.join(self.missing_annotations_path, str(self.p), img_id))
            elif self.mode == 'valid':
                target_path = os.path.join(self.annotations_path, '100', img_id)
            else:
                raise Exception('Wrong mode: {}'.format(self.mode))
            
            # Image
            img_path = os.path.join(self.images_path, img_id)
            img = np.load(img_path)
            img = np.moveaxis(img, 0, -1)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            img_batch.append(img)

            # Annotation
            target = np.load(target_path)
            target = tf.keras.utils.to_categorical(target, self.nb_classes)
            target_batch.append(target)

            # Ambiguity map
            ambiguity_map = np.zeros_like(target, dtype=np.float32)
            ambiguity_map[:,:,0] = 1.0
            for i, kept in enumerate(missing_organs):
                if kept == 1:
                    ambiguity_map[:,:,i+1] = 1.0

            ambiguity_batch.append(ambiguity_map)
            
            
        img_batch = np.reshape(img_batch, (-1, self.img_size[0], self.img_size[1], 3))
        ambiguity_batch = np.reshape(ambiguity_batch, (-1, self.img_size[0], self.img_size[1], self.nb_classes))
        target_batch = np.reshape(target_batch, (-1, self.img_size[0], self.img_size[1], self.nb_classes))

        output['image'] = img_batch
        output['ambiguity'] = ambiguity_batch
        output['segmentation'] = target_batch

        return output


    def init_cooc(self):
        pass
