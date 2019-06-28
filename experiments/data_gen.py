import os
import random
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

random.seed(2506)

IMG_HEIGHT = 224
IMG_WIDTH = 224

LABELS = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


class PascalVOCDataGenerator(object):
    """
    PascalVOCDataGenerator defines a generator on the PascalVOC 2007 dataset

    Here are the links to download the data :
    val and train  :  http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    test           :  http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    
    prop: proportion of known labels
    """

    def __init__(self, subset, data_path, prop=None):

        assert subset in ['train', 'val', 'trainval', 'test']
        self.subset = subset
        self.prop = prop

        self.data_path = data_path
        self.images_path = os.path.join(self.data_path, 'JPEGImages')
        self.labels_path = os.path.join(self.data_path, 'ImageSets', 'Main')

        # The id_to_label dict has the following structure
        # key : image's id (e.g. 00084)
        # value : image's label (e.g. [0, 0, 1, 0, ..., 1, 0])
        self.id_to_label = {}

        self.labels = LABELS
        self.nb_classes = len(self.labels) # 20 classes for PascalVOC
        
        # Get all the images' ids for the given subset
        self.images_ids_in_subset = self._get_images_ids_from_subset(self.subset)
        
        if self.subset.startswith('train'):
            self.load_csv_data()
        else:
            self.load_data()
        
    def load_data(self):
        '''
        aka: the old way (the working way)
        '''

        # Create the id_to_label dict with all the images' ids
        # but the values are arrays with nb_classes (20) zeros
        self._initialize_id_to_label_dict()

        # Fill the values in the id_to_label dict by putting 1 when
        # the label is in the image given by the key
        self._fill_id_to_label_dict_with_classes()
        
    def load_csv_data(self):
        '''
        the new way
        /!\ loads with -1 instead of 0 for unknown labels
        '''
        csv_path = os.path.join(self.data_path, 'Annotations', 'annotations_multilabel_%s_partial_%s_1.csv' % (self.subset, self.prop))
        print('loading dataset from %s' % csv_path)
        with open(csv_path, 'r') as f_csv:
            for line in f_csv:
                parts = line.split(',')
                image_id = parts[0]
                labels = [int(l) for l in parts[1:]]
                self.id_to_label[image_id] = labels
                
    def _initialize_id_to_label_dict(self):
        for image_id in self.images_ids_in_subset:
            self.id_to_label[image_id] = np.zeros(self.nb_classes)

    def _fill_id_to_label_dict_with_classes(self):
        """_fill_id_to_label_dict_with_classes
        For each class, the <class>_<subset>.txt file contain the presence information
        of this class in the image
        """
        for i in range(self.nb_classes):
            label = self.labels[i]
            # Open the <class>_<subset>.txt file
            with open(os.path.join(self.labels_path, "%s_%s.txt" % (label, self.subset)), 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    splited_line = line.split()
                    image_id = splited_line[0]
                    is_present = int(splited_line[1])
                    if is_present == 1:
                        self.id_to_label[image_id][i] = 1

    def _get_images_ids_from_subset(self, subset):
        """_get_images_ids_from_subset
        The images' ids are found in the <subset>.txt file in ImageSets/Main
        """
        with open(os.path.join(self.labels_path, subset + '.txt'), 'r') as f:
            images_ids = f.read().splitlines()
        return images_ids
    
    def get_labels(self, image_id):
        '''
        for tests
        negative labels:
        -1 for train 
        0 for test or val
        '''
        labels = self.id_to_label[image_id]
        # if self.subset.startswith('train'):
            # labels = [-1 if l == 0 else l for l in labels]
        if self.subset in ['val', 'test']:
            labels = [0 if l == -1 else l for l in labels]
            
        return labels

    def flow(self, batch_size=32):
        """flow
        This is a generator which load the images and preprocess them on the fly
        When calling next python build in function, it returns a batch with a given size
        with a X_batch of size (None, IMG_HEIGHT, IMG_WIDTH, 3)
        and a Y_batch of size (None, nb_classes)
        The first dimension is the batch_size if there is enough images left otherwise
        it will be less

        :param batch_size: the batch's size
        """
        nb_batches = int(len(self.images_ids_in_subset) / batch_size) + 1
        while True:
            # Before each epoch we shuffle the images' ids
            random.shuffle(self.images_ids_in_subset)
            for i in range(nb_batches):
                # We first get all the images' ids for the next batch
                current_bach = self.images_ids_in_subset[i*batch_size:(i+1)*batch_size]
                X_batch = []
                Y_batch = []
                for image_id in current_bach:
                    # Load the image and resize it. We get a PIL Image object
                    img = image.load_img(os.path.join(self.images_path, image_id + '.jpg'), grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    # Cast the Image object to a numpy array and put the channel has the last dimension
                    img_arr = image.img_to_array(img, data_format='channels_last')
                    X_batch.append(img_arr)
                    # Y_batch.append(self.id_to_label[image_id])
                    Y_batch.append(self.get_labels(image_id))
                # resize X_batch in (batch_size, IMG_HEIGHT, IMG_WIDTH, 3)
                X_batch = np.reshape(X_batch, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
                # resize Y_batch in (None, nb_classes)
                Y_batch = np.reshape(Y_batch, (-1, self.nb_classes))
                # The preprocess consists of substracting the ImageNet RGB means values
                # https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L72
                X_batch = preprocess_input(X_batch, data_format='channels_last')
                yield(X_batch, Y_batch)

