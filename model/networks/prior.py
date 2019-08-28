import os

# because tensorflow.keras.applications doesn't have ResNet101 for some reason we have this workaround
# ONLY works with keras_applications=1.0.7 since 1.0.6 doesn't have ResNet101 an 1.0.8 removed the set_keras_submodules function
import tensorflow

import keras_applications
keras_applications.set_keras_submodules(
    backend=tensorflow.keras.backend,
    layers=tensorflow.keras.layers,
    models=tensorflow.keras.models,
    utils=tensorflow.keras.utils
)

from tensorflow.keras import Model, Input
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation

from model.losses import get_loss
from model.metrics import MAP
from model.networks import BaseModel

from config.config import cfg

import numpy as np


class PriorModel(BaseModel):

    def __init__(self, exp_folder, n_classes, p=1):
        '''
        p is the proportion of known labels
        '''
        if p > 1.0:
            p = p / 100

        self.exp_folder = exp_folder

        self.p = p
        self.n_classes = n_classes
        self.input_shape = (cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, cfg.IMAGE.N_CHANNELS)
        self.verbose = cfg.VERBOSE

        print("Init input_shape %s" % str(self.input_shape))

    def build(self):
        '''
        '''

        self.prior = np.zeros((self.n_classes, self.n_classes), dtype='int32')
        self.build_classifier()

        inp = Input(shape=self.input_shape, name='image_input')
        self.cooc_matrix = self.load_matrix()

        # classifier
        x = self.cls_model(inp)

        # dense + sigmoid for multilabel classification
        x = GlobalAveragePooling2D()(x)
        sk = Dense(self.n_classes)(x)
        output = Activation('sigmoid')(sk)

        self.model = Model(inputs=inp, outputs=output)
        self.log('Outputs shape %s' % str(self.model.output_shape))

        optimizer = self.build_optimizer()
        loss = get_loss(cfg.ARCHI.LOSS, params={'prop': self.p})
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy'])

        if self.verbose:
            self.log('Final model summary')
            self.model.summary()

    def load_matrix(self):

        matrix_filename = 'cooc_matrix_trainval_partial_100_1.npy'
        matrix_filepath = os.path.join(cfg.DATASET.PATH, 'Annotations', matrix_filename)
        return np.load(matrix_filepath)

    def build_classifier(self):
        if cfg.ARCHI.CLASSIFIER == 'resnet101':
            resnet = keras_applications.resnet.ResNet101(include_top=False, weights='imagenet', input_shape=self.input_shape)
        elif cfg.ARCHI.CLASSIFIER == 'resnet50':
            resnet = ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)

        self.cls_model = Model(inputs=resnet.inputs, outputs=resnet.output, name='cls_model')

    def update_prior(self, y_true_data):
        '''
        callback prior update
        '''
        print(type(y_true_data))
        print(y_true_data)
