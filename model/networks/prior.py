import os

# because tensorflow.keras.applications doesn't have ResNet101 for some reason we have this workaround
# ONLY works with keras_applications=1.0.7 since 1.0.6 doesn't have ResNet101 an 1.0.8 removed the set_keras_submodules function
import tensorflow as tf

import keras_applications
keras_applications.set_keras_submodules(
    backend=tf.keras.backend,
    layers=tf.keras.layers,
    models=tf.keras.models,
    utils=tf.keras.utils
)
import tensorflow.keras.backend as K

from tensorflow.keras import Model, Input
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Lambda, Add

from model.losses import get_loss
from model.metrics import MAP
from model.networks import BaseModel
from model import layers

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
        logits = Dense(self.n_classes)(x)

        # prior integration
        prior = Lambda(self.prior_layer, name='prior')(logits)
        output = Lambda(self.activation_layer2, name='custom_activations')((logits, prior))

        # output = Activation('sigmoid')(logits)

        self.model = Model(inputs=inp, outputs=output)
        self.log('Outputs shape %s' % str(self.model.output_shape))

        optimizer = self.build_optimizer()
        loss = get_loss(cfg.ARCHI.LOSS, params={'prop': self.p})
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy'])

        if self.verbose:
            self.log('Final model summary')
            self.model.summary()

    def load_matrix(self):

        # matrix_filename = 'cooc_matrix_trainval_partial_100_1.npy'
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

    def kullback_leibler_div2(self, logits):
        '''
        logits is only one example of shape (1, K)
        output is the prior for this logit of shape (1, K)
        '''
        print('inside KL2 logits size %s' % str(logits.shape))

        def kl(cooc_line):
            '''
            TODO test with other p / q ordering
            '''
            print('inside KL2 cooc line size %s' % str(cooc_line.shape))
            p = K.clip(logits, K.epsilon(), 1)
            q = K.clip(cooc_line, K.epsilon(), 1)
            print('dtype p %s, dtype q %s' % (K.dtype(p), K.dtype(q)))
            # return K.sum(p * K.log(p / q), axis=-1)
            return K.sum(p * K.log(layers.div_layer(p, q)), axis=-1)

        return tf.map_fn(kl, self.cooc_matrix, dtype=tf.float32)

    def prior_layer(self, logits):
        '''
        '''
        print('logits shape: %s and type %s' % (str(logits.shape), type(logits)))
        # broad_logits = tf.tile(logits, [3, 1])
        # out = tf.map_fn(self.kullback_leibler_div2, (self.cooc_matrix, broad_logits), dtype=(tf.float32, tf.float64))
        out = tf.map_fn(self.kullback_leibler_div2, logits, dtype=tf.float32)
        print('prior shape %s' % str(out))
        return out

    def activation_layer(self, logits, prior):
        '''
        '''
        pk_tilde = layers.log_layer(prior)
        combination = Add()([logits, pk_tilde])
        num = layers.exp_layer(combination)

        denom = layers.sum_tensor_layer(combination)
        return layers.div_layer(num, denom)

    def activation_layer2(self, inputs):
        '''
        '''
        logits = inputs[0]
        prior = inputs[1]

        pk_tilde = K.log(prior)
        combination = logits + pk_tilde
        # return tf.nn.softmax(combination)
        return tf.nn.sigmoid(combination)

        # num = K.exp(combination)

        # denom = layers.sum_tensor_layer(combination)
        # return num / denom
