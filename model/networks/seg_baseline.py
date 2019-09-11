import tensorflow as tf

import keras_applications
keras_applications.set_keras_submodules(
    backend=tf.keras.backend,
    layers=tf.keras.layers,
    models=tf.keras.models,
    utils=tf.keras.utils
)

from tensorflow.keras import Model, Input
from tensorflow.keras.applications import ResNet50, Xception

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from model.losses import get_loss
from model.networks import BaseModel

from config.config import cfg


class DiceScore(tf.keras.metrics.Metric):

    def __init__(self, class_id, name=None, dtype=None):
        super(DiceScore, self).__init__(name=name, dtype=dtype)
        self.class_id = class_id
        self.intersection_sum = self.add_weight(name='intersection_sum', initializer='zeros')
        self.true_sum = self.add_weight(name='true_sum', initializer='zeros')
        self.pred_sum = self.add_weight(name='pred_sum', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.equal(tf.argmax(y_true, axis=-1), self.class_id)
        y_pred = tf.math.equal(tf.argmax(y_pred, axis=-1), self.class_id)

        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        self.intersection_sum.assign_add(intersection)
        self.true_sum.assign_add(tf.reduce_sum(y_true))
        self.pred_sum.assign_add(tf.reduce_sum(y_pred))

    def result(self):
        epsilon = tf.keras.backend.epsilon()
        return (2. * self.intersection_sum + epsilon) / (self.true_sum + self.pred_sum + epsilon)



class WeightedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, class_weights):
        super().__init__()
        self.name = 'weighted_cross_entropy'
        self.class_weights = tf.constant(class_weights, dtype=tf.float32, name='class_weights')
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        unweighted_cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        cross_entropy = tf.reduce_mean(unweighted_cross_entropy * weights)
        return cross_entropy


class WeightedCrossEntropy_v2(tf.keras.losses.Loss):
    def __init__(self, class_weights):
        super().__init__()
        self.name = 'weighted_cross_entropy'
        self.class_weights = tf.constant(class_weights, shape=(4,), dtype=tf.float32, name='class_weights')
    
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, 4))
        y_pred = tf.reshape(y_pred, shape=(-1, 4))
    
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        unweighted_cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        cross_entropy = tf.reduce_mean(unweighted_cross_entropy * weights)
        return cross_entropy


class SegBaseline(BaseModel):

    def __init__(self, exp_folder, n_classes, p=1):
        self.exp_folder = exp_folder

        self.p = p
        self.n_classes = n_classes
        self.input_shape = (cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, cfg.IMAGE.N_CHANNELS)
        self.verbose = cfg.VERBOSE

        print("Init input_shape %s" % str(self.input_shape))

    def build(self):

        resnet = ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)
        
        x = resnet.output
        x = tf.keras.layers.Conv2D(self.n_classes, 1, activation=None)(x)
        x = tf.keras.layers.UpSampling2D(size=(32,32), interpolation='bilinear')(x)
        #proba = tf.keras.layers.Activation('relu')(x)
        #x = tf.keras.layers.Conv2D(self.n_classes, 3, activation=None, padding='same')(x)
        #x = tf.keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, [self.input_shape[0], self.input_shape[1]]))(x)
        
        proba = tf.keras.layers.Activation('softmax')(x)

        self.model = Model(inputs=resnet.inputs, outputs=proba, name='cls_model')

        self.log('Outputs shape %s' % str(self.model.output_shape))

        optimizer = self.build_optimizer()
        
        metrics = ['categorical_accuracy',
                   DiceScore(class_id=1, name='liver_dice'),
                   DiceScore(class_id=2, name='pancreas_dice'),
                   DiceScore(class_id=3, name='stomach_dice')]
        
        #loss = WeightedCrossEntropy([1.0, 3.0, 50.0, 30.0])
        #loss = WeightedCrossEntropy_v2([1.011276563, 144.6087607, 1090.794364, 301.3094671])
        loss = WeightedCrossEntropy_v2([0.04292439486, 0.6141659478, 5.935304885, 2.689794809])
        #loss = WeightedCrossEntropy([0.004869942426, 2.160194604, 3.037742885, 2.479012777])
        #loss = tf.keras.losses.CategoricalCrossentropy()
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        if self.verbose:
            self.log('Final model summary')
            self.model.summary()
