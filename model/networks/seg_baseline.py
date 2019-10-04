import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.applications import ResNet50, Xception

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

import numpy as np

from model.losses import get_loss
from model.networks import BaseModel

from config.config import cfg


def prior_pre_processing(prior_map, gamma):
    """
    Prior Map post processing
    given prior map should be of size [H, W, K]
    """
    non_zero_indices = np.where(prior_map != 0)
    # eps = np.min(prior_map[non_zero_indices]) / 2
    eps = 1e-7
    prior_map[non_zero_indices] = prior_map[non_zero_indices] ** gamma
    prior_map = prior_map / np.expand_dims(np.sum(prior_map, axis=-1), axis=-1)
    prior_map[np.where(prior_map == 0)] = eps

    return prior_map



class DiceScore(tf.keras.metrics.Metric):

    def __init__(self, class_id, name=None, dtype=None):
        super(DiceScore, self).__init__(name=name, dtype=dtype)
        self.class_id = class_id
        self.intersection_sum = self.add_weight(name='intersection_sum', initializer='zeros')
        self.true_sum = self.add_weight(name='true_sum', initializer='zeros')
        self.pred_sum = self.add_weight(name='pred_sum', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        #background_prob, liver_prob, pancreas_prob, stomach_prob = tf.unstack(y_pred, num=4, axis=3)
        # Assemble prob threshold with organs probabilities
        #organs_prob = tf.stack([tf.ones_like(background_prob, dtype=tf.float32)*0.5, liver_prob, pancreas_prob, stomach_prob], axis=3, name='organs_prob')
        # Keep the maximum probabilities as the predicted value

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

    def __init__(self, ambiguity_map, class_weights):
        super().__init__()
        self.name = 'weighted_cross_entropy'
        self.class_weights = tf.constant(class_weights, shape=(len(class_weights),), dtype=tf.float32, name='class_weights')
        self.ambiguity_map = ambiguity_map

    def call(self, y_true, y_pred):
        
        weights = tf.reduce_sum(self.ambiguity_map * self.class_weights * y_true, axis=-1)
        
        unweighted_cross_entropy = tf.keras.losses.binary_crossentropy(y_true * self.ambiguity_map, y_pred * self.ambiguity_map)
        #unweighted_cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        #cross_entropy = tf.reduce_mean(unweighted_cross_entropy * weights)
        return tf.reduce_mean(unweighted_cross_entropy)


class SegBaseline(BaseModel):

    def __init__(self, exp_folder, n_classes, p=1):
        self.exp_folder = exp_folder

        self.p = p
        self.n_classes = n_classes
        self.input_shape = (cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, cfg.IMAGE.N_CHANNELS)
        self.output_shape = (cfg.IMAGE.IMG_SIZE, cfg.IMAGE.IMG_SIZE, self.n_classes)
        self.verbose = cfg.VERBOSE

        self.prior = np.load('/local/DEEPLEARNING/IRCAD_liver_pancreas_stomach/priors/pancreas_{}.npy'.format(self.p))
        self.prior = prior_pre_processing(self.prior, 2.0)

        self.decision_threshold = 0.5

        print("Init input_shape %s" % str(self.input_shape))

        
    def build(self):

        ambiguity_map = tf.keras.Input(shape=self.output_shape, name='ambiguity_map')

        resnet = ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)

        x = resnet.output
        x = tf.keras.layers.Conv2D(self.n_classes, 1, activation=None)(x)
        x = tf.keras.layers.UpSampling2D(size=(32,32), interpolation='bilinear')(x)

        proba = tf.keras.layers.Activation('sigmoid')(x)

        probas = tf.unstack(proba, axis=-1)
        proba = tf.stack([1.0 - probas[1], probas[1]], axis=-1)

        organ_prior_map = tf.constant(self.prior, dtype=tf.float32, shape=self.prior.shape, name='prior_map')

        #proba = (proba * organ_prior_map) / tf.expand_dims(tf.reduce_sum(proba * organ_prior_map, axis=-1), axis=-1)


        self.model = Model(inputs=[resnet.inputs, ambiguity_map], outputs=proba, name='cls_model')

        self.log('Outputs shape %s' % str(self.model.output_shape))

        optimizer = self.build_optimizer()
        
        metrics = ['binary_accuracy', DiceScore(class_id=1, name='pancreas_dice')]
        
        #loss = WeightedCrossEntropy([1.0, 3.0, 50.0, 30.0])
        #loss = WeightedCrossEntropy_v2([1.011276563, 144.6087607, 1090.794364, 301.3094671])
        #loss = WeightedCrossEntropy_v2(ambiguity_map, [0.04292439486, 0.6141659478, 5.935304885, 2.689794809])
        #loss = WeightedCrossEntropy_v2(ambiguity_map, [1.0, 1.0]) #[0.00171322633574941, 0.998286773664251])
        #loss = WeightedCrossEntropy([0.004869942426, 2.160194604, 3.037742885, 2.479012777])
        loss = WeightedCrossEntropy(ambiguity_map, [1.0, 1.0])
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        if self.verbose:
            self.log('Final model summary')
            self.model.summary()
