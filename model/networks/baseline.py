
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import ResNet101, ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, Adam

from model.networks import BaseModel

from model.utils.config import cfg


class Baseline(BaseModel):

    def build(self):
        self.build_classifier()

        inp = Input(shape=self.input_shape, name='image_input')

        # classifier
        x = self.cls_model(inp)

        # dense + sigmoid for multilabel classification
        output = Dense(self.n_classes, activation='sigmoid')(x)

        self.model = Model(inputs=inp, outputs=output)
        self.log("Outputs shape %s" % self.model.output_shape)

        optimizer = self.build_optimizer()
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)

        if self.verbose:
            self.log("Final model summary")
            self.model.summary()

    def build_classifier(self):
        cls_model = ResNet101(include_top=False, weights='imagenet', input_shape=self.input_shape)
        self.cls_model = Model(inputs=cls_model.inputs, outputs=cls_model.output, name='cls_model')

    def build_optimizer(self):
        '''
        TODO: something better than an ugly switch <3
        '''
        if cfg.TRAINING.OPTIMIZER == 'rmsprop':
            return RMSprop(lr=cfg.TRAINING.START_LR)
        elif cfg.TRAINING.OPTIMIZER == 'adam':
            return Adam(lr=cfg.TRAINING.START_LR)
        raise Exception('Unknown optimizer %s' % cfg.TRAINING.OPTIMIZER)
