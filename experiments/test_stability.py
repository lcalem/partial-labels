import datetime
import os
import sys

from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import load_model

import numpy as np
from sklearn import metrics

from experiments.data_gen import PascalVOCDataGenerator

from model.callbacks.metric_callbacks import MAPCallback
from model.callbacks.save_callback import SaveModel

from model.losses import BCE


class Runner():
    
    def __init__(self, exp_folder):
        self.exp_folder = exp_folder
    
    def load_data(self, prop):
        data_dir = '/share/DEEPLEARNING/datasets/pascalvoc/VOCdevkit/VOC2007/'
        self.data_generator_train = PascalVOCDataGenerator('trainval', data_dir, prop=prop)

        self.data_generator_test = PascalVOCDataGenerator('test', data_dir)
        print('test length %s' % len(self.data_generator_test.images_ids_in_subset))
        
        # load eval dataset at once
        batch_size = len(self.data_generator_test.images_ids_in_subset)
        generator_test = self.data_generator_test.flow(batch_size=batch_size)
        self.X_test, self.Y_test = next(generator_test)
        print('X test %s, Y test %s' % (str(self.X_test.shape), str(self.Y_test.shape)))
        
    def build_model(self):
        input_shape = (224, 224, 3)
        resnet = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

        inp = Input(shape=input_shape, name='image_input')
        x = resnet(inp)
        x = Flatten()(x)
        output = Dense(self.data_generator_train.nb_classes, activation='sigmoid')(x)
        self.model = Model(inputs=inp, outputs=output)
  
    def get_callbacks(self, prop):
        # callbacks
        cb_list = list()
        cb_list.append(SaveModel(self.exp_folder, prop))

        map_cb = MAPCallback(self.X_test, self.Y_test, self.exp_folder)
        cb_list.append(map_cb)
        return cb_list

    def evaluate(self, model):
        y_pred_test = model.predict(self.X_test)
        AP_test = np.zeros(20)
        for c in range(20):
            #AP_train[c] = average_precision_score(Y_train[:, c], y_pred_train[:, c])
            AP_test[c] = metrics.average_precision_score(Y_test[:, c], y_pred_test[:, c])
              
        print("MAP TEST =", AP_test.mean()*100)
        print(AP_test)
        
    def train(self, prop):
        self.load_data(prop)
        self.build_model()
              
        lr = 0.1
        loss = BCE()
        self.model.compile(loss=loss, optimizer=SGD(lr=lr), metrics=['binary_accuracy'])
        cb_list = self.get_callbacks(prop)
              
        batch_size=32
        nb_epochs=20
        steps_per_epoch_train = int(len(self.data_generator_train.id_to_label) / batch_size) + 1
        self.model.fit_generator(self.data_generator_train.flow(batch_size=batch_size),
                            steps_per_epoch=steps_per_epoch_train,
                            epochs=nb_epochs,
                            callbacks=cb_list,
                            verbose=1)
        
# python3 test_stability.py 1 90
# python3 test_stability.py 1 all
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])
    prop = sys.argv[2]
              
    exp_folder = '%s/partial_experiments/exp_%s_%s' % (os.environ['HOME'], datetime.datetime.now().strftime("%Y%m%d_%H%M"), 'TEST_script')
    os.makedirs(exp_folder)
    
    runner = Runner(exp_folder)
    
    if prop == 'all':
        for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            runner.train(p)
        
    else:
        runner.train(int(prop))
            