import datetime
import os
import sys

from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

import numpy as np
from sklearn import metrics

from experiments.data_gen import PascalVOCDataGenerator

from model.callbacks.metric_callbacks import MAPCallback
from model.callbacks.save_callback import SaveModel


class Runner():
    
    def __init__(self, exp_folder):
        self.exp_folder = exp_folder
    
    def load_data(self):
        data_dir = '/share/DEEPLEARNING/datasets/pascalvoc/VOCdevkit/VOC2007/'
        self.data_generator_train = PascalVOCDataGenerator('trainval', data_dir)

        self.data_generator_test = PascalVOCDataGenerator('test', data_dir)
        print('test length %s' % len(self.data_generator_test.images_ids_in_subset))
        
        # load eval dataset at once
        batch_size = len(self.data_generator_test.images_ids_in_subset)
        generator_test = self.data_generator_test.flow(batch_size=batch_size)
        self.X_test, self.Y_test = next(generator_test)
        print('X test %s, Y test %s' % (str(self.X_test.shape), str(self.Y_test.shape)))
        
    def build_model(self):
        model = ResNet50(include_top=True, weights='imagenet')
        model.layers.pop()
        x = model.layers[-1].output
        x = Dense(self.data_generator_train.nb_classes, activation='sigmoid', name='predictions')(x)
        self.model = Model(inputs=model.input, outputs=x)
              
    def get_callbacks(self):
        # callbacks
        cb_list = list()
        cb_list.append(SaveModel(self.exp_folder, 100))

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
        
    def train(self):
        self.load_data()
        self.build_model()
              
        lr = 0.1
        self.model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr), metrics=['binary_accuracy'])
        cb_list = self.get_callbacks()
              
        batch_size=32
        nb_epochs=20
        steps_per_epoch_train = int(len(self.data_generator_train.id_to_label) / batch_size) + 1
        self.model.fit_generator(self.data_generator_train.flow(batch_size=batch_size),
                            steps_per_epoch=steps_per_epoch_train,
                            epochs=nb_epochs,
                            callbacks=cb_list,
                            verbose=1)
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])
              
    exp_folder = '%s/partial_experiments/exp_%s_%s' % (os.environ['HOME'], datetime.datetime.now().strftime("%Y%m%d_%H%M"), 'TEST_script')
    os.makedirs(exp_folder)
    
    runner = Runner(exp_folder)
    runner.train()
            
    for filename in os.listdir():
        if not filename.startswith('model_100'):
              continue
              
        model = load_model(os.path.join(exp_folder, filename))
        print('eval for %s' % filename)
        runner.evaluate()
    