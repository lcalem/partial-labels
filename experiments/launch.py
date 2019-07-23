import argparse
import datetime
import os
import shutil
import yaml

from pprint import pprint

from tensorflow.keras.callbacks import TensorBoard

from config import config_utils

from data.pascalvoc.pascalvoc import PascalVOC
from data.coco.coco import CocoGenerator
from experiments.data_gen import PascalVOCDataGenerator

from model.callbacks.metric_callbacks import MAPCallback
from model.callbacks.save_callback import SaveModel
from model.networks.baseline import Baseline
from model.utils.config import cfg

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

ALL_PCT = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def parse_options_file(filepath):
    '''
    - load options
    - checks options

    TODO: the check part
    '''

    # add the .yaml part if needed
    if not filepath.endswith('.yaml'):
        filepath = filepath + '.yaml'

    # not an absolute path -> try to find it in the configs/ folder
    if not filepath.startswith('/'):
        filepath = '../config/%s' % filepath
        if not os.path.isfile(filepath):
            raise Exception('config file %s not found' % filepath)

    with open(filepath, 'r') as f_in:
        config = yaml.safe_load(f_in)

    print('\n========================')
    print('Loaded config\n')
    pprint(config)
    print('========================\n')

    return config


def exp_init(exps_folder=None, exp_name=None):
    '''
    common actions for setuping an experiment:
    - create experiment folder
    - dump config in it
    - dump current model code in it (because for now we only save weights)
    '''
    if exps_folder is None:
        exps_folder = os.path.join(os.environ['HOME'], 'partial_experiments')

    # model folder
    name_suffix = ('_%s' % exp_name) if exp_name else ''
    model_folder = '%s/exp_%s_%s%s' % (exps_folder, datetime.datetime.now().strftime("%Y%m%d_%H%M"), cfg.ARCHI.NAME, name_suffix)
    os.makedirs(model_folder)
    print("Conducting experiment for %s epochs in folder %s" % (cfg.TRAINING.N_EPOCHS, model_folder))

    # config
    config_path = os.path.join(model_folder, 'config.yaml')
    with open(config_path, 'w+') as f_conf:
        yaml.dump(cfg, f_conf, default_flow_style=False)

    # model
    src_folder = os.path.dirname(os.path.realpath(__file__)) + '/..'
    dst_folder = os.path.join(model_folder, 'model_src/')
    shutil.copytree(src_folder, dst_folder)

    return model_folder


class Launcher():

    def __init__(self, exp_folder, percent=None):
        '''
        exp_percents: the known label percentages of the sequential experiments to launch (default: all of them)
        '''
        self.exp_folder = exp_folder   # still not sure this should go in config or not
        self.data_dir = cfg.DATASET.PATH

        if percent is None:
            self.exp_percents = ALL_PCT
        elif isinstance(percent, int):
            assert percent in ALL_PCT
            self.exp_percents = [percent]
        elif isinstance(percent, str):
            parts = [int(p) for p in percent.split(',')]
            assert all([p in ALL_PCT for p in parts])
            self.exp_percents = parts

        print('Launching with percentages %s' % str(self.exp_percents))

    def launch(self):
        '''
        launch one experiment per known label proportion
        '''
        for p in self.exp_percents:
            print('\n=====================\nLaunching experiment for percentage %s \n' % p)
            self.launch_percentage(p)
            print('\n=====================')

    def launch_percentage(self, p):
        '''
        For a given known label percentage p:

        1. load dataset
        3. callbacks
        4. load / build model
        5. train
        '''

        self.dataset_train = self.load_dataset(mode=cfg.DATASET.TRAIN, y_keys=['multilabel'], percentage=p)
        self.dataset_test = self.load_dataset(mode=cfg.DATASET.TEST, y_keys=['multilabel'])

        # callbacks
        cb_list = self.build_callbacks(p)

        # model
        self.build_model(self.dataset_train.nb_classes, p)
        steps_per_epoch = int(len(self.dataset_train.id_to_label) / cfg.BATCH_SIZE) + 1
        self.model.train(self.dataset_train.flow(batch_size=cfg.BATCH_SIZE), steps_per_epoch=steps_per_epoch, cb_list=cb_list)

    def load_dataset(self, mode, y_keys, percentage=None):
        '''
        we keep an ugly switch for now
        TODO: better dataset mode management
        '''
        if cfg.DATASET.NAME == 'pascalvoc':
            dataset = PascalVOCDataGenerator(mode, self.data_dir, prop=percentage)

            # dataset = PascalVOC(cfg.DATASET.PATH, batch_size, mode, x_keys=['image'], y_keys=y_keys, p=percentage)
        elif cfg.DATASET.NAME == 'coco':
            dataset = CocoGenerator(mode, self.data_dir, prop=percentage)
        else:
            raise Exception('Unknown dataset %s' % cfg.DATASET.NAME)

        return dataset

    def build_model(self, n_classes, p):
        '''
        TODO: we keep an ugly switch for now, do a more elegant importlib base loader after
        TODO: that percentage omg
        '''
        print("building model")
        if cfg.ARCHI.NAME == 'baseline':
            self.model = Baseline(self.exp_folder, n_classes)

        self.model.build(p / 100)

    def build_callbacks(self, prop):
        '''
        prop = proportion of known labels of current run

        TensorBoard
        SaveModel
        MAPCallback
        '''
        print("building callbacks")
        cb_list = list()

        # tensorboard
        logs_folder = os.environ['HOME'] + '/partial_experiments/tensorboard/' + self.exp_folder.split('/')[-1] + '/prop%s' % prop
        print('Tensorboard log folder %s' % logs_folder)
        tensorboard = TensorBoard(log_dir=os.path.join(logs_folder, 'tensorboard'))
        cb_list.append(tensorboard)

        # MAP
        batch_size = len(self.dataset_test)
        generator_test = self.dataset_test.flow(batch_size=batch_size)
        print("test data length %s" % len(self.dataset_test))
        X_test, Y_test = next(generator_test)

        # x_val, y_val = self.dataset_test[0]
        map_cb = MAPCallback(X_test, Y_test, self.exp_folder, prop)
        cb_list.append(map_cb)

        # Save Model
        cb_list.append(SaveModel(self.exp_folder, prop))

        return cb_list


# python3 launch.py -o pv_baseline50_sgd -g 1 -p 100
# python3 launch.py -o pv_baseline50_sgd -g 2 -p 10,30,50,70,90
# python3 launch.py -o pv_partial50_sgd -g 2 -p 10
# python3 launch.py -o coco_baseline50_sgd -g 1 -p 100
# python3 launch.py -o coco_baseline50_sgd -g 2 -p 90,70,50,30,10
# python3 launch.py -o coco_partial50_sgd -g 3 -p 90,70,50,30,10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', '-o', required=True, help='options yaml file')
    parser.add_argument('--gpu', '-g', required=True, help='# of the gpu device')
    parser.add_argument('--percent', '-p', help='the specific percentage of known labels. When not specified all percentages are sequentially launched')
    parser.add_argument('--exp_name', '-n', help='optional experiment name')

    args = parser.parse_args()
    options = parse_options_file(args.options)

    config_utils.update_config(options)

    exp_folder = exp_init(args.exp_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    launcher = Launcher(exp_folder, percent=args.percent)
    launcher.launch()


