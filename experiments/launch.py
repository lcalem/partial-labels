import argparse
import datetime
import os
import shutil
import yaml

from pprint import pprint

from config import config_utils

from data.pascalvoc import PascalVOC

from model.networks.baseline import Baseline
from model.utils.config import cfg


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

    def launch(self):
        '''
        1. load dataset (TODO same for test data)
        3. callbacks (TODO)
        4. load / build model
        5. train
        '''

        dataset_train = self.load_dataset(mode='train', y_keys=['multilabel'])

        # callbacks (no callbacks for now)

        # model
        self.build_model(dataset_train.n_classes)
        self.model.train(dataset_train, steps_per_epoch=len(dataset_train), cb_list=[])

    def load_dataset(self):
        '''
        we keep an ugly switch for now
        TODO: better dataset mode management
        '''
        if cfg.DATASET.NAME == 'pascalvoc':
            dataset = PascalVOC(cfg.DATASET.PATH, 'train')
        else:
            raise Exception('Unknown dataset %s' % cfg.DATASET.NAME)

        return dataset

    def build_model(self, n_classes):
        '''
        TODO: we keep an ugly switch for now, do a more elegant importlib base loader after
        '''
        if cfg.ARCHI.NAME == 'baseline':
            self.model = Baseline(n_classes)

        self.model.build()


# python3 launch.py -o baseline -g 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', '-o', required=True, help='options yaml file')
    parser.add_argument('--gpu', '-g', required=True, help='# of the gpu device')
    parser.add_argument('--exp_name', '-n', help='optional experiment name')

    args = parser.parse_args()
    options = parse_options_file(args.options)

    config_utils.update_config(options)

    exp_folder = exp_init(args.exp_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    launcher = Launcher()
    launcher.launch()


