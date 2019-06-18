import argparse
import datetime
import os
import shutil
import yaml

from pprint import pprint

from config import config_utils
from data.pascalvoc import PascalVOC
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
        filepath = '../configs/%s' % filepath
        if not os.path.isfile(filepath):
            raise Exception('config file %s not found' % filepath)

    with open(filepath, 'r') as f_in:
        config = yaml.safe_load(f_in)

    print('\n========================')
    print('Loaded config\n')
    pprint(config)
    print('========================\n')

    return config


def exp_init(params, exps_folder=None):
    '''
    common actions for setuping an experiment:
    - create experiment folder
    - dump config in it
    - dump current model code in it (because for now we only save weights)
    '''
    if exps_folder is None:
        exps_folder = os.path.join(os.environ['HOME'], 'partial_experiments')

    # model folder
    model_folder = '%s/exp_%s_%s_%s' % (exps_folder, datetime.datetime.now().strftime("%Y%m%d_%H%M"), params['architecture'], params.get('name', ''))
    os.makedirs(model_folder)
    print("Conducting experiment for %s epochs in folder %s" % (params['n_epochs'], model_folder))

    # config
    config_path = os.path.join(model_folder, 'config.yaml')
    with open(config_path, 'w+') as f_conf:
        yaml.dump(params, f_conf, default_flow_style=False)

    # model
    src_folder = os.path.dirname(os.path.realpath(__file__)) + '/..'
    dst_folder = os.path.join(model_folder, 'model_src/')
    shutil.copytree(src_folder, dst_folder)

    return model_folder


class Launcher():

    def __init__(self, options):
        config_utils.update_config(options)

    def launch(self):

        dataset = self.load_dataset()
        data_train = BatchLoader(
            dataset,
            ['multilabel'],
            TRAIN_MODE,
            batch_size=cfg.BATCH_SIZE,
            shuffle=cfg.DATASET.SHUFFLE)

    def load_dataset(self):
        '''
        we keep an ugly switch for now
        '''
        if cfg.DATASET.NAME == 'pascalvoc':
            dataset = PascalVOC(cfg.DATASET.PATH)
        else:
            raise Exception('Unknown dataset %s' % cfg.DATASET.NAME)

        return dataset


# python3 launch.py -o baseline -g 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', '-o', required=True, help='options yaml file')
    parser.add_argument('--gpu', '-g', required=True, help='# of the gpu device')

    args = parser.parse_args()
    options = parse_options_file(args.options)

    exp_folder = exp_init(options)
    options['exp_folder'] = exp_folder

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    launcher = Launcher(options)
    launcher.launch()


