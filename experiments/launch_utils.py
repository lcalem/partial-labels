import datetime
import os
import shutil
import yaml

from config.config import cfg

from pprint import pprint


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
    # TODO: this is a tad too ugly
    if not filepath.startswith('/'):
        possible_paths = ['../config/%s' % filepath, '../config/old/%s' % filepath, '%s/partial-labels/config/%s' % (os.environ['HOME'], filepath), '%s/partial-labels/config/old/%s' % (os.environ['HOME'], filepath)]
        for filepath in possible_paths:
            if os.path.isfile(filepath):
                break
        else:
            raise Exception('config file %s not found' % filepath)

    with open(filepath, 'r') as f_in:
        config = yaml.safe_load(f_in)

    print('\n========================')
    print('Loaded config\n')
    pprint(config)
    print('========================\n')

    return config


def exp_init(cmd, exps_folder=None, exp_name=None):
    '''
    common actions for setuping an experiment:
    - create experiment folder
    - dump config in it
    - create cmd file
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

    # cmd
    cmd_path = os.path.join(model_folder, 'cmd.txt')
    with open(cmd_path, 'w+') as f_cmd:
        f_cmd.write(cmd + '\n')

    # model
    src_folder = os.path.dirname(os.path.realpath(__file__)) + '/..'
    dst_folder = os.path.join(model_folder, 'model_src/')
    shutil.copytree(src_folder, dst_folder)

    return model_folder
