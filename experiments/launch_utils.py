import datetime
import os
import shutil
import yaml

from config.config import cfg

from model.utils import log

from pprint import pprint


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

    n_epochs = cfg.TRAINING.NB_EPOCH if cfg.RELABEL.EPOCHS is None else sum(cfg.RELABEL.EPOCHS)
    log.printcn(log.OKBLUE, "Conducting experiment for %s epochs in folder %s" % (n_epochs, model_folder))

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

    shutil.copytree(src_folder, dst_folder, ignore=shutil.ignore_patterns('.*'))

    return model_folder
