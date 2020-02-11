import collections
import os
import yaml

from pprint import pprint

from easydict import EasyDict as edict

from config.config import cfg


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

    return config


def dict_recursive_update(d_src, d_update):
    '''
    works with upper or lower case keys (will be converted to uppercase)
    '''
    for k, v in d_update.items():
        k = k.upper()
        if isinstance(v, collections.Mapping):
            d_src[k] = dict_recursive_update(d_src.get(k, {}), v)
        else:
            d_src[k] = v
    return d_src


def update_config(options_dict):
    if 'parent_config' in options_dict:
        parent_config = parse_options_file(options_dict['parent_config'])
        update_config(parent_config)
        del options_dict['parent_config']

    new_config = edict(options_dict)
    dict_recursive_update(cfg, new_config)
