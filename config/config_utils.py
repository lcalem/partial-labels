import collections

from easydict import EasyDict as edict

from model.utils.config import cfg


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
    new_config = edict(options_dict)
    dict_recursive_update(cfg, new_config)
