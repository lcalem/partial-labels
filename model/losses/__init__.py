from model.losses.bce import BCE
from model.losses.partialbce import PartialBCE


LOSSES = {
    'bce': BCE,
    'partialbce': PartialBCE
}


def get_losses(loss_names):
    '''
    Return instanciated losses
    '''
    return [get_loss(name) for name in loss_names]


def get_loss(loss_name, params):
    '''
    ugly
    '''
    if loss_name == 'bce':
        return BCE()
    elif loss_name == 'partialbce':
        return PartialBCE(**params)

