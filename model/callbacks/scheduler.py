from model.utils import log
from config.config import cfg


def lr_scheduler(epoch, lr):

    if epoch in cfg.CALLBACK.LR_TRIGGER:
        newlr = cfg.CALLBACK.LR_FACTOR * lr
        log.printcn(log.WARNING, 'lr_scheduler: lr %g -> %g @ %d' % (lr, newlr, epoch))
    else:
        newlr = lr
        log.printcn(log.OKBLUE, 'lr_scheduler: lr %g @ %d' % (newlr, epoch))

    return newlr
