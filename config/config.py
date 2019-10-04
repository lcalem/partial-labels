from easydict import EasyDict as edict

cfg = edict()

cfg.BATCH_SIZE = 32
cfg.RANDOM_SEED = 1

cfg.VERBOSE = True
cfg.EPSILON = 1e-7
cfg.CLEANUP = False

# debug config
cfg.DEBUG = edict()
cfg.DEBUG.IS_TEST = False

# size of images
cfg.IMAGE = edict()
cfg.IMAGE.N_CHANNELS = 3
cfg.IMAGE.IMG_SIZE = 448

cfg.DATASET = edict()
cfg.DATASET.SHUFFLE = True

# Data Augmentation
cfg.DATAAUGMENTATION = edict()
cfg.DATAAUGMENTATION.DEFAULT_DICT = {"horizontal_flip": True}

# defining what callback to use
cfg.CALLBACK = edict()
cfg.CALLBACK.LR_FACTOR = 0.1
cfg.CALLBACK.LR_TRIGGER = []
cfg.CALLBACK.PATIENCE_LR = 2
cfg.CALLBACK.PATIENCE = 10
cfg.CALLBACK.MIN_DELTA = 0.0

cfg.CALLBACK.TENSORBOARD = edict()
cfg.CALLBACK.TENSORBOARD.USE_TENSORBOARD = False
cfg.CALLBACK.TENSORBOARD.SAVE_GRAPH = True

cfg.CALLBACK.VAL_CB = None

# define training params
cfg.TRAINING = edict()
cfg.TRAINING.OPTIMIZER = "adam"
cfg.TRAINING.START_LR = 0.0001
cfg.TRAINING.STEPS_PER_EPOCH = None

# multiprocessing
cfg.MULTIP = edict()
cfg.MULTIP.N_WORKERS = 2
cfg.MULTIP.MAX_QUEUE_SIZE = 10
cfg.MULTIP.USE_MULTIPROCESS = False

# specifics
cfg.RELABEL = edict()
cfg.RELABEL.ACTIVE = False
cfg.RELABEL.EPOCHS = None
