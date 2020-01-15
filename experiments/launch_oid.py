import argparse

import os
import sys

from config import config_utils

# Import Mask RCNN
from model.mrcnn.config import Config
from model.mrcnn import utils as mutils
import model.mrcnn.model as modellib

from experiments import launch_utils as utils

from data.openimage.oid_first import OIDataset


# Directory to save logs and trained model
MODEL_DIR = os.path.join(os.environ['HOME'], 'partial_experiments')

# Local path to trained weights file (TODO this is shit)
COCO_MODEL_PATH = os.path.join(os.environ['HOME'], 'partial-labels', 'experiments', 'frcnn', 'mask_rcnn_coco.h5')
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print('downloading coco weights')
    mutils.download_trained_weights(COCO_MODEL_PATH)


# from config.config import cfg

ALL_PCT = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


class OIDConfig(Config):
    '''
    Configuration for training on the OID dataset.
    Derives from the base Config class and overrides values specific to OID
    '''
    # Give the configuration a recognizable name
    NAME = "openimages"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NB_CLASSES = 500 + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (64, 128, 256)  # anchor side in pixels

    # STEPS_PER_EPOCH = 2000   # Size of the dataset
    STEPS_PER_EPOCH = 3000
    NB_EPOCH = 200

    # old config
    IMG_SIZE = 448
    DATASET_PATH = '/local/DEEPLEARNING/oid/'
    NB_CHANNELS = 3

    DATASET_TRAIN = 'train'
    DATASET_TEST = 'val'

    BATCH_SIZE = 8
    TEST_BATCH_SIZE = 'all'


class Launcher():

    def __init__(self, exp_folder, percent=100, initial_weights=None):
        '''
        exp_percents: the known label percentages of the sequential experiments to launch (default: 100)
        '''
        # temporary config
        self.config = OIDConfig()
        self.config.display()

        self.exp_folder = exp_folder   # still not sure this should go in config or not
        self.data_dir = self.config.DATASET_PATH
        self.relabel = False
        self.initial_weights = initial_weights

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

            # made two separate functions to avoid clutter
            if self.relabel:
                self.launch_percentage_relabel(p)
            else:
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

        self.dataset_train = self.load_dataset(mode=self.config.DATASET_TRAIN, batch_size=self.config.BATCH_SIZE, p=p)
        self.dataset_test = self.load_dataset(mode=self.config.DATASET_TEST, batch_size=self.config.TEST_BATCH_SIZE)
        # self.dataset_train = self.load_dataset(mode=cfg.DATASET.TRAIN, batch_size=cfg.BATCH_SIZE, p=p)
        # self.dataset_test = self.load_dataset(mode=cfg.DATASET.TEST, batch_size=cfg.TEST_BATCH_SIZE)

        # callbacks
        # cb_list = self.build_callbacks(p)

        # model
        self.build_model(self.dataset_train.nb_classes, p)

        # Fine tune all layers
        self.model.train(self.dataset_train,
                         self.dataset_val,
                         learning_rate=self.config.LEARNING_RATE / 10,
                         epochs=self.config.NB_EPOCHS,
                         layers="all")

        # # cleaning (to release memory before next launch)
        # K.clear_session()
        # del self.model

    def load_dataset(self, mode, batch_size, p=None):
        '''
        TODO: when regular cfg is used fallback on the switch
        '''
        print('loading dataset %s' % mode)
        dataset = OIDataset()
        dataset.load_oid(self.config.DATASET_PATH, batch_size, mode, self.config)
        dataset.prepare()

        return dataset

    def build_model(self, n_classes, p):
        '''
        TODO uniformiser avec le vrai launch quand model est mergé
        '''
        print("building model")
        # Create model in training mode
        self.model = modellib.MaskRCNN(mode="training",
                                       config=self.config,
                                       model_dir=self.exp_folder)

        # Which weights to start with?
        init_with = "coco"  # imagenet, coco, or last

        if init_with == "imagenet":
            self.model.load_weights(self.model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            self.model.load_weights(COCO_MODEL_PATH,
                                    by_name=True,
                                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            self.model.load_weights(self.model.find_last(), by_name=True)

    def build_callbacks(self, prop, relabel_step=None):
        '''
        prop = proportion of known labels of current run

        TensorBoard
        MAPCallback
        SaveModel
        LearningRateScheduler
        '''
        # log.printcn(log.OKBLUE, 'Building callbacks')
        # cb_list = list()

        # # # tensorboard
        # # logs_folder = os.environ['HOME'] + '/partial_experiments/tensorboard/' + self.exp_folder.split('/')[-1] + '/prop%s' % prop
        # # log.printcn(log.OKBLUE, 'Tensorboard log folder %s' % logs_folder)
        # # tensorboard = TensorBoard(log_dir=os.path.join(logs_folder, 'tensorboard'))
        # # cb_list.append(tensorboard)

        # # Validation callback
        # if cfg.CALLBACK.VAL_CB is not None:
        #     cb_list.append(self.build_val_cb(cfg.CALLBACK.VAL_CB, p=prop, relabel_step=relabel_step))
        # else:
        #     log.printcn(log.WARNING, 'Skipping validation callback')

        # # Save Model
        # cb_list.append(SaveModel(self.exp_folder, prop, relabel_step=relabel_step))

        # # Learning rate scheduler
        # cb_list.append(LearningRateScheduler(lr_scheduler))

        # return cb_list
        return list()


# python3 launch_oid.py -g 0 -o oid_baseline
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', '-o', required=True, help='options yaml file')
    parser.add_argument('--gpu', '-g', required=True, help='# of the gpu device')
    parser.add_argument('--percent', '-p', default=100, help='the specific percentage of known labels')
    parser.add_argument('--exp_name', '-n', help='optional experiment name')

    # options management
    args = parser.parse_args()
    options = utils.parse_options_file(args.options)
    config_utils.update_config(options)

    # init
    exp_folder = utils.exp_init(' '.join(sys.argv), exp_name=(args.exp_name or args.options))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        launcher = Launcher(exp_folder, percent=args.percent)
        launcher.launch()
    finally:
        # cleanup if needed (test folders)
        pass   # TODO when cfg is back
        # if cfg.CLEANUP is True:
        #     log.printcn(log.OKBLUE, 'Cleaning folder %s' % (exp_folder))
        #     shutil.rmtree(exp_folder)
