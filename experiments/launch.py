import argparse
import os
import sys

from pprint import pprint

from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import backend as K

from config import config_utils


# from data.pascalvoc.data_gen import PascalVOCDataGenerator
from data.pascalvoc.pascalvoc import PascalVOC
from data.coco.coco2 import CocoGenerator
from data.ircad_lps.ircad_lps import IrcadLPS
from experiments import launch_utils as utils

from model.callbacks.metric_callbacks import MAPCallback
from model.callbacks.save_callback import SaveModel
from model.callbacks.scheduler import lr_scheduler
from model.networks.baseline import Baseline
from model.networks.seg_baseline import SegBaseline
from model.utils import log

from model import priors

from config.config import cfg


ALL_PCT = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


class Launcher():

    def __init__(self, exp_folder, percent=None):
        '''
        exp_percents: the known label percentages of the sequential experiments to launch (default: all of them)
        '''
        self.exp_folder = exp_folder   # still not sure this should go in config or not
        self.data_dir = cfg.DATASET.PATH
        self.relabel = cfg.RELABEL.ACTIVE

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

        self.dataset_train = self.load_dataset(mode=cfg.DATASET.TRAIN, y_keys=['multilabel'], batch_size=cfg.BATCH_SIZE, p=p)
        self.dataset_test = self.load_dataset(mode=cfg.DATASET.TEST, y_keys=['multilabel'], batch_size=cfg.BATCH_SIZE)

        # callbacks
        cb_list = self.build_callbacks(p)

        # model
        self.build_model(self.dataset_train.nb_classes, p)

        steps_per_epoch = len(self.dataset_train)
        self.model.train(self.dataset_train, steps_per_epoch=steps_per_epoch, cb_list=cb_list, dataset_val=self.dataset_test)

        # cleaning (to release memory before next launch)
        K.clear_session()
        del self.model

    def launch_percentage_relabel(self, p):
        '''
        For a given known label percentage p:

        1. load dataset
        3. callbacks
        4. load / build model
        5. train
        '''

        self.dataset_test = self.load_dataset(mode=cfg.DATASET.TEST, y_keys=['multilabel'], batch_size='all')
        self.dataset_train = self.load_dataset(mode=cfg.DATASET.TRAIN, y_keys=['multilabel'], batch_size=cfg.BATCH_SIZE, p=p)

        # model
        self.build_model(self.dataset_train.nb_classes, p)

        self.prior = self.load_prior(cfg.RELABEL.PRIOR)

        for relabel_step in range(cfg.RELABEL.STEPS):
            log.printcn(log.OKBLUE, '\nDoing relabel step %s' % (relabel_step))

            # callbacks
            cb_list = self.build_callbacks(p, relabel_step=relabel_step)

            # actual training
            self.model.train(self.dataset_train, steps_per_epoch=len(self.dataset_train), cb_list=cb_list, dataset_val=self.dataset_test)

            # relabeling
            self.relabel_dataset(relabel_step)

        # cleaning (to release memory before next launch)
        K.clear_session()
        del self.model

    def load_dataset(self, mode, y_keys, batch_size, p=None):
        '''
        we keep an ugly switch for now
        TODO: better dataset mode management
        '''
        if cfg.DATASET.NAME == 'pascalvoc':
            # dataset = PascalVOCDataGenerator(mode, self.data_dir, prop=percentage)

            dataset = PascalVOC(self.data_dir, batch_size, mode, x_keys=['image'], y_keys=y_keys, p=p)
        elif cfg.DATASET.NAME == 'coco':
            dataset = CocoGenerator(self.data_dir, batch_size, mode, x_keys=['image'], y_keys=y_keys, year=cfg.DATASET.YEAR, p=p)
        elif cfg.DATASET.NAME == 'ircad_lps':
            dataset = IrcadLPS(self.data_dir, batch_size, mode, x_keys=['image'], y_keys=['segmentation'], split_name='split_1', valid_split_number=0, p=p)
        else:
            raise Exception('Unknown dataset %s' % cfg.DATASET.NAME)

        return dataset

    def build_model(self, n_classes, p):
        '''
        TODO: we keep an ugly switch for now, do a more elegant importlib base loader after
        '''
        print("building model")
        if cfg.ARCHI.NAME == 'baseline':
            self.model = Baseline(self.exp_folder, n_classes, p)
        elif cfg.ARCHI.NAME == 'seg_baseline':
            self.model = SegBaseline(self.exp_folder, n_classes, p)

        self.model.build()

    def load_prior(self, name):
        if name == 'conditional':
            return priors.ConditionalPrior(cfg.RELABEL.PRIOR_PATH)

    def build_callbacks(self, prop, relabel_step=None):
        '''
        prop = proportion of known labels of current run

        TensorBoard
        MAPCallback
        SaveModel
        LearningRateScheduler
        '''
        log.printcn(log.OKBLUE, 'Building callbacks')
        cb_list = list()

        # tensorboard
        logs_folder = os.environ['HOME'] + '/partial_experiments/tensorboard/' + self.exp_folder.split('/')[-1] + '/prop%s' % prop
        log.printcn(log.OKBLUE, 'Tensorboard log folder %s' % logs_folder)
        tensorboard = TensorBoard(log_dir=os.path.join(logs_folder, 'tensorboard'))
        cb_list.append(tensorboard)

        # Validation callback
        if cfg.CALLBACK.VAL_CB is not None:
            cb_list.append(self.build_val_cn(cfg.CALLBACK.VAL_CB, p=prop))
        else:
            log.printcn(log.WARNING, 'Skipping validation callback')

        # Save Model
        cb_list.append(SaveModel(self.exp_folder, prop, relabel_step=relabel_step))

        # Learning rate scheduler
        cb_list.append(LearningRateScheduler(lr_scheduler))

        return cb_list

    def build_val_cb(self, cb_name, p):
        '''
        Validation callback
        Different datasets require different validations, like mAP, DICE, etc
        This callback is instanciated here
        '''
        if cb_name == 'map':
            log.printcn(log.OKBLUE, 'loading mAP callback')
            X_test, Y_test = self.dataset_test[0]

            map_cb = MAPCallback(X_test, Y_test, self.exp_folder, p)
            return map_cb

        else:
            raise Exception('Invalid validation callback %s' % cb_name)

    def relabel_dataset(self, relabel_step):
        '''
        Use model to make predictions
        Use predictions to relabel elements (create a new relabeled csv dataset)
        Use created csv to update dataset train
        '''
        log.printcn(log.OKBLUE, '\nDoing relabeling inference step')

        # save new targets as file
        targets_path = os.path.join(self.exp_folder, 'relabeling', 'relabeling_%s_%sp.csv' % (relabel_step, self.prop))
        os.makedirs(os.path.dirname(targets_path), exist_ok=True)

        with open(targets_path, 'w+') as f_relabel:

            # predict
            for i in range(len(self.dataset_train)):
                x_batch, y_batch = self.dataset_train[i]

                y_pred = self.model.predict(x_batch)   # TODO not the logits!!!!!!!!
                p_k = self.prior.compute_pk(y_batch)

                y_k = self.prior.combine(y_pred, p_k)
                relabeling = self.prior.pick_relabel(y_k, y_batch)  # (BS, K)

                # write batch to relabel csv
                for i in range(len(relabeling)):
                    parts = relabeling[i]
                    relabel_line = '%s,%s\n' % (str(parts[0]), ','.parts[1:])
                    f_relabel.write(relabel_line)

        # update dataset
        self.dataset_train.update_targets(targets_path)


# python3 launch.py -o pv_baseline50_sgd_448lrs -g 2 -p 100
# python3 launch.py -o pv_baseline50_sgd_448lrs -g 2 -p 90,70,50,30,10
# python3 launch.py -o pv_partial50_sgd_448lrs -g 3 -p 90,70,50,30,10
# python3 launch.py -o coco14_baseline_lrs_nomap -g 3 -p 90
# python3 launch.py -o pv_relabel -g 3 -p 50
# python3 launch.py -o relabel_test -g 3 -p 50
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', '-o', required=True, help='options yaml file')
    parser.add_argument('--gpu', '-g', required=True, help='# of the gpu device')
    parser.add_argument('--percent', '-p', help='the specific percentage of known labels. When not specified all percentages are sequentially launched')
    parser.add_argument('--exp_name', '-n', help='optional experiment name')

    # options management
    args = parser.parse_args()
    options = utils.parse_options_file(args.options)
    config_utils.update_config(options)

    # init
    exp_folder = utils.exp_init(' '.join(sys.argv), args.exp_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    launcher = Launcher(exp_folder, percent=args.percent)
    launcher.launch()


