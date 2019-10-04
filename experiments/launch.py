import argparse
import os
import shutil
import sys

from pprint import pprint
import numpy as np

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
from model.utils import log

from model.networks.seg_baseline import prior_pre_processing

from model import priors
from model import relabel


from config.config import cfg


ALL_PCT = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


class Launcher():

    def __init__(self, exp_folder, percent=None, initial_weights=None):
        '''
        exp_percents: the known label percentages of the sequential experiments to launch (default: all of them)
        '''
        self.exp_folder = exp_folder   # still not sure this should go in config or not
        self.data_dir = cfg.DATASET.PATH
        self.relabel = cfg.RELABEL.ACTIVE
        self.relabel_mode = cfg.RELABEL.MODE
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

        self.dataset_train = self.load_dataset(mode=cfg.DATASET.TRAIN, batch_size=cfg.BATCH_SIZE, p=p)
        self.dataset_test = self.load_dataset(mode=cfg.DATASET.TEST, batch_size='all')

        # callbacks
        cb_list = self.build_callbacks(p)

        # model
        self.build_model(self.dataset_train.nb_classes, p)

        steps_per_epoch = len(self.dataset_train)
        self.model.train(self.dataset_train, steps_per_epoch=steps_per_epoch, cb_list=cb_list, n_epochs=cfg.TRAINING.N_EPOCHS, dataset_val=self.dataset_test)

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


        self.dataset_test = self.load_dataset(mode=cfg.DATASET.TEST, batch_size=cfg.BATCH_SIZE)
        self.dataset_train = self.load_dataset(mode=cfg.DATASET.TRAIN, batch_size=cfg.BATCH_SIZE, p=p)

        # model
        self.build_model(self.dataset_train.nb_classes, p)

        self.relabelator = self.load_relabelator(p, self.dataset_train.nb_classes)

        for relabel_step in range(cfg.RELABEL.STEPS):
            log.printcn(log.OKBLUE, '\nDoing relabel step %s' % (relabel_step))

            # callbacks
            cb_list = self.build_callbacks(p, relabel_step=relabel_step)

            # actual training
            n_epochs = cfg.TRAINING.N_EPOCHS if cfg.RELABEL.EPOCHS is None else cfg.RELABEL.EPOCHS[relabel_step]
            steps_per_epoch = len(self.dataset_train) if not cfg.TRAINING.STEPS_PER_EPOCH else cfg.TRAINING.STEPS_PER_EPOCH
            self.model.train(self.dataset_train, steps_per_epoch=steps_per_epoch, cb_list=cb_list, n_epochs=n_epochs, dataset_val=self.dataset_test)
            # self.model.train(self.dataset_train, steps_per_epoch=10, cb_list=cb_list)

            # relabeling
            self.relabel_dataset(relabel_step)

        # cleaning (to release memory before next launch)
        K.clear_session()
        del self.model

        
        
    def load_dataset(self, mode, batch_size, p=None):
        '''
        we keep an ugly switch for now
        TODO: better dataset mode management
        '''
        if cfg.DATASET.NAME == 'pascalvoc':
            dataset = PascalVOC(self.data_dir, batch_size, mode, x_keys=['image', 'image_id'], y_keys=['multilabel'], p=p)
        elif cfg.DATASET.NAME == 'coco':
            dataset = CocoGenerator(self.data_dir, batch_size, mode, x_keys=['image', 'image_id'], y_keys=['multilabel'], year=cfg.DATASET.YEAR, p=p)
        elif cfg.DATASET.NAME == 'ircad_lps':
            dataset = IrcadLPS(self.data_dir, batch_size, mode, x_keys=['image', 'ambiguity', 'image_id'], y_keys=['segmentation'], split_name='split_1', valid_split_number=0, p=p)
        else:
            raise Exception('Unknown dataset %s' % cfg.DATASET.NAME)

        return dataset

    def build_model(self, n_classes, p):
        '''
        TODO: we keep an ugly switch for now, do a more elegant importlib base loader after
        '''
        print("building model")
        if cfg.ARCHI.NAME == 'baseline':
            from model.networks.baseline import Baseline
            self.model = Baseline(self.exp_folder, n_classes, p)

        elif cfg.ARCHI.NAME == 'baseline_logits':
            from model.networks.baseline_logits import BaselineLogits
            self.model = BaselineLogits(self.exp_folder, n_classes, p)

        elif cfg.ARCHI.NAME == 'seg_baseline':
            from model.networks.seg_baseline import SegBaseline
            self.model = SegBaseline(self.exp_folder, n_classes, p)
            
        self.model.build()
        
        if self.initial_weights is not None:
            self.model.load_weights(self.initial_weights, load_config=False)

    def load_relabelator(self, p, nb_classes):
        '''
        Selects the right class for managing the relabeling depending on the option specified
        '''
        if cfg.RELABEL.NAME == 'relabel_prior':
            return relabel.PriorRelabeling(self.exp_folder, p, nb_classes, cfg.RELABEL.OPTIONS.TYPE, cfg.RELABEL.OPTIONS.THRESHOLD)
        elif cfg.RELABEL.NAME == 'relabel_sk':
            return relabel.SkRelabeling(self.exp_folder, p, nb_classes)
        elif cfg.RELABEL.NAME == 'relabel_all':
            return relabel.AllSkRelabeling(self.exp_folder, p, nb_classes)
        elif cfg.RELABEL.NAME == 'relabel_baseline':
            return relabel.BaselineRelabeling(self.exp_folder, p, nb_classes, cfg.RELABEL.OPTIONS.TYPE, cfg.RELABEL.OPTIONS.THRESHOLD)

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

        # # tensorboard
        # logs_folder = os.environ['HOME'] + '/partial_experiments/tensorboard/' + self.exp_folder.split('/')[-1] + '/prop%s' % prop
        # log.printcn(log.OKBLUE, 'Tensorboard log folder %s' % logs_folder)
        # tensorboard = TensorBoard(log_dir=os.path.join(logs_folder, 'tensorboard'))
        # cb_list.append(tensorboard)

        # Validation callback
        if cfg.CALLBACK.VAL_CB is not None:
            cb_list.append(self.build_val_cb(cfg.CALLBACK.VAL_CB, p=prop, relabel_step=relabel_step))
        else:
            log.printcn(log.WARNING, 'Skipping validation callback')

        # Save Model
        cb_list.append(SaveModel(self.exp_folder, prop, relabel_step=relabel_step))

        # Learning rate scheduler
        cb_list.append(LearningRateScheduler(lr_scheduler))

        return cb_list

    def build_val_cb(self, cb_name, p, relabel_step=None):
        '''
        Validation callback
        Different datasets require different validations, like mAP, DICE, etc
        This callback is instanciated here
        '''
        if cb_name == 'map':
            log.printcn(log.OKBLUE, 'loading mAP callback')
            X_test, Y_test = self.dataset_test[0]

            map_cb = MAPCallback(X_test, Y_test, self.exp_folder, p, relabel_step=relabel_step)
            return map_cb

        else:
            raise Exception('Invalid validation callback %s' % cb_name)

    def relabel_dataset(self, relabel_step):
        '''
        Use model to make predictions
        Use predictions to relabel elements (create a new relabeled csv dataset)
        Use created csv to update dataset train
        '''
        log.printcn(log.OKBLUE, '\nDoing relabeling inference step %s' % relabel_step)

        self.relabelator.init_step(relabel_step)

        # predict
        for i in range(len(self.dataset_train)):
            x_batch, y_batch = self.dataset_train[i]

            y_pred = self.model.predict(x_batch)   # TODO not the logits!!!!!!!!

            self.relabelator.relabel(x_batch, y_batch, y_pred)

        self.relabelator.finish_step(relabel_step)
        targets_path = self.relabelator.targets_path

        # update dataset
        self.dataset_train.update_targets(targets_path)



    def relabel_segmentation_dataset(self, relabel_step, p):
        '''
        Relabel the segmentation annotations with the current model
        '''


        def get_new_missing_labels(probabilities, prediction, reannotation_proportion):
            proba_of_the_predicted_labels = probabilities * prediction
            flatten_proba = proba_of_the_predicted_labels.flatten()
            flatten_proba = flatten_proba[np.where(flatten_proba > 0.0)]
            if len(flatten_proba) > 0:
                flatten_proba = np.sort(flatten_proba)
                nb_indices_to_keep = int(len(flatten_proba) * reannotation_proportion) # 20% of the greatest values
                if nb_indices_to_keep == 0:
                    thresholded_image = np.zeros_like(probabilities, dtype=np.int32)
                else:
                    if flatten_proba[-nb_indices_to_keep] < 0.9999:
                        thresholded_image = (probabilities > flatten_proba[-nb_indices_to_keep]).astype(np.int32)
                    else:
                        thresholded_image = np.zeros_like(probabilities, dtype=np.int32)
                        
                        where_ones = np.where(probabilities > 0.9999)
                        nb_ones = len(where_ones[0])
                        choosen_idx = np.random.choice(range(nb_ones), nb_indices_to_keep, replace=False)
                        for pos in choosen_idx:
                            indx = where_ones[0][pos]
                            indy = where_ones[1][pos]
                            thresholded_image[indx, indy] = 1
            else:
                thresholded_image = np.zeros_like(probabilities, dtype=np.int32)
                
            return thresholded_image


        exp_name = self.exp_folder.split('/')[-1]
        
        prior = np.load('/local/DEEPLEARNING/IRCAD_liver_pancreas_stomach/priors/pancreas_{}.npy'.format(p))
        prior = prior_pre_processing(prior, 1.0)

        log.printcn(log.OKBLUE, '\nDoing segmentation relabeling inference step {}, relabeling {}% of the prediction'.format(relabel_step, 0.33*(relabel_step+1)))
        targets_annotations_path = os.path.join(self.data_dir, 'relabeled_annotations', exp_name, 'annotations', str(p), str(relabel_step))
        targets_missing_organs_path = os.path.join(self.data_dir, 'relabeled_annotations', exp_name, 'missing_organs', str(p), str(relabel_step))

        saved_targets_annotations_path = os.path.join(self.exp_folder, 'relabeled_annotations', 'annotations', str(p), str(relabel_step))
        saved_targets_missing_organs_path = os.path.join(self.exp_folder, 'relabeled_annotations', 'missing_organs', str(p), str(relabel_step))
        
        TP = [0,]*self.dataset_train.nb_classes
        FP = [0,]*self.dataset_train.nb_classes
        
        for b in range(len(self.dataset_train)):
            x_batch, y_batch = self.dataset_train[b]
            y_pred = self.model.predict(x_batch)

            for i in range(cfg.BATCH_SIZE):
                y_true_example = np.argmax(y_batch[0][i,:,:,:], axis=-1)
                y_true_100 = np.load(os.path.join(self.data_dir, 'annotations', '100', x_batch[2][i]))
                # Get the initial missing organs array for getting the right missing organs (because not yet relabeled)
                y_missing_organs_p = np.load(os.path.join(self.data_dir, 'missing_organs', str(p), x_batch[2][i]))
                
                new_annotation = np.copy(y_true_example).astype(np.uint8)
                new_missing_organ = np.copy(x_batch[1][i,:,:,:]).astype(np.uint8)
                    
                missing_organs = np.where(np.sum(y_missing_organs_p, axis=(0,1)) == 0)[0]
                    
                y_proba_example = y_pred[i,:,:,:]
                # Include the prior
                y_proba_example = (y_proba_example * prior) / np.expand_dims(np.sum(y_proba_example * prior, axis=-1), axis=-1)
                
                y_pred_example = np.argmax(y_proba_example, axis=-1)
                
                for m_organ_id in missing_organs:
                    if m_organ_id != 0:
                        pred_for_this_class = (y_pred_example == m_organ_id)
                        gt_for_this_class = (y_true_example == m_organ_id)
                        gt_100_for_this_class = (y_true_100 == m_organ_id)
                        proba_for_this_class = y_proba_example[:,:,m_organ_id]

                        new_annotation[pred_for_this_class] = m_organ_id
                        thresholded_image = get_new_missing_labels(proba_for_this_class, pred_for_this_class, reannotation_proportion=0.33*(relabel_step+1))

                        new_missing_organ[:,:,m_organ_id] = thresholded_image

                        TP[m_organ_id-1] += np.sum(np.logical_and(thresholded_image, gt_100_for_this_class))
                        FP[m_organ_id-1] += np.sum(np.logical_and(thresholded_image, np.logical_not(gt_100_for_this_class)))
                        

                # Save reannotation in data_dir
                annotations_file = os.path.join(targets_annotations_path, x_batch[2][i])
                missing_organs_file = os.path.join(targets_missing_organs_path, x_batch[2][i])
                
                os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
                os.makedirs(os.path.dirname(missing_organs_file), exist_ok=True)
                
                np.save(annotations_file, new_annotation)
                np.save(missing_organs_file, new_missing_organ)

                # Save reannotation in exp_folder
                annotations_file = os.path.join(saved_targets_annotations_path, x_batch[2][i])
                missing_organs_file = os.path.join(saved_targets_missing_organs_path, x_batch[2][i])
                
                os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
                os.makedirs(os.path.dirname(missing_organs_file), exist_ok=True)
                
                np.save(annotations_file, new_annotation)
                np.save(missing_organs_file, new_missing_organ)


        # log relabelling stats
        relabel_logpath = os.path.join(self.exp_folder, 'relabeling', 'log_relabeling.csv')
        os.makedirs(os.path.dirname(relabel_logpath), exist_ok=True)
        with open(relabel_logpath, 'a') as f_log:
            f_log.write('{},{},{},{}\n'.format(p, relabel_step, TP, FP))

        log.printcn(log.OKBLUE, '\tAdded {} TP and {} FP, logging into {}'.format(TP, FP, relabel_logpath))
                
        # update dataset
        targets_path_template = os.path.join(self.data_dir, 'relabeled_annotations', exp_name, '{}', str(p), str(relabel_step))
        self.dataset_train.update_targets(targets_path_template)

        
        


# python3 launch.py -o pv_baseline50_sgd_448lrs -g 2 -p 100
# python3 launch.py -o pv_baseline50_sgd_448lrs -g 2 -p 90,70,50,30,10
# python3 launch.py -o pv_partial50_sgd_448lrs -g 3 -p 90,70,50,30,10
# python3 launch.py -o coco14_baseline_lrs_nomap -g 3 -p 90
# python3 launch.py -o pv_relabel_base_nocurriculum -g 1 -p 10
# python3 launch.py -o relabel_test -g 0 -p 10
# python3 launch.py -o coco14_baseline -g 0 -p 100
# python3 launch.py -o pv_baseline -g 0 -p 10
# python3 launch.py -o pv_relabel_base_b -g 0 -p 10
# python3 launch.py -o pv_baseline101_test -g 2 -p 10
# python3 launch.py -o pv_baseline101_val -g 0 -p 10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', '-o', required=True, help='options yaml file')
    parser.add_argument('--gpu', '-g', required=True, help='# of the gpu device')
    parser.add_argument('--percent', '-p', help='the specific percentage of known labels. When not specified all percentages are sequentially launched')
    parser.add_argument('--exp_name', '-n', help='optional experiment name')
    parser.add_argument('--initial_weights', '-w', help='optional path to the weights that should be loaded')

    # options management
    args = parser.parse_args()
    options = utils.parse_options_file(args.options)
    config_utils.update_config(options)

    # init
    exp_folder = utils.exp_init(' '.join(sys.argv), exp_name=args.exp_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        launcher = Launcher(exp_folder, percent=args.percent, initial_weights=args.initial_weights)
        launcher.launch()
    finally:
        # cleanup if needed (test folders)
        if cfg.CLEANUP is True:
            log.printcn(log.OKBLUE, 'Cleaning folder %s' % (exp_folder))
            shutil.rmtree(exp_folder)


