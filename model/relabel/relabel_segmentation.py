import os
import numpy as np

from model.relabel.base import Relabelator
from model.networks.seg_baseline import prior_pre_processing

from config.config import cfg
from model.utils import log


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





class SegmentationRelabeling(object):

    def __init__(self, exp_folder, data_dir, nb_classes, p):
        self.exp_folder = exp_folder
        self.data_dir = data_dir
        self.exp_name = self.exp_folder.split('/')[-1]
        self.nb_classes = nb_classes
        self.p = p

        self.prior = np.load('/local/DEEPLEARNING/IRCAD_liver_pancreas_stomach/priors/pancreas_{}.npy'.format(self.p))
        self.prior = prior_pre_processing(self.prior, 1.0)


    def init_step(self, relabel_step):

        self.targets_annotations_path = os.path.join(self.data_dir, 'relabeled_annotations', self.exp_name, 'annotations', str(self.p), str(relabel_step))
        self.targets_missing_organs_path = os.path.join(self.data_dir, 'relabeled_annotations', self.exp_name, 'missing_organs', str(self.p), str(relabel_step))

        self.saved_targets_annotations_path = os.path.join(self.exp_folder, 'relabeled_annotations', 'annotations', str(self.p), str(relabel_step))
        self.saved_targets_missing_organs_path = os.path.join(self.exp_folder, 'relabeled_annotations', 'missing_organs', str(self.p), str(relabel_step))

        
        self.TP = [0,]*self.nb_classes
        self.FP = [0,]*self.nb_classes

        self.reannotation_proportion= 0.33*(relabel_step+1)
        
    
    def relabel(self, x_batch, y_batch, y_pred):

        for i in range(cfg.BATCH_SIZE):
            y_true_example = np.argmax(y_batch[0][i,:,:,:], axis=-1)
            y_true_100 = np.load(os.path.join(self.data_dir, 'annotations', '100', x_batch[2][i]))
            # Get the initial missing organs array for getting the right missing organs (because not yet relabeled)
            y_missing_organs_p = np.load(os.path.join(self.data_dir, 'missing_organs', str(self.p), x_batch[2][i]))
                
            new_annotation = np.copy(y_true_example).astype(np.uint8)
            new_missing_organ = np.copy(x_batch[1][i,:,:,:]).astype(np.uint8)
                    
            missing_organs = np.where(np.sum(y_missing_organs_p, axis=(0,1)) == 0)[0]
                    
            y_proba_example = y_pred[i,:,:,:]
            # Include the prior
            #y_proba_example = (y_proba_example * prior) / np.expand_dims(np.sum(y_proba_example * prior, axis=-1), axis=-1)
                
            y_pred_example = np.argmax(y_proba_example, axis=-1)
                
            for m_organ_id in missing_organs:
                if m_organ_id != 0:
                    pred_for_this_class = (y_pred_example == m_organ_id)
                    gt_for_this_class = (y_true_example == m_organ_id)
                    gt_100_for_this_class = (y_true_100 == m_organ_id)
                    proba_for_this_class = y_proba_example[:,:,m_organ_id]

                    new_annotation[pred_for_this_class] = m_organ_id
                    thresholded_image = get_new_missing_labels(proba_for_this_class, pred_for_this_class, reannotation_proportion=self.reannotation_proportion)

                    new_missing_organ[:,:,m_organ_id] = thresholded_image

                    self.TP[m_organ_id-1] += np.sum(np.logical_and(thresholded_image, gt_100_for_this_class))
                    self.FP[m_organ_id-1] += np.sum(np.logical_and(thresholded_image, np.logical_not(gt_100_for_this_class)))
                        

            # Save reannotation in data_dir
            annotations_file = os.path.join(self.targets_annotations_path, x_batch[2][i])
            missing_organs_file = os.path.join(self.targets_missing_organs_path, x_batch[2][i])
                
            os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
            os.makedirs(os.path.dirname(missing_organs_file), exist_ok=True)
                
            np.save(annotations_file, new_annotation)
            np.save(missing_organs_file, new_missing_organ)

            # Save reannotation in exp_folder
            annotations_file = os.path.join(self.saved_targets_annotations_path, x_batch[2][i])
            missing_organs_file = os.path.join(self.saved_targets_missing_organs_path, x_batch[2][i])
                
            os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
            os.makedirs(os.path.dirname(missing_organs_file), exist_ok=True)
                
            np.save(annotations_file, new_annotation)
            np.save(missing_organs_file, new_missing_organ)
        

    def finish_step(self, relabel_step):
        # log relabelling stats
        relabel_logpath = os.path.join(self.exp_folder, 'relabeling', 'log_relabeling.csv')
        os.makedirs(os.path.dirname(relabel_logpath), exist_ok=True)
        with open(relabel_logpath, 'a') as f_log:
            f_log.write('{},{},{},{}\n'.format(self.p, relabel_step, self.TP, self.FP))

        log.printcn(log.OKBLUE, '\tAdded {} TP and {} FP, logging into {}'.format(self.TP, self.FP, relabel_logpath))

        # Setting the path to the new targets
        self.targets_path = os.path.join(self.data_dir, 'relabeled_annotations', self.exp_name, '{}', str(self.p), str(relabel_step))
