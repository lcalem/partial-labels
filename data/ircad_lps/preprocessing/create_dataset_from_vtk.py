import numpy as np
import glob
import os
import sys
import random

from vtk_utils import read_vtk


def preprocess_images_from_vtk(data_dir):
    """
    Create the images directory from vtk images

    """
    
    all_patient_paths = glob.glob(os.path.join(data_dir, 'volumes/*'))

    for c, patient_path in enumerate(all_patient_paths):
        pid = int(patient_path.split('/')[-1])

        # Loading volume image
        image_path = os.path.join(patient_path, '{}_image.vtk'.format(pid))
        image, _ = read_vtk(image_path)

        if image.shape[1] != 512 or image.shape[2] != 512:
            print('Wrong shape : ', image.shape)
            continue

        # Preprocess
        image = image.astype(np.float32)
        image = np.clip(image, -1000, 2000) # Clip image values between [-1000, 2000]
        
        image_mean = np.mean(image)
        image_std = np.std(image)

        # Image normalization, zero mean and unit variance at a patient scale
        #image = (image - image_mean) / image_std
        image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image.astype(np.int16)

        # Save the slices
        nb_slices, height, width = image.shape
        patient_images_path = os.path.join(data_dir, 'images/{}'.format(pid))
        if not os.path.isdir(patient_images_path):
            os.makedirs(patient_images_path)

        for i in range(nb_slices):
            slice_image_path = os.path.join(data_dir, 'images/{}/{}.npy'.format(pid, str(i).zfill(3)))
            np.save(slice_image_path, [image[i,:,:],]*3)

        print('Preprocessing images : {} / {} ({:.2f}%)'.format(c, len(all_patient_paths), c/len(all_patient_paths)*100), end='\r')


def preprocess_annotations_from_vtk(data_dir, keep_annotation_proportion=100):
    """
    Create a dataset from vtk images

    keep_annotation_proportion, int which gives the percentage of annotation to keep

    {background : 0, liver : 1, pancreas : 2, stomach : 3}
    """
    
    all_patient_paths = glob.glob(os.path.join(data_dir, 'volumes/*'))

    # Removing annotations 
    keep_annotation = {}
    for patient in all_patient_paths:
        keep_annotation[patient] = [1, 1, 1]

    for i in range(3):
        patient_with_removed_organ = random.sample(all_patient_paths, int(len(all_patient_paths) * (1 - keep_annotation_proportion/100)))
        for patient in patient_with_removed_organ:
            keep_annotation[patient][i] = 0

    for c, patient_path in enumerate(all_patient_paths):
        pid = int(patient_path.split('/')[-1])

        # Loading volume annotations for the Liver Pancreas and Stomach
        liver_annot_path = os.path.join(patient_path, '{}_mask_T_Liver.vtk'.format(pid))
        pancreas_annot_path = os.path.join(patient_path, '{}_mask_Pancreas.vtk'.format(pid))
        stomach_annot_path = os.path.join(patient_path, '{}_mask_Stomach.vtk'.format(pid))

        liver_annot, _ = read_vtk(liver_annot_path)
        pancreas_annot, _ = read_vtk(pancreas_annot_path)
        stomach_annot, _ = read_vtk(stomach_annot_path)

        if liver_annot.shape[1] != 512 or liver_annot.shape[2] != 512:
            print('Wrong shape : ', liver_annot.shape)
            continue


        # Assemble annotations
        complete_annotation = np.zeros_like(liver_annot, dtype=np.uint8)
        if keep_annotation[patient_path][0] == 1:
            complete_annotation[np.where(liver_annot > 0)] = 1
        if keep_annotation[patient_path][2] == 1:
            complete_annotation[np.where(stomach_annot > 0)] = 3
        if keep_annotation[patient_path][1] == 1:
            complete_annotation[np.where(pancreas_annot > 0)] = 2

        # Save the slices
        nb_slices, height, width = liver_annot.shape
        patient_annotation_path = os.path.join(data_dir, 'annotations/{}/{}'.format(keep_annotation_proportion, pid))
        if not os.path.isdir(patient_annotation_path):
            os.makedirs(patient_annotation_path)
        missing_organs_path = os.path.join(data_dir, 'missing_organs/{}/{}'.format(keep_annotation_proportion, pid))
        if not os.path.isdir(missing_organs_path):
            os.makedirs(missing_organs_path)


        for i in range(nb_slices):
            slice_annot_path =  os.path.join(data_dir, 'annotations/{}/{}/{}.npy'.format(keep_annotation_proportion, pid, str(i).zfill(3)))
            slice_missing_annotation_path = os.path.join(data_dir, 'missing_organs/{}/{}/{}.npy'.format(keep_annotation_proportion, pid, str(i).zfill(3)))
            
            np.save(slice_annot_path, complete_annotation[i,:,:])
            np.save(slice_missing_annotation_path, keep_annotation[patient_path])
            

        print('Creating downsampled dataset ({}) : {} / {} ({:.2f}%)'.format(keep_annotation_proportion, c, len(all_patient_paths), c/len(all_patient_paths)*100), end='\r')


# python3 create_dataset_from_vtk /path/to/data/dir
if __name__ == "__main__":
    # Seed for sampling reproductibility
    random.seed(654654)
    organ_proportions = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    print('Start dataset creation')
    preprocess_images_from_vtk(sys.argv[1])
    for prop in organ_proportions:
        preprocess_annotations_from_vtk(sys.argv[1], prop)
    print('Done.')
