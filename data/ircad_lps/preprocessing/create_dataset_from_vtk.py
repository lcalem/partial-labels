import numpy as np
import glob
import os
import sys

from vtk_utils import read_vtk

def create_dataset_from_vtk(data_dir):
    """
    Create a dataset from vtk images

    {background : 0, liver : 1, pancreas : 2, stomach : 3}
    """
    
    all_patient_paths = glob.glob(os.path.join(data_dir, 'volumes/*'))

    for c, patient_path in enumerate(all_patient_paths):
        pid = int(patient_path.split('/')[-1])

        # Loading volume image and annotations for the Liver Pancreas and Stomach
        image_path = os.path.join(patient_path, '{}_image.vtk'.format(pid))
        liver_annot_path = os.path.join(patient_path, '{}_mask_T_Liver.vtk'.format(pid))
        pancreas_annot_path = os.path.join(patient_path, '{}_mask_Pancreas.vtk'.format(pid))
        stomach_annot_path = os.path.join(patient_path, '{}_mask_Stomach.vtk'.format(pid))

        image, _ = read_vtk(image_path)
        liver_annot, _ = read_vtk(liver_annot_path)
        pancreas_annot, _ = read_vtk(pancreas_annot_path)
        stomach_annot, _ = read_vtk(stomach_annot_path)

        if image.shape[1] != 512 or image.shape[2] != 512:
            print('Wrong shape : ', image.shape)
            continue
        
        # Clip image values between [-1000, 2000]
        image = np.clip(image, -1000, 2000)
        
        image_mean = np.mean(image)
        image_std = np.std(image)

        # Image normalization, zero mean and unit variance at a patient scale
        image = (image - image_mean) / image_std

        # Assemble annotations
        complete_annotation = np.zeros_like(liver_annot, dtype=np.uint8)
        complete_annotation[np.where(liver_annot == 1)] = 1
        complete_annotation[np.where(pancreas_annot == 1)] = 2
        complete_annotation[np.where(stomach_annot == 1)] = 3

        # Save the slices
        nb_slices, height, width = image.shape
        patient_images_path = os.path.join(data_dir, 'images/{}'.format(pid))
        patient_annotation_path = os.path.join(data_dir, 'annotations/{}'.format(pid))
        if not os.path.isdir(patient_images_path):
            os.makedirs(patient_images_path)
        if not os.path.isdir(patient_annotation_path):
            os.makedirs(patient_annotation_path)

        for i in range(nb_slices):
            slice_image_path = os.path.join(data_dir, 'images/{}/{}.npy'.format(pid, str(i).zfill(3)))
            slice_annot_path = os.path.join(data_dir, 'annotations/{}/{}.npy'.format(pid, str(i).zfill(3)))
            
            np.save(slice_image_path, image[i,:,:])
            np.save(slice_annot_path, complete_annotation[i,:,:])

        print('Creating dataset : {} / {} ({:.2f}%)'.format(c, len(all_patient_paths), c/len(all_patient_paths)*100), end='\r')


# python3 create_dataset_from_vtk /path/to/data/dir
if __name__ == "__main__":
    print('Start dataset creation')
    create_dataset_from_vtk(sys.argv[1])
    print('Done.')
