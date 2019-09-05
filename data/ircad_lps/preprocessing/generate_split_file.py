import os
import sys
import random

def generate_split_files(data_dir, nb_splits, split_name):
    all_patients = os.listdir(os.path.join(data_dir, 'annotations'))
    random.shuffle(all_patients)

    nb_patients = len(all_patients)
    average_patients_per_split = int(nb_patients/nb_splits)
    nb_patients_per_split = [average_patients_per_split,]*nb_splits

    if average_patients_per_split*nb_splits < nb_patients:
        for i in range(nb_patients - average_patients_per_split*nb_splits):
            nb_patients_per_split[i] += 1


    print(nb_patients_per_split)
        
    cursor = 0
    for i in range(nb_splits):
        split_ids = all_patients[cursor:cursor+nb_patients_per_split[i]]
        with open(os.path.join(data_dir, 'splits', '{}.{}'.format(split_name, i)), 'w') as f:
            for idx in split_ids:
                f.write(idx+';')
        cursor += nb_patients_per_split[i]


# python3 generate_split_file.py /local/DEEPLEARNING/IRCAD_liver_pancreas_stomach/ split_1 5 1515465
if __name__ == "__main__":
    data_dir = sys.argv[1]
    split_name = sys.argv[2]
    nb_splits = int(sys.argv[3])
    seed = sys.argv[4]

    print('Generate split file : {}, seed : {}, nb_splits : {}'.format(split_name, seed, nb_splits)) 
    
    random.seed(seed)
    generate_split_files(data_dir, nb_splits, split_name)
