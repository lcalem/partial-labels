import sys


def read_dataset(dataset_path):

    data = dict()

    with open(dataset_path, 'r') as f_in:
        for line in f_in:
            parts = line.split(',')
            data[parts[0]] = [int(elt) for elt in parts[1:]]

    return data


def compare_relabel_dataset(dataset_original, dataset_relabel):
    '''
    dataset_original: the dataset used for training (with 50% missing labels as 0 for example)
    dataset_relabel: the dataset created by the relabeling step, replacing some of the zeros

    ground truth dataset: the 100% version of the dataset_original file, with no 0 and ground truths
    '''

    parts = dataset_original.split('_')
    parts[4] = '100'
    gt_dataset = '_'.join(parts)

    print('ground truth dataset path %s' % gt_dataset)

    # open datasets to get data
    gt_data = read_dataset(gt_dataset)
    ori_data = read_dataset(dataset_original)
    relabel_data = read_dataset(dataset_relabel)

    count_relabeled = 0
    count_zeros = 0

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # for each unknown label in the original dataset, check if it has been relabeled and if the relabel is correct
    for img_id, img_labels in ori_data.items():

        for i, val in enumerate(img_labels):
            # we only consider the unknown labels ofc
            if val == 0:
                count_zeros += 1
                if relabel_data[img_id][i] != 0:
                    count_relabeled += 1

                    if relabel_data[img_id][i] == 1:
                        if gt_data[img_id][i] == 1:
                            true_positives += 1
                        elif gt_data[img_id][i] == -1:
                            false_positives += 1

                    elif relabel_data[img_id][i] == -1:
                        if gt_data[img_id][i] == -1:
                            true_negatives += 1
                        elif gt_data[img_id][i] == 1:
                            false_negatives += 1
            else:
                # check we didn't relabel data that wasn't zero
                assert gt_data[img_id][i] == val
                assert relabel_data[img_id][i] == val

    count_correct = true_positives + true_negatives
    count_incorrect = count_relabeled - count_correct
    print('Out of %s zeros in the original dataset, %s (%2f %%) have been relabeled, %s (%2f %%) correct, %s (%2f %%) incorrect' % (count_zeros, count_relabeled, count_relabeled * 100 / count_zeros, count_correct, count_correct * 100 / count_relabeled, count_incorrect, count_incorrect * 100 / count_relabeled))
    print('TP: %s\nFP: %s\nTN: %s\nFN: %s' % (true_positives, false_positives, true_negatives, false_negatives))

# python3 compare_relabel.py /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_50_1.csv /home/caleml/partial_experiments/exp_20190911_1812_baseline/relabeling/relabeling_0_50p.csv
# python3 compare_relabel.py /home/caleml/datasets/pascalvoc/VOCdevkit/VOC2007/Annotations/annotations_multilabel_trainval_partial_50_1.csv /home/caleml/partial_experiments/exp_20191007_1907_baseline_logits/relabeling/relabeling_0_50p.csv
if __name__ == '__main__':
    compare_relabel_dataset(sys.argv[1], sys.argv[2])
