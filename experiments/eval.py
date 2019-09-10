import argparse
import os

from tensorflow.keras import backend as K

from data.coco.coco import CocoGenerator

from model import metrics
from model.networks.baseline import Baseline


def get_weights_paths(path, folder, epoch, prop):
    '''
    a filename looks like: weights_90_004.h5
    '''
    weights_paths = list()

    if path is not None:
        assert folder is None and epoch is None and prop is None, "can't combine exact weights path with other specifications"
        weights_paths = [path]

    else:
        assert folder is not None and epoch is not None and prop is not None, "need full specification (all is ok)"

        for filename in os.listdir(folder):
            if not filename.startswith('weights'):
                continue

            parts = filename.split('_')
            if epoch is not 'all' and int(parts[2]) != int(epoch):
                continue
            if prop is not 'all' and int(parts[1]) != int(prop):
                continue

            weights_paths.append(os.path.join(folder, filename))

        assert len(weights_paths) > 0, "didn't find any weights path for specifications!"

    print("Found %s weights paths matching specifications" % len(weights_paths))
    return weights_paths


def extract_folder_prop_epoch(weights_path):
    parts = weights_path.split('/')
    parts_name = parts[-1].split('.')[0].split('_')
    assert len(parts_name) == 3
    prop = int(parts_name[1])
    epoch = int(parts_name[2])

    exp_folder = '/'.join(parts[:-1])
    return exp_folder, prop, epoch


def main(path, folder, epoch, prop, config):

    config_path = '%s/partial-labels/config/%s.yaml' % (os.environ['HOME'], config)
    data_dir = '%s/datasets/mscoco' % os.environ['HOME']

    map_fn = metrics.map.MAP()

    # load val dataset (/!\ LONG)
    dataset_test = CocoGenerator('val', data_dir, year='2017')
    print("test data length %s" % len(dataset_test))
    X_test, Y_test = dataset_test.load_test()

    # eval for every weigths path
    for weights_path in get_weights_paths(path, folder, epoch, prop):
        exp_folder, prop, epoch = extract_folder_prop_epoch(weights_path)

        # load model
        model = Baseline('%s/partial_experiments/' % os.environ['HOME'], 80, prop)
        model.load_weights(weights_path, config_file=config_path)

        # execute mAP measures
        y_pred = model.predict(X_test)
        ap_scores = map_fn.compute_separated(Y_test, y_pred)
        print('ap scores type %s' % type(ap_scores))
        map_score = sum(ap_scores) / len(ap_scores)

        with open(os.path.join(exp_folder, 'map.csv'), 'a') as f_out:
            line = '%d,%d,%6f,' % (prop, epoch, map_score) + ','.join([str(s) for s in ap_scores]) + '\n'
            f_out.write(line)

        print("interval evaluation - epoch: {:d} - mAP score: {:.6f}".format(epoch, map_score))

        K.clear_session()
        del model


# python3 eval.py --config 'coco_baseline50_sgd_448lrs' --path '~/partial_experiments/exp_20190729_1614_baseline/weights_100_020.h5'
# python3 eval.py --config 'coco_baseline50_sgd_448lrs' --folder '~/partial_experiments/exp_20190729_1614_baseline' --epoch 20 --prop 100
# python3 eval.py --config 'coco_baseline50_sgd_448lrs' --folder '~/partial_experiments/exp_20190729_1614_baseline' --epoch 20 --prop all
# python3 eval.py --config 'coco_baseline50_sgd_448lrs' --folder '~/partial_experiments/exp_20190729_1614_baseline' --epoch all --prop all
if __name__ == '__main__':
    '''
    TODO: remove config option when config saving/loading is fixed
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='exact weights path')
    parser.add_argument('--folder', help='folder to find the weiths path(s)')
    parser.add_argument('--epoch', help='specific epoch for weights')
    parser.add_argument('--prop', help='specific prop for weights')
    parser.add_argument('--config', required=True, help='config file name (to be removed soon)')

    args = parser.parse_args()
    main(args.path, args.folder, args.epoch, args.prop, args.config)
