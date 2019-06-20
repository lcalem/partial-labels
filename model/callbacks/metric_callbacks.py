import os

from keras.callbacks import Callback

from model import metrics


class MAPCallback(Callback):
    def __init__(self, validation_data, exp_folder):
        super(Callback, self).__init__()

        self.exp_folder = exp_folder
        self.x_val, self.y_val = validation_data
        self.map = metrics.MAP()

    def on_epoch_end(self, epoch, logs={}):
        '''
        line:
        epoch,map,ap_0,ap_1,...ap_K
        where K is the number of classes
        '''

        y_pred = self.model.predict_proba(self.x_val)
        ap_scores = self.map.compute_separated(self.y_val, y_pred)
        print('ap scores type %s' % type(ap_scores))
        map_score = sum(ap_scores) / len(ap_scores)

        with open(os.path.join(self.exp_folder, 'map.csv'), 'w+') as f_out:
            line = '%d,%6f,' % (epoch, map_score) + ','.join(ap_scores) + '\n'
            f_out.write(line)

        print("interval evaluation - epoch: {:d} - mAP score: {:.6f}".format(epoch, map_score))
