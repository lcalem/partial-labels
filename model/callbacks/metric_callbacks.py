import os

from tensorflow.keras.callbacks import Callback

from model.metrics.map import MAP


class MAPCallback(Callback):
    def __init__(self, x_val, y_val, exp_folder, prop):
        super(Callback, self).__init__()

        self.exp_folder = exp_folder
        self.x_val = x_val
        self.y_val = y_val
        self.prop = prop
        self.map = MAP()

    def on_epoch_end(self, epoch, logs={}):
        '''
        line:
        epoch,map,ap_0,ap_1,...ap_K
        where K is the number of classes
        '''

        y_pred = self.model.predict(self.x_val)
        ap_scores = self.map.compute_separated(self.y_val, y_pred)
        print('ap scores type %s' % type(ap_scores))
        map_score = sum(ap_scores) / len(ap_scores)

        with open(os.path.join(self.exp_folder, 'map.csv'), 'a') as f_out:
            line = '%d,%d,%6f,' % (self.prop, epoch, map_score) + ','.join([str(s) for s in ap_scores]) + '\n'
            f_out.write(line)

        print("interval evaluation - epoch: {:d} - mAP score: {:.6f}".format(epoch, map_score))
