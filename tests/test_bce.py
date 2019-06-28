import argparse
import os

import numpy as np
import tensorflow as tf

from model import losses


def test_bce_weights(gpu=1):
    y_true = np.array([[-1, -1, 1, 1],
                       [1, -1, -1, 0],
                       [-1, 1, 1, -1],
                       [1, 1, -1, 0],
                       [-1, -1, 1, -1],
                       [-1, 0, -1, 1]]).astype(np.int64)
    y_true = tf.convert_to_tensor(y_true, np.int64)

    y_pred = np.array([[0.1, 0.2, 0.6, 0.1],
                       [0.8, 0.05, 0.1, 0.05],
                       [0.3, 0.4, 0.1, 0.2],
                       [0.6, 0.25, 0.1, 0.05],
                       [0.1, 0.2, 0.6, 0.1],
                       [0.9, 0.0, 0.03, 0.07]]).astype(np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)

    bce_loss = losses.BCE()
    bce_value_op = bce_loss.compute_loss(y_true, y_pred, trace=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    bce_value = sess.run(bce_value_op)
    print('bce_value %s' % str(bce_value))
    assert bce_value['final_bce'].shape == (6,)
    
    nonzeros_gt = [4, 3, 4, 3, 4, 3]
    assert all([bce_value['nonzeros'][i] == gt for i, gt in enumerate(nonzeros_gt)])

    assert all([0 < val < 5 for val in bce_value['final_bce']])


if __name__ == '__main__':
    test_bce_weights()

