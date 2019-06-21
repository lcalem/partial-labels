import numpy as np
import tensorflow as tf

from sklearn.metrics import average_precision_score

from model import metrics

from pprint import pprint


def test_map_metric():
    map_values = {}
    # map_metric = metrics.MAP()

    y_true = np.array([[0, 0, 1, 1],
                       [1, 0, 0, 0],
                       [0, 1, 1, 0],
                       [1, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]).astype(np.int64)

    y_pred_1 = np.array([[0.1, 0.2, 0.6, 0.1],
                         [0.8, 0.05, 0.1, 0.05],
                         [0.3, 0.4, 0.1, 0.2],
                         [0.6, 0.25, 0.1, 0.05],
                         [0.1, 0.2, 0.6, 0.1],
                         [0.9, 0.0, 0.03, 0.07]]).astype(np.float32)

    y_pred_2 = np.array([[0.1, 0.2, 0.6, 0.5],
                         [0.8, 0.1, 0.1, 0.2],
                         [0.3, 0.8, 0.9, 0.2],
                         [0.6, 0.3, 0.1, 0.05],
                         [0.1, 0.2, 0.6, 0.1],
                         [0.1, 0.0, 0.03, 0.7]]).astype(np.float32)

    ap_1 = average_precision_score(y_true, y_pred_1)
    ap_2 = average_precision_score(y_true, y_pred_2)
    print(ap_1, ap_2)
    map_values['sklearn'] = [ap_1, ap_2]

    assert ap_1 < ap_2

    ########################
    num_batches = y_true.shape[0]
    num_classes = y_true.shape[1]

    y_true_tf = tf.identity(y_true)
    y_pred_1_tf = tf.identity(y_pred_1)
    y_pred_2_tf = tf.identity(y_pred_2)

    print('shape true %s, shape pred %s' % (str(y_true_tf.shape), str(y_pred_1_tf.shape)))

    _, m_ap_1 = tf.metrics.average_precision_at_k(y_true_tf, y_pred_1_tf, 4)
    _, m_ap_2 = tf.metrics.average_precision_at_k(y_true_tf, y_pred_2_tf, 4)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())

    tf_map_1, tf_map_2 = sess.run([m_ap_1, m_ap_2])
    print(tf_map_1, tf_map_2)
    map_values['tensorflow_ap'] = [tf_map_1, tf_map_2]

    ######################
    map_1 = metrics.calculate_map(y_true, y_pred_1)
    map_2 = metrics.calculate_map(y_true, y_pred_2)
    print(map_1, map_2)
    map_values['other'] = [map_1, map_2]

    pprint(map_values)


if __name__ == '__main__':
    test_map_metric()
