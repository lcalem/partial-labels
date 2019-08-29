
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Lambda


def kullback_leibler_div(p, q):
    p = K.clip(p, K.epsilon(), 1)
    q = K.clip(q, K.epsilon(), 1)
    return K.sum(p * K.log(p / q), axis=-1)


def kullback_leibler_div2(c):
    p = K.clip(c[0], K.epsilon(), 1)
    q = K.clip(c[1], K.epsilon(), 1)
    return K.sum(p * K.log(p / q), axis=-1)


def log_func(x):
    return K.log(x)


def exp_func(x):
    return K.exp(x)


def sum_func(x):
    return K.sum(x)


def log_layer(x):
    return Lambda(log_func)(x)


def exp_layer(x):
    return Lambda(exp_func)(x)


def div_layer(x, y):
    return Lambda(lambda inputs: inputs[0] / inputs[1])([x, y])


def sum_tensor_layer(x):
    return Lambda(sum_func)(x)
