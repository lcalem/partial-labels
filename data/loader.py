

from tensorflow.keras.utils import Sequence


class BatchLoader(Sequence):

    def __init__(self,
                 dataset,
                 y_keys,
                 mode,
                 batch_size,
                 shuffle):
        pass
