

from tensorflow.keras.utils import Sequence


class BatchLoader(Sequence):

    def __init__(self,
                 dataset_path,
                 y_keys)
