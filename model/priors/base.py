
class BasePrior():

    def compute_pk(self, y_true):
        raise NotImplementedError

    def combine(self, sk, pk):
        raise NotImplementedError
