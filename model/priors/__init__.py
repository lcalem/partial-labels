from conditional import ConditionalPrior


class BasePrior():

    def compute_pk(self, y_true):
        raise NotImplementedError

    def combine(self, sk, pk):
        raise NotImplementedError

    def pick_relabel(self, yk, y_true):
        raise NotImplementedError
