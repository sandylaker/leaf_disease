from abc import ABCMeta, abstractmethod


class BaseTransform(metaclass=ABCMeta):
    def __init__(self):
        pass

    def __call__(self, *batch_output):
        y_pred, y, weight = batch_output
        return self._transform(y_pred, y, weight)

    @abstractmethod
    def _transform(self, y_pred, y, weight):
        raise NotImplementedError