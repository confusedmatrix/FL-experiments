from abc import ABC, abstractmethod
import torch


class Metrics():
    def __init__(self, metrics):
        self.metrics = metrics

    def get(self, name):
        return list(filter(lambda m: m.name == name, self.metrics))[0]

    def get_all(self):
        return self.metrics

    def get_all_custom(self):
        return list(filter(lambda m: m.custom == True, self.metrics))

    def update(self, y_pred, y_true):
        for m in self.metrics:
            m.accumulate(y_pred, y_true)

    def reset(self):
        for m in self.metrics:
            m.reset()

    def print_results(self):
        output = ','.join([f'{m.name}: {m.result()}' for m in self.metrics])
        return output


class Metric(ABC):
    def __init__(self, type='count'):
        self.type = type
        self.name = ''
        self.custom = False
        self.value = 0
        self.calls = 0

    @abstractmethod
    def accumulate(self, y_pred, y_true):
        pass

    def _add(self, value):
        self.value += value
        self.calls += 1

    def result(self):
        if self.type == 'average':
            return self.value / self.calls
        else:
            return self.value

    def reset(self):
        self.value = 0
        self.calls = 0


class LossMetric(Metric):
    def __init__(self, loss_fn):
        super().__init__(type="average")
        self.name = 'loss'
        self.loss_fn = loss_fn

    def accumulate(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)
        self._add(loss.item())


class AccuracyMetric(Metric):
    def __init__(self, acc_fn):
        super().__init__(type="average")
        self.name = 'accuracy'
        self.acc_fn = acc_fn

    def accumulate(self, y_pred, y_true):
        loss = self.acc_fn(y_pred, y_true)
        self._add(loss.item())


class CountMetric(Metric):
    def __init__(self):
        super().__init__()
        self.name = 'count'

    def accumulate(self, y_pred, y_true):
        self._add(y_pred.shape[0])


class CustomMetric(Metric):
    def __init__(self, name, metric_fn):
        super().__init__(type="average")
        self.name = name
        self.custom = True
        self.metric_fn = metric_fn

    def accumulate(self, y_pred, y_true):
        result = self.metric_fn(y_pred, y_true)
        self._add(result.item())
