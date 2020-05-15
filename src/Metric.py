from abc import ABC, abstractmethod
import torch


def sparse_categorical_accuracy(log_ps, labels):
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


class Metrics():
    def __init__(self, metrics):
        self.metrics = metrics

    def get(self, name):
        return list(filter(lambda m: m.name == name, self.metrics))[0]

    def update(self, y_pred, y_true):
        for m in self.metrics:
            m.accumulate(y_pred, y_true)

    def reset(self):
        for m in self.metrics:
            m.reset()


class Metric(ABC):
    def __init__(self, type='count'):
        self.type = type
        self.name = ''
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


class CountMetric(Metric):
    def __init__(self):
        super().__init__()
        self.name = 'count'

    def accumulate(self, y_pred, y_true):
        self._add(y_pred.shape[0])


class AccuracyMetric(Metric):
    def __init__(self, accuracy_fn):
        super().__init__(type="average")
        self.name = 'accuracy'
        self.accuracy_fn = accuracy_fn

    def accumulate(self, y_pred, y_true):
        accuracy = self.accuracy_fn(y_pred, y_true)
        self._add(accuracy.item())


class LabelDiversityMetric(Metric):
    def __init__(self, n_labels):
        super().__init__()
        self.name = 'labelDiversity'
        self.n_global_labels = n_labels
        self.client_labels = set()

    def accumulate(self, y_pred, y_true):
        for label in y_true:
            self.client_labels.add(label.item())

        self.value = len(self.client_labels) / self.n_global_labels

    def reset(self):
        super().reset()
        self.client_labels = set()


# TODO how to do this? Required global model params and local model params
class ModelDivergenceMetric():
    pass
