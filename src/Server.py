from abc import ABC, abstractmethod
from functools import reduce
from collections import OrderedDict
import math
import numpy as np
import torch

from src.config import GLOBAL_WEIGHTS_FILE_PATH
from src.Client import ClientFactory


class AbstractServer(ABC):
    def __init__(self, settings, device, dataset, model_fn, loss_fn, train_metrics_fn, test_metrics_fn):
        self.settings = settings
        self.device = device
        self.dataset = dataset
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.train_metrics_fn = train_metrics_fn
        self.test_metrics_fn = test_metrics_fn

        self.init_model()
        self.model.to(self.device)

    def init_model(self):
        self.model = self.model_fn()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class AbstractFederatedLearningServer(AbstractServer):
    def __init__(self, settings, device, dataset, model_fn, loss_fn, train_metrics_fn, test_metrics_fn):
        super().__init__(settings, device, dataset, model_fn,
                         loss_fn, train_metrics_fn, test_metrics_fn)
        self.settings = settings
        self.clients = None
        self.client_factory = ClientFactory(settings=self.settings,
                                            dataset=self.dataset,
                                            model_fn=self.model_fn,
                                            loss_fn=self.loss_fn,
                                            train_metrics_fn=self.train_metrics_fn,
                                            test_metrics_fn=self.test_metrics_fn,
                                            device=self.device)

        self.dataset.partition(n_splits=self.settings['n_clients'])
        self.save_model_weights()

    def save_model_weights(self):
        torch.save(self.model.state_dict(), GLOBAL_WEIGHTS_FILE_PATH)

    def load_model_weights(self, weights):
        self.model.load_state_dict(weights)

    def sample_clients(self):
        n_clients = 1 if self.settings['client_fraction'] == 0 else math.ceil(
            self.settings['client_fraction'] * self.settings['n_clients'])
        self.clients = np.random.permutation(
            self.settings['n_clients'])[:n_clients]

    def sample_clients_from(self, client_idxs):
        n_clients = 1 if self.settings['client_fraction'] == 0 else math.ceil(
            self.settings['client_fraction'] * len(client_idxs))
        self.clients = np.random.permutation(client_idxs)[:n_clients]

    def train(self):
        """
        Performs training only on sampled clients 
        """
        assert self.clients is not None, "Clients must be sampled before calling this method"
        results = []
        for k in self.clients:
            client = self.client_factory.make_client(k)
            results.append(client.train())

        weights, metrics = zip(*results)
        return weights, metrics

    def train_on(self, client_idxs):
        """
        Performs training on given client indexes
        """
        results = []
        for k in client_idxs:
            client = self.client_factory.make_client(k)
            results.append(client.train())

        weights, metrics = zip(*results)
        return weights, metrics

    def evaluate(self):
        """
        Evaluates test metrics on all clients
        """
        metrics = []
        for k in range(self.settings['n_clients']):
            client = self.client_factory.make_client(k)
            metrics.append(client.evaluate())

        return metrics

    def evaluate_on(self, client_idxs):
        """
        Evaluates test metrics on given client indexes
        """
        metrics = []
        for k in client_idxs:
            client = self.client_factory.make_client(k)
            metrics.append(client.evaluate())

        return metrics

    def evaluate_on_sample(self):
        """
        Evaluates test metrics only on sampled clients
        """
        assert self.clients is not None, "Clients must be sampled before calling this method"
        metrics = []
        for k in self.clients:
            client = self.client_factory.make_client(k)
            metrics.append(client.evaluate())

        return metrics

    @abstractmethod
    def aggregate_models(self, models):
        pass


class CentralizedServer(AbstractServer):
    def __init__(self, settings, device, dataset, model_fn, loss_fn, train_metrics_fn, test_metrics_fn):
        super().__init__(settings, device, dataset, model_fn,
                         loss_fn, train_metrics_fn, test_metrics_fn)

        self.train_loader = self.dataset.get_train_data(
            batch_size=self.settings['batch_size'])
        self.test_loader = self.dataset.get_test_data()

        self.train_metrics = self.train_metrics_fn()
        self.test_metrics = self.test_metrics_fn()

    def train(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.settings['learning_rate'])
        for epoch in range(self.settings['n_epochs']):
            self.train_metrics.reset()
            for features, labels in self.train_loader:
                features, labels = features.to(
                    self.device), labels.to(self.device)
                preds = self.model(features)
                # print(preds.size(), labels.size())
                loss = self.loss_fn(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.train_metrics.update(preds, labels)
            else:
                print(
                    f"TRAINING: Epoch {epoch+1}, #examples {self.train_metrics.get('count').result()}, Loss {self.train_metrics.get('loss').result()}, Accuracy {self.train_metrics.get('accuracy').result()}")

        return self.model.state_dict(), self.train_metrics

    def evaluate(self):
        self.test_metrics.reset()
        for features, labels in self.test_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(features)
                self.test_metrics.update(preds, labels)

        print(
            f"EVALUATION: #examples {self.test_metrics.get('count').result()}, Loss {self.test_metrics.get('loss').result()}, Accuracy {self.test_metrics.get('accuracy').result()}\n")

        return self.test_metrics


class FedAvgServer(AbstractFederatedLearningServer):
    def aggregate_models(self, weights, metrics):
        n_examples = list(map(lambda m: m.get('count').result(), metrics))

        # Determine client weightings based on number of samples on each client
        n_examples = torch.as_tensor(n_examples)
        client_weightings = n_examples.type(
            torch.float32) / torch.sum(n_examples)

        # Average the parameters across all layers
        keys = weights[0].keys()
        averaged_weights = OrderedDict.fromkeys(keys)
        for params in zip(*weights):
            param_sum = 0
            for k, (p, w) in enumerate(zip(params, client_weightings)):
                param_sum += weights[k][p] * w

            averaged_weights[p] = param_sum

        # Update global model weights
        self.model.load_state_dict(averaged_weights)
        self.save_model_weights()

