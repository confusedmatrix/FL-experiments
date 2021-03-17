from abc import ABC, abstractmethod
from collections import OrderedDict
import math
import numpy as np
import torch

from .config import GLOBAL_WEIGHTS_FILE_PATH
from .Client import ClientFactory


class AbstractServer(ABC):
    def __init__(self, settings, device, dataset, model_fn, loss_fn, optim_fn, train_metrics_fn, test_metrics_fn):
        self.settings = settings
        self.device = device
        self.dataset = dataset
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.train_metrics_fn = train_metrics_fn
        self.test_metrics_fn = test_metrics_fn

        self.init_model()
        self.model.to(self.device)

    def init_model(self):
        self.model = self.model_fn()

    def save_model_weights(self, weights=None, filename=None):
        if weights is None:
            torch.save(self.model.state_dict(), filename if filename is not None else GLOBAL_WEIGHTS_FILE_PATH)
        else:
            for k, v in enumerate(weights):
                torch.save(v, f'client-weights-{k}.pth')

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class AbstractFederatedLearningServer(AbstractServer):
    def __init__(self, settings, device, dataset, model_fn, loss_fn, optim_fn, train_metrics_fn, test_metrics_fn):
        super().__init__(settings, device, dataset, model_fn,
                         loss_fn, optim_fn, train_metrics_fn, test_metrics_fn)
        self.settings = settings
        self.clients = None
        self.client_factory = ClientFactory(settings=self.settings,
                                            dataset=self.dataset,
                                            model_fn=self.model_fn,
                                            loss_fn=self.loss_fn,
                                            optim_fn=self.optim_fn,
                                            train_metrics_fn=self.train_metrics_fn,
                                            test_metrics_fn=self.test_metrics_fn,
                                            device=self.device)

        self.dataset.partition()
        
        # Load initial global weights from file if necessary
        if self.settings['init_weights_file'] is not None:
            self.load_model_weights(torch.load(self.settings['init_weights_file'], map_location=self.device))

        self.save_model_weights()

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

    def train(self, weights=None):
        """
        Performs training only on sampled clients 
        """
        assert self.clients is not None, "Clients must be sampled before calling this method"
        results = []
        for i, k in enumerate(self.clients):
            client = self.client_factory.make_client(k)
            client_weights = None if weights is None else weights[i]
            results.append(client.train(client_weights))

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

    def evaluate(self, weights=None):
        """
        Evaluates test metrics on all clients
        """
        metrics = []
        for k in range(self.settings['n_clients']):
            client = self.client_factory.make_client(k)
            client_weights = None if weights is None else weights[k]
            metrics.append(client.evaluate(client_weights))

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
    def __init__(self, settings, device, dataset, model_fn, loss_fn, optim_fn, train_metrics_fn, test_metrics_fn):
        super().__init__(settings, device, dataset, model_fn,
                         loss_fn, optim_fn, train_metrics_fn, test_metrics_fn)

        self.train_loader = self.dataset.get_train_data(
            batch_size=self.settings['batch_size'])
        self.test_loader = self.dataset.get_test_data()

        self.train_metrics = self.train_metrics_fn()
        self.test_metrics = self.test_metrics_fn()

    def train(self):
        optim = self.optim_fn(self.model)
        for epoch in range(self.settings['n_epochs']):
            self.train_metrics.reset()
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                self.model.train()
                preds = self.model(features)
                loss = self.loss_fn(preds, labels)

                optim.zero_grad()
                loss.backward()
                optim.step()

                self.train_metrics.update(preds, labels)
            else:
                print(
                    f"TRAINING: Epoch {epoch+1}, {self.train_metrics.print_results()}")

        self.save_model_weights()
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
            f"EVALUATION: {self.test_metrics.print_results()}\n")

        return self.test_metrics


class LocalLearningOnlyServer(AbstractFederatedLearningServer):
    def sample_clients(self):
        if self.settings['client_idxs'] is not None:
            self.clients = self.settings['client_idxs']
        else:
            self.clients = range(0, self.settings['n_clients'])

    def evaluate(self, weights=None):
        """
        Evaluates test metrics on all clients
        """
        metrics = []
        for i, k in enumerate(self.clients):
            client = self.client_factory.make_client(k)
            client_weights = None if weights is None else weights[i]
            metrics.append(client.evaluate(client_weights))

        return metrics

    def aggregate_models(self, weights, metrics):
        pass


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

