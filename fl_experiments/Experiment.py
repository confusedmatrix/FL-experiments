from collections import OrderedDict
import csv
import importlib
import math
from time import time
import sys
import itertools

import GPUtil
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from .config import MAX_TIMEOUT, EXPERIMENT_SETTINGS_FILE_NAME, ROUND_RESULTS_FILE_NAME, FINAL_RESULTS_FILE_NAME
from .ExperimentSettings import ExperimentSettings
from .Metric import Metrics, LossMetric, CountMetric, AccuracyMetric, sparse_categorical_accuracy
from .Server import CentralizedServer, FedAvgServer


class Experiment():
    """
    Stores and saves the results of run/running experiment
    """

    def __init__(self, settings):
        """
        Takes in an ExperimentSettings object (as settings) and sets it as an instance variable. Initialises loss function, dataset etc
        """
        assert isinstance(settings, ExperimentSettings), "settings must implement ExperimentSettings"
        self.settings = settings
        self.config = settings.config

        # Set up dataset
        dataset_name = self.config['dataset']
        assert dataset_name in self.settings.datasets, f'Unrecognised Dataset: "{dataset_name}"'
        Dataset = self.settings.datasets[dataset_name]
        self.dataset = Dataset(self.config)

        # Set up model
        model_name = self.config['model']
        assert model_name in self.settings.models, f'Unrecognised Model: "{model_name}"'
        Model = self.settings.models[model_name]
        self.model_fn = lambda: Model(self.dataset)

        # Set up loss fn
        loss_fn_name = self.config['loss_fn']
        assert loss_fn_name in self.settings.loss_fns, f'Unrecognised Loss function: "{loss_fn_name}"'
        LossFn = self.settings.loss_fns[loss_fn_name]
        self.loss_fn = LossFn()

        # Set up optimizer
        optim_fn_name = self.config['optim_fn']
        assert optim_fn_name in self.settings.optim_fns, f'Unrecognised Optimizer: "{optim_fn_name}"'
        Optim = self.settings.optim_fns[optim_fn_name]
        self.optim_fn = lambda model: Optim(model, self.config)

        # Set up metrics
        self.train_metrics_fn = lambda: Metrics([CountMetric(),
                                                LossMetric(loss_fn=self.loss_fn),
                                                AccuracyMetric(accuracy_fn=sparse_categorical_accuracy)])
        self.test_metrics_fn = self.train_metrics_fn

        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])

        use_cuda = torch.cuda.is_available()
        self.device = torch.device(
            f'cuda:{GPUtil.getFirstAvailable()[0]}' if use_cuda else 'cpu')

        self.init_server()
        self.write_csv(EXPERIMENT_SETTINGS_FILE_NAME, self.config)

    def write_csv(self, filepath, results):
        """
        Writes experiment results for a given communication round to CSV
        """
        no_headers = False
        try:
            with open(filepath, 'r') as f:
                # File empty
                no_headers = f.readline() == ''

        # File does not exist
        except(FileNotFoundError):
            no_headers = True

        with open(filepath, 'a') as f:
            writer = csv.writer(f)
            if no_headers:
                writer.writerow(results.keys())

            writer.writerow(results.values())

    def model_weights_to_vector(self, weights):
        """
        Converts weights in PyTorch models layers into a flattened vector
        """
        weight_vector = torch.tensor([], device=self.device)
        for layer_weights in weights.values():
            weight_vector = torch.cat(
                (weight_vector, torch.flatten(layer_weights)), dim=0)

        return weight_vector

    def init_server(self):
        Server = CentralizedServer if self.config['algorithm'] == 'Centralized' else FedAvgServer
        self.server = Server(settings=self.config,
                             device=self.device,
                             dataset=self.dataset,
                             model_fn=self.model_fn,
                             loss_fn=self.loss_fn,
                             optim_fn=self.optim_fn,
                             train_metrics_fn=self.train_metrics_fn,
                             test_metrics_fn=self.test_metrics_fn)

    def get_train_test_stats(self, train_metrics, test_metrics):
        # Federated learning case
        if type(train_metrics) is tuple:
            n_train_examples = np.sum(
                list(map(lambda m: m.get('count').result(), train_metrics)))
            train_loss = np.mean(
                list(map(lambda m: m.get('loss').result(), train_metrics)))
            train_acc = np.mean(
                list(map(lambda m: m.get('accuracy').result(), train_metrics)))
            test_loss = np.mean(
                list(map(lambda m: m.get('loss').result(), test_metrics)))
            test_accs = list(
                map(lambda m: m.get('accuracy').result(), test_metrics))
            test_acc = np.mean(test_accs)
            test_acc_p10 = np.percentile(test_accs, 10)
            test_acc_p90 = np.percentile(test_accs, 90)
            test_acc_reached_target = len(list(filter(lambda x: x >= self.config['target_accuracy'], test_accs))) / len(
                test_accs) if self.config['target_accuracy'] is not None else None

        # Centralized learning case
        else:
            n_train_examples = train_metrics.get('count').result()
            train_loss = train_metrics.get('loss').result()
            train_acc = train_metrics.get('accuracy').result()
            test_loss = test_metrics.get('loss').result()
            test_acc = test_metrics.get('accuracy').result()
            test_acc_p10 = None
            test_acc_p90 = None
            test_acc_reached_target = None

        return n_train_examples, train_loss, train_acc, test_loss, test_acc, test_acc_p10, test_acc_p90, test_acc_reached_target

    def format_round_results(self, c, n_clients, client_idxs, n_train_examples, train_loss, train_acc, test_loss, test_acc, test_acc_p10, test_acc_p90, test_acc_reached_target):
        results = OrderedDict()
        results['settings_id'] = self.config['id']
        results['timestamp'] = time()
        results['n_clients'] = n_clients
        results['client_idxs'] = client_idxs
        results['n_train_examples'] = n_train_examples
        results['train_loss'] = train_loss
        results['train_acc'] = train_acc
        results['test_loss'] = test_loss
        results['test_acc'] = test_acc
        results['test_acc_p10'] = test_acc_p10
        results['test_acc_p90'] = test_acc_p90
        results['test_acc_reached_target'] = test_acc_reached_target
        results['round'] = c

        return results

    def run(self):
        print("EXPERIMENT SETTINGS")
        for k, v in self.config.items():
            print(f"{k}: {v}")
        print(f"Using device: {self.device}")

        self.start_time = time()
        c = 0
        test_acc = 0
        elapsed = 0
        cached_global_weights = None
        global_weight_delta_norm = math.inf

        # If running training to convergence (global_weight_delta_norm_threshold sets a measure for when FL has reached a stationary solution)
        if 'global_weight_delta_norm_threshold' in self.config and self.config['global_weight_delta_norm_threshold'] is not None:
            print(
                f"Running learning until convergence (global delta norm < {self.config['global_weight_delta_norm_threshold']})\n")
            def condition(
            ): return global_weight_delta_norm > self.config['global_weight_delta_norm_threshold']

        # If running training for a fixed number of communication rounds
        elif 'n_rounds' in self.config and isinstance(self.config['n_rounds'], int):
            print("Running learning for %d communication rounds\n" %
                  self.config['n_rounds'])

            def condition(): return c < self.config['n_rounds']

        # Otherwise stop once target accuracy is achieved
        else:
            print("Running learning until test set accuracy reaches: %0.2f%%\n" % (
                self.config['target_accuracy']*100))

            def condition(): return test_acc < self.config['target_accuracy']

        # Start distributed learning
        while(condition() and elapsed < MAX_TIMEOUT):
            c += 1
            print("\nCommunication round {}".format(c))

            # Cache weights before this round of training
            cached_global_weights = self.model_weights_to_vector(
                self.server.model.state_dict())

            # In distributed setting only
            if self.config['algorithm'] != 'Centralized':
                self.server.sample_clients()

            weights, train_metrics = self.server.train()

            # In distributed setting only
            if self.config['algorithm'] != 'Centralized':
                self.server.aggregate_models(weights, train_metrics)

            test_metrics = self.server.evaluate()
            n_train_examples, train_loss, train_acc, test_loss, test_acc, test_acc_p10, test_acc_p90, test_acc_reached_target = self.get_train_test_stats(
                train_metrics, test_metrics)

            print(
                f"MEAN TEST ACCURACY: {test_acc}, 10th percentile: {test_acc_p10}, 90th percentile: {test_acc_p90}, % clients reaching target accuracy: {test_acc_reached_target}")

            results = self.format_round_results(
                c, self.config['n_clients'], None, n_train_examples, train_loss, train_acc, test_loss, test_acc, test_acc_p10, test_acc_p90, test_acc_reached_target)
            self.write_csv(ROUND_RESULTS_FILE_NAME, results)

            global_weight_delta_norm = torch.dist(
                cached_global_weights, self.model_weights_to_vector(self.server.model.state_dict()), p=2)
            print(f"GLOBAL WEIGHTS DELTA NORM: {global_weight_delta_norm}")

            elapsed = time() - self.start_time
            print(f"{round(elapsed)}s elapsed")

        # Record final results to a separate CSV
        results['time_elapsed'] = elapsed
        results['timed_out'] = elapsed >= MAX_TIMEOUT
        self.write_csv(FINAL_RESULTS_FILE_NAME, results)
        print(
            f"Training {'stopped after' if elapsed >= MAX_TIMEOUT else 'complete in'} {c} communication rounds")


class FCFLExperiment(Experiment):
    """
    My implementation of a new clustered federated learning algorithm
    """

    def __init__(self, settings, client_idxs=None):
        super().__init__(settings)
        self.initial_client_idxs = client_idxs if client_idxs is not None else list(
            range(self.config['n_clients']))
        assert 'cluster_dist_metric' in self.config and self.config['cluster_dist_metric'] in (
            'manhattan', 'euclidean', 'cosine'), 'cluster_dist_metric must be: manhattan, euclidean or cosine'

        if self.config['cluster_algorithm'] == 'hierarchical':
            assert 'cluster_hierarchical_dist_threshold' in self.config and isinstance(
                self.config['cluster_hierarchical_dist_threshold'], float), 'cluster_hierarchical_dist_threshold setting must be set to a float'
            assert 'cluster_hierarchical_linkage' in self.config and self.config['cluster_hierarchical_linkage'] in (
                'ward', 'average', 'complete', 'single'), 'cluster_hierarchical_linkage setting must be: ward, average, complete or single'
            if self.config['cluster_hierarchical_linkage'] == 'ward':
                assert self.config['cluster_dist_metric'] == 'euclidean', 'When using cluster_hierarchical_linkage=ward, cluster_dist_metric must be set to euclidean'

    def run(self):
        print("EXPERIMENT SETTINGS")
        for k, v in self.config.items():
            print(f"{k}: {v}")
        print(f"Using device: {self.device}")

        self.start_time = time()
        self.runFCFL('1', self.initial_client_idxs,
                     self.server.model.state_dict())
        print("FCFL completed")

    def runFCFL(self, cluster_name, client_idxs, initial_weights, continue_clustering=True):
        print(
            f'\nRunning FCFL with {len(client_idxs)} client idxs:', client_idxs)
        # global_weights, client_weights = self.runFL(
        #     cluster_name, client_idxs, initial_weights)
        global_weights, _ = self.runFL(
            cluster_name, client_idxs, initial_weights)

        # TODO call self.server.train_on() to get the weight updated for all client in the cluster
        #  (will use global settings for n_epoch, batch_size, lr)
        # This will decouple FL from the this step so that client_fraction can be set to < 1.0
        print("\nRunning 1 round of training on all client in cluster to obtain client weight updates")
        client_weights, _ = self.server.train_on(client_idxs)
        client_weights = list(
            map(self.model_weights_to_vector, client_weights))

        # Can't cluster with with less than 2 clients
        if len(client_weights) == 1:
            print(f"Cluster {cluster_name} will not be split any further")
            return

        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity=self.config['cluster_dist_metric'],
            linkage=self.config['cluster_hierarchical_linkage'],
            distance_threshold=self.config['cluster_hierarchical_dist_threshold']).fit([cw.cpu().numpy() for cw in client_weights])
        n_clusters = clustering.n_clusters_

        print(clustering.labels_)
        print(n_clusters)

        # Check continuation criterion
        if (n_clusters > 1 and continue_clustering):
            clusters = [[] for i in range(n_clusters)]
            for i, c in enumerate(clustering.labels_):
                # clusters[c].append(self.server.clients[i])
                clusters[c].append(client_idxs[i])
            print(f"{n_clusters} clusters found", clusters)

            global_weights = self.server.model.state_dict()
            for i, c in enumerate(clusters):
                # Stop running FCFL if we exceed MAX_TIMEOUT (results up to this point will be saved in the CSVs)
                if time() - self.start_time > MAX_TIMEOUT:
                    print('FCFL stopped after max timeout')
                    return
                self.runFCFL(f"{cluster_name}.{i+1}", c,
                             global_weights, continue_clustering=False)  # dissallow sub-clustering
        else:
            print(f"Cluster {cluster_name} will not be split any further")

    def runFL(self, cluster_name, client_idxs, initial_weights):
        c = 0
        test_acc = 0
        elapsed = 0
        cached_global_weights = None
        global_weight_delta_norm = math.inf

        # If running training to convergence (global_weight_delta_norm_threshold sets a measure for when FL has reached a stationary solution)
        if 'global_weight_delta_norm_threshold' in self.config and self.config['global_weight_delta_norm_threshold'] is not None:
            print(
                f"Running learning until convergence (global delta norm < {self.config['global_weight_delta_norm_threshold']})\n")
            def condition(
            ): return global_weight_delta_norm > self.config['global_weight_delta_norm_threshold']

        # If n_pre_cluster_rounds is set and we are operating before clustering has occcured
        elif cluster_name == '1' and 'n_pre_cluster_rounds' in self.config and self.config['n_pre_cluster_rounds'] is not None:
            print("Running learning for %d communication rounds prior to clustering\n" %
                  self.config['n_pre_cluster_rounds'])

            def condition(): return c < self.config['n_pre_cluster_rounds']

        # If running training for a fixed number of communication rounds
        elif 'n_rounds' in self.config and self.config['n_rounds'] is not None:
            print("Running learning for %d communication rounds\n" %
                  self.config['n_rounds'])

            def condition(): return c < self.config['n_rounds']

        # Otherwise stop once target accuracy is achieved
        else:
            print("Running learning until test set accuracy reaches: %0.2f%%\n" % (
                self.config['target_accuracy']*100))

            def condition(): return test_acc < self.config['target_accuracy']

        # Load initial global weights on server
        self.server.load_model_weights(initial_weights)

        # Run distributed training
        while(condition() and elapsed < MAX_TIMEOUT):
            c += 1
            print("Communication round {}".format(c))

            # Cache weights before this round of training
            cached_global_weights = self.model_weights_to_vector(
                self.server.model.state_dict())

            # self.server.clients = client_idxs # use this to use all clients in cluster
            self.server.sample_clients_from(client_idxs)
            weights, train_metrics = self.server.train()

            self.server.aggregate_models(weights, train_metrics)
            test_metrics = self.server.evaluate_on(client_idxs)
            n_train_examples, train_loss, train_acc, test_loss, test_acc, test_acc_p10, test_acc_p90, test_acc_reached_target = self.get_train_test_stats(
                train_metrics, test_metrics)

            print(f"CLUSTER: {cluster_name}, MEAN TEST ACCURACY: {test_acc}, 10th percentile: {test_acc_p10}, 90th percentile: {test_acc_p90}, % clients reaching target accuracy: {test_acc_reached_target}")

            results = self.format_round_results(
                c, len(client_idxs), sorted(client_idxs), n_train_examples, train_loss, train_acc, test_loss, test_acc, test_acc_p10, test_acc_p90, test_acc_reached_target)
            results['cluster_name'] = cluster_name
            self.write_csv(ROUND_RESULTS_FILE_NAME, results)

            global_weight_delta_norm = torch.dist(
                cached_global_weights, self.model_weights_to_vector(self.server.model.state_dict()), p=2)
            print(f"GLOBAL WEIGHTS DELTA NORM: {global_weight_delta_norm}")

            elapsed = time() - self.start_time
            print(f"{round(elapsed)}s elapsed")

        # Record final results to a separate CSV
        results['time_elapsed'] = elapsed
        results['timed_out'] = elapsed >= MAX_TIMEOUT
        self.write_csv(FINAL_RESULTS_FILE_NAME, results)
        print(
            f"Training {'stopped after' if elapsed >= MAX_TIMEOUT else 'complete in'} {c} communication rounds")

        # Returns cached global weights and individual client weights (as flattened vectors)
        return cached_global_weights, list(map(self.model_weights_to_vector, weights))
