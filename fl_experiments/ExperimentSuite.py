import csv
import itertools
import math
import random
from .ExperimentSettings import SETTINGS_DEFINITIONS


class ExperimentSuite():
    """
    The base class defines methods for generating groups of settings that can be saved to CSV (and to later run as experiments)
    """
    setting_groups = []

    def __init__(self):
        pass

    @classmethod
    def generate_experiments(cls):
        """
        Generates all combinations of settings from the settings_groups
        """
        def get_setting_value(grp, name):
            option = grp[name] if name in grp else SETTINGS_DEFINITIONS[name]['default']
            if type(option) is not list:
                option = [option]
            return option

        opt_combs = []
        # Loop over setting groups and form a list of all setting combos
        for grp in cls.setting_groups:
            opts = list(map(lambda name: get_setting_value(
                grp, name), SETTINGS_DEFINITIONS))
            opt_combs += list(itertools.product(*opts))

        combs = []
        # Loop over all setting combos and set random seed for each between 0 and 9999
        seed_key = list(SETTINGS_DEFINITIONS).index('seed')
        for opt in opt_combs:
            opt = list(opt)
            opt[seed_key] = math.floor(random.random() * 9999)
            combs.append(opt)

        return combs

    @classmethod
    def write_experiments(cls, filepath, include_headers=False, overwrite=False):
        """
        Write all combinations of the settings into a CSV
        """
        opt_combs = cls.generate_experiments()

        # Get the number of lines in the CSV
        try:
            with open(filepath, 'r') as f:
                n = sum(1 for line in f)

        except(FileNotFoundError):
            n = 0

        # Open CSV for writing or appending
        mode = 'w' if overwrite else 'a'
        with open(filepath, mode) as f:
            writer = csv.writer(f)

            # Write CSV headers
            if include_headers and n == 0:
                writer.writerow(SETTINGS_DEFINITIONS.keys())
                n += 1

            # Write CSV data rows
            for k, opt in enumerate(opt_combs):
                # Add ID field (useful for debugging later)
                _id = n + k
                row = [_id] + list(opt[1:])
                writer.writerow(row)


class FedAvgExperimentSuite(ExperimentSuite):
    setting_groups = [
        # Local training (MNIST iid, MNIST non-iid, MNIST clustered, FEMNIST)
        # TODO no way to do this at the moment as evaluation is performed on the aggregated model
        # {
        #     'group_name': 'Local training only on MNIST varying partition_strategy ',
        #     'algorithm': 'FedAvg',
        #     'use_gpu': True,
        #     'model': 'CNN',
        #     'batch_size': 10,
        #     'client_fraction': 1.0,
        #     'dataset': 'MNIST',
        #     'n_epochs': 100,
        #     'learning_rate': 0.1,
        #     'n_clients': 100,
        #     'partition_strategy': ['iid', 'non-iid', 'clustered'],
        #     'target_accuracy': 0.99,
        #     'n_rounds': 1
        # },
        # {
        #     'group_name': 'Local training only on FEMNIST varying partition_strategy ',
        #     'algorithm': 'FedAvg',
        #     'use_gpu': True,
        #     'model': 'CNN',
        #     'batch_size': 10,
        #     'client_fraction': 1.0,
        #     'dataset': 'FEMNIST',
        #     'n_epochs': 367,
        #     'learning_rate': 0.1,
        #     'n_clients': 100,
        #     'partition_strategy': 'non-iid',
        #     'target_accuracy': 0.99,
        #     'n_rounds': 1
        # },

        # FedAvg (MNIST iid, MNIST non-iid, MNIST clustered, FEMNIST)
        {
            'group_name': 'FedAvg on MNIST varying client_fraction and partition_strategy',
            'algorithm': 'FedAvg',
            'use_gpu': True,
            'model': 'CNN',
            'batch_size': 10,
            'client_fraction': [0.1, 0.2, 0.5, 1.0],
            'dataset': 'MNIST',
            'n_epochs': 3,
            'learning_rate': 0.1,
            'n_clients': 100,
            'partition_strategy': ['iid', 'non-iid', 'clustered'],
            'target_accuracy': 0.99,
            'n_rounds': 100
        },
        {
            'group_name': 'FedAvg on FEMNIST varying client_fraction and partition_strategy',
            'algorithm': 'FedAvg',
            'use_gpu': True,
            'model': 'CNN',
            'batch_size': 10,
            'client_fraction': [0.1, 0.2, 0.5, 1.0],
            'dataset': 'FEMNIST',
            'n_epochs': 3,
            'learning_rate': 0.1,
            'n_clients': 367,
            'partition_strategy': 'non-iid',
            'target_accuracy': 0.80,
            'n_rounds': 100
        },

        # CFL (MNIST iid, MNIST non-iid, MNIST clustered, FEMNIST)
        # {
        #     'group_name': 'CFL on MNIST varying partition_strategy ',
        #     'algorithm': 'CFL',
        #     'use_gpu': True,
        #     'model': 'CNN',
        #     'batch_size': 10,
        #     'client_fraction': 1.0,
        #     'dataset': 'MNIST',
        #     'n_epochs': 3,
        #     'learning_rate': 0.1,
        #     'n_clients': 100,
        #     'partition_strategy': ['iid', 'non-iid', 'clustered'],
        #     'target_accuracy': 0.99,
        #     'n_rounds': 20,
        #     'max_client_weight_delta_norm_threshold': 1.0,
        #     'empirical_risk_approximation_error_bound': 0.075,
        # },
        # {
        #     'group_name': 'CFL on FEMNIST ',
        #     'algorithm': 'CFL',
        #     'use_gpu': True,
        #     'model': 'CNN',
        #     'batch_size': 10,
        #     'client_fraction': 1.0,
        #     'dataset': 'FEMNIST',
        #     'n_epochs': 3,
        #     'learning_rate': 0.1,
        #     'n_clients': 367,
        #     'partition_strategy': 'non-iid',
        #     'target_accuracy': 0.99,
        #     'n_rounds': 20,
        #     'max_client_weight_delta_norm_threshold': 1.0,
        #     'empirical_risk_approximation_error_bound': 0.075,
        # },

        # FCFL (MNIST iid, MNIST non-iid, MNIST clustered, FEMNIST)
        # TODO reduce number of options - too many experiments
        # {
        #     'group_name': 'hierarchical_clustering',
        #     'algorithm': 'FCFL',
        #     'use_gpu': True,
        #     'model': 'CNN',
        #     'batch_size': 10,
        #     'client_fraction': [0.1, 0.2, 0.5, 1.0],
        #     'dataset': 'MNIST',
        #     'n_epochs': [1, 2, 5],
        #     'learning_rate': 0.1,
        #     'n_clients': 100,
        #     'partition_strategy': ['iid', 'non-iid', 'clustered'],
        #     'target_accuracy': 0.99,
        #     'n_pre_cluster_rounds': [1, 2, 3, 5, 10, 20],
        #     'n_rounds': 50,
        #     'cluster_dist_metric': ['euclidean', 'manhattan', 'cosine'],
        #     'cluster_algorithm': 'hierarchical',
        #     'cluster_hierarchical_linkage': ['ward', 'average', 'complete', 'single'],
        #     'cluster_hierarchical_dist_threshold': [1.0, 3.0, 10.0]
        # },
        {
            'group_name': 'FCFL on MNIST varying client_fraction, partition_strategy and n_pre_cluster_rounds',
            'algorithm': 'FCFL',
            'use_gpu': True,
            'model': 'CNN',
            'batch_size': 10,
            'client_fraction': [0.1, 0.2, 0.5, 1.0],
            'dataset': 'MNIST',
            'n_epochs': 3,
            'learning_rate': 0.1,
            'n_clients': 100,
            'partition_strategy': ['iid', 'non-iid', 'clustered'],
            'target_accuracy': 0.99,
            'n_pre_cluster_rounds': [1, 3, 5, 10],
            'n_rounds': 50,
            'cluster_dist_metric': 'euclidean',
            'cluster_algorithm': 'hierarchical',
            'cluster_hierarchical_linkage': 'ward',
            'cluster_hierarchical_dist_threshold': 3.0,
        },
        {
            'group_name': 'FCFL on FEMNIST varying client_fraction and n_pre_cluster_rounds',
            'algorithm': 'FCFL',
            'use_gpu': True,
            'model': 'CNN',
            'batch_size': 10,
            'client_fraction': [0.1, 0.2, 0.5, 1.0],
            'dataset': 'FEMNIST',
            'n_epochs': 3,
            'learning_rate': 0.1,
            'n_clients': 367,
            'partition_strategy': 'non-iid',
            'target_accuracy': 0.80,
            'n_pre_cluster_rounds': [1, 3, 5, 10],
            'n_rounds': 50,
            'cluster_dist_metric': 'euclidean',
            'cluster_algorithm': 'hierarchical',
            'cluster_hierarchical_linkage': 'ward',
            'cluster_hierarchical_dist_threshold': 3.0,
        },
        {
            'group_name': 'FCFL on MNIST varying HC hyperparams',
            'algorithm': 'FCFL',
            'use_gpu': True,
            'model': 'CNN',
            'batch_size': 10,
            'client_fraction': [0.2, 0.5],
            'dataset': 'MNIST',
            'n_epochs': 3,
            'learning_rate': 0.1,
            'n_clients': 100,
            'partition_strategy': ['iid', 'non-iid', 'clustered'],
            'target_accuracy': 0.99,
            'n_pre_cluster_rounds': 10,
            'n_rounds': 50,
            'cluster_dist_metric': ['euclidean', 'manhattan'],
            'cluster_algorithm': 'hierarchical',
            'cluster_hierarchical_linkage': ['ward', 'average', 'complete', 'single'],
            'cluster_hierarchical_dist_threshold': [1.0, 3.0, 10.0]
        },
        {
            'group_name': 'FCFL on FEMNIST varying HC hyperparams',
            'algorithm': 'FCFL',
            'use_gpu': True,
            'model': 'CNN',
            'batch_size': 10,
            'client_fraction': [0.2, 0.5],
            'dataset': 'FEMNIST',
            'n_epochs': 3,
            'learning_rate': 0.1,
            'n_clients': 367,
            'partition_strategy': 'non-iid',
            'target_accuracy': 0.80,
            'n_pre_cluster_rounds': 10,
            'n_rounds': 50,
            'cluster_dist_metric': ['euclidean', 'manhattan'],
            'cluster_algorithm': 'hierarchical',
            'cluster_hierarchical_linkage': ['ward', 'average', 'complete', 'single'],
            'cluster_hierarchical_dist_threshold': [1.0, 3.0, 10.0]
        },
        {
            'group_name': 'FCFL on MNIST varying HC hyperparams',
            'algorithm': 'FCFL',
            'use_gpu': True,
            'model': 'CNN',
            'batch_size': 10,
            'client_fraction': [0.2, 0.5],
            'dataset': 'MNIST',
            'n_epochs': 3,
            'learning_rate': 0.1,
            'n_clients': 100,
            'partition_strategy': ['iid', 'non-iid', 'clustered'],
            'target_accuracy': 0.99,
            'n_pre_cluster_rounds': 10,
            'n_rounds': 50,
            'cluster_dist_metric': 'cosine',
            'cluster_algorithm': 'hierarchical',
            'cluster_hierarchical_linkage': ['average', 'complete', 'single'],
            'cluster_hierarchical_dist_threshold': [0.010, 0.013, 0.015]
        },
        {
            'group_name': 'FCFL on FEMNIST varying HC hyperparams',
            'algorithm': 'FCFL',
            'use_gpu': True,
            'model': 'CNN',
            'batch_size': 10,
            'client_fraction': [0.2, 0.5],
            'dataset': 'FEMNIST',
            'n_epochs': 3,
            'learning_rate': 0.1,
            'n_clients': 367,
            'partition_strategy': 'non-iid',
            'target_accuracy': 0.80,
            'n_pre_cluster_rounds': 10,
            'n_rounds': 50,
            'cluster_dist_metric': 'cosine',
            'cluster_algorithm': 'hierarchical',
            'cluster_hierarchical_linkage': ['average', 'complete', 'single'],
            'cluster_hierarchical_dist_threshold': [0.010, 0.013, 0.015]
        },



        # Table 1 (2NN) - McMahan 2017 (20 experiments)
        # {
        #     'group_name': 'mcmahan-1-2NN',
        #     'algorithm': 'FedAvg',
        #     'use_gpu': True,
        #     'model': 'FullyConnected',
        #     'batch_size': [0, 10],  # 0 corresponds to all client data
        #     # 0 corresponds to a single client
        #     'client_fraction': [0, 0.1, 0.2, 0.5, 1.0],
        #     'dataset': 'MNIST',
        #     'n_epochs': 1,
        #     # 'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], # 0.3 is ideal on 10^(1/3) grid
        #     'learning_rate': 0.3,
        #     'n_clients': 100,
        #     'partition_strategy': ['iid', 'non-iid'],
        #     'target_accuracy': 0.973,
        # },
        # # Table 1 (CNN) - McMahan 2017 (20 experiments)
        # {
        #     'group_name': 'mcmahan-1-CNN',
        #     'algorithm': 'FedAvg',
        #     'use_gpu': True,
        #     'model': 'CNN',
        #     'batch_size': [0, 10],  # 0 corresponds to all client data
        #     # 0 corresponds to a single client
        #     'client_fraction': [0, 0.1, 0.2, 0.5, 1.0],
        #     'dataset': 'MNIST',
        #     'n_epochs': 5,
        #     # 'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], # 0.3 is ideal on 10^(1/3) grid
        #     'learning_rate': 0.1,
        #     'n_clients': 100,
        #     'partition_strategy': ['iid', 'non-iid'],
        #     'target_accuracy': 0.993,
        # },
        # # Table 4 - McMahan 2017 (18 experiments)
        # {
        #     'group_name': 'mcmahan-4',
        #     'algorithm': 'FedAvg',
        #     'use_gpu': True,
        #     'model': 'FullyConnected',
        #     'batch_size': [0, 10, 50],  # 0 corresponds to all client data
        #     'client_fraction': 0.1,
        #     'dataset': 'MNIST',
        #     'n_epochs': [1, 5, 10, 20],
        #     # 'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], # 0.3 is ideal on 10^(1/3) grid
        #     'learning_rate': 0.3,
        #     'n_clients': 100,
        #     'partition_strategy': ['iid', 'non-iid'],
        #     'target_accuracy': 0.973,
        # },
        # # Table 2 (CNN) - McMahan 2017 (18 experiments)
        # {
        #     'group_name': 'mcmahan-2-CNN',
        #     'algorithm': 'FedAvg',
        #     'use_gpu': True,
        #     'model': 'CNN',
        #     'batch_size': [0, 10, 50],  # 0 corresponds to all client data
        #     'client_fraction': 0.1,
        #     'dataset': 'MNIST',
        #     'n_epochs': [1, 5, 10, 20],
        #     # 'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], # 0.1 is ideal on 10^(1/3) grid
        #     'learning_rate': 0.1,
        #     'n_clients': 100,
        #     'partition_strategy': ['iid', 'non-iid'],
        #     'target_accuracy': 0.993,
        # }
    ]

    def __init__(self):
        super().__init__()
