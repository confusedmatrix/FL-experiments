import itertools
from collections import OrderedDict

SETTINGS_DEFINITIONS = OrderedDict()
SETTINGS_DEFINITIONS['id'] = {
    'type': int,
    'default': 0,
    'command': ['-i', '--id'],
}
SETTINGS_DEFINITIONS['group_name'] = {
    'type': str,
    'default': None,
    'command': ['-k', '--group-name'],
}
SETTINGS_DEFINITIONS['algorithm'] = {
    'type': str,
    'default': 'FedAvg',
    'choices': ['Centralized', 'FedAvg', 'FL+HC'],
    'command': ['-a', '--algorithm'],
}
SETTINGS_DEFINITIONS['model'] = {
    'type': str,
    'default': None,
    'command': ['-m', '--model'],
}
SETTINGS_DEFINITIONS['batch_size'] = {
    'type': int,
    'default': 10,
    'command': ['-b', '--batch-size'],
}
SETTINGS_DEFINITIONS['client_fraction'] = {
    'type': float,
    'default': 0.1,
    'command': ['-c', '--client-fraction'],
}
SETTINGS_DEFINITIONS['dataset'] = {
    'type': str,
    'default': None,
    'command': ['-d', '--dataset'],
}
SETTINGS_DEFINITIONS['n_epochs'] = {
    'type': int,
    'default': 1,
    'command': ['-e', '--n_epochs'],
}
SETTINGS_DEFINITIONS['learning_rate'] = {
    'type': float,
    'default': 0.01,
    'command': ['-l', '--learning-rate'],
}
SETTINGS_DEFINITIONS['n_clients'] = {
    'type': int,
    'default': 100,
    'command': ['-n', '--n-clients'],
}
SETTINGS_DEFINITIONS['n_rounds'] = {
    'type': int,
    'default': None,
    'command': ['-r', '--n-rounds'],
}
SETTINGS_DEFINITIONS['seed'] = {
    'type': int,
    'default': 0,
    'command': ['-s', '--seed'],
}
SETTINGS_DEFINITIONS['target_accuracy'] = {
    'type': float,
    'default': 0.5,
    'command': ['-t', '--target-accuracy'],
}
# This is a flag specific to GCP AI platform jobs and indicates the location to save results to
SETTINGS_DEFINITIONS['job_dir'] = {
    'type': str,
    'default': None,
    'command': ['-j', '--job-dir'],
}
# Stops federated learning once aggregate model delta norm falls below this number - Overrides n_rounds and n_pre_cluster_rounds if set
SETTINGS_DEFINITIONS['global_weight_delta_norm_threshold'] = {
    'type': float,
    'default': None,
    'command': ['--global-weight-delta-norm-threshold'],
}

############ FL+HC specific settings ############
# Set for label-swapped classes - max = 5
SETTINGS_DEFINITIONS['n_clusters'] = {
    'type': int,
    'default': 4,
    'command': ['--n-clusters'],
}
SETTINGS_DEFINITIONS['n_pre_cluster_rounds'] = {
    'type': int,
    'default': 1,
    'command': ['--n-pre-cluster-rounds'],
}
SETTINGS_DEFINITIONS['cluster_dist_metric'] = {
    'type': str,
    'default': 'euclidean',
    'choices': ['manhattan', 'euclidean', 'cosine'],
    'command': ['--cluster-dist-metric'],
}
SETTINGS_DEFINITIONS['cluster_hierarchical_linkage'] = {
    'type': str,
    'default': 'average',
    'choices': ['ward', 'average', 'complete', 'single'],
    'command': ['--cluster-hierarchical-linkage'],
}
SETTINGS_DEFINITIONS['cluster_hierarchical_dist_threshold'] = {
    'type': float,
    'default': 5.0,
    'command': ['--cluster-hierarchical-dist_threshold'],
}


class ExperimentSettings():
    """
    Stores the specified settings
    """

    def __init__(self, datasets, models, **kwargs):
        """
        Takes settings in as kwargs and sets them as instance variables
        """

        # Set the available datasets/models
        self.datasets = datasets
        self.models = models

        # Set defaults
        for name in SETTINGS_DEFINITIONS:
            setattr(self, name, SETTINGS_DEFINITIONS[name]['default'])

        # Overwrite defaults with values from kwargs
        for name, value in kwargs.items():
            assert name in SETTINGS_DEFINITIONS, f'Unrecognized experiment setting: "{name}"'
            if 'choices' in SETTINGS_DEFINITIONS[name]:
                assert value in SETTINGS_DEFINITIONS[name]['choices'], f'Invalid value "{value}" for experiment setting: "{name}"'
            setattr(self, name, value)

    def get_settings(self):
        """
        Get settings as a sorted list of tuples (key, value)
        """
        return dict(sorted(vars(self).items()))

    def get_settings_as_command_line_arg_list(self):
        """
        Returns all settings as an argument list to be passed on the command line, e.g. ['-l', '0.01', '-r', '100', ...]
        """
        arg_list = []
        for (key, val) in self.get_settings():
            # Skip settings with a value of None (likely optional settings)
            if val is None:
                continue

            arg_list.append(SETTINGS_DEFINITIONS[key]['command'][0])
            arg_list.append(val) if val is not None else arg_list.append(
                SETTINGS_DEFINITIONS[key]['default'])

        return arg_list
