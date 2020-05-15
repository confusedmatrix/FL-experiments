import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.datasets.Dataset import Dataset, FederatedDataset
from src.ExperimentSettings import SETTINGS_DEFINITIONS, ExperimentSettings
from src.Experiment import Experiment

# Define a dataset or federated dataset
class DummyDataset(Dataset):
    def load(self):
        """
        Loads the dataset from source (called when class is initialised)
        """
        self.train_set = TensorDataset(torch.randn(1000, 10), 
                                       torch.randint(0, 2, (1000,)))
        self.test_set = TensorDataset(torch.randn(100, 10), 
                                      torch.randint(0, 2, (100,)))
        self.meta = {
            'n_features': 10,
            'n_labels': 2
        }

    def get_train_data(self, batch_size=64, shuffle=True):
        """
        Return training dataset as a dataloader (e.g. batched, shuffled etc)
        """
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def get_test_data(self, batch_size=256, shuffle=False):
        """
        Return test dataset as a dataloader (e.g. batched, shuffled etc)
        """
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle)

class DummyFederatedDataset(FederatedDataset, DummyDataset):
    """
    Optionally extends DummyDataset (inherits load method from DummyDataset)
    """

    def partition(self, n_splits=100, strategy=None):
        """
        Partitions train and test datasets via given strategy
        """
        self.client_train_sets = random_split(self.train_set, [int(len(self.train_set) / n_splits) for _ in range(n_splits)])
        self.client_test_sets = random_split(self.test_set, [int(len(self.test_set) / n_splits) for _ in range(n_splits)])

    def get_train_data_for_client(self, k, batch_size=64, shuffle=True):
        """
        Return train dataset for a single client as a dataloader (e.g. batched, shuffled etc)
        """
        return DataLoader(self.client_train_sets[k], batch_size=batch_size, shuffle=shuffle)

    def get_test_data_for_client(self, k, batch_size=256, shuffle=False):
        """
        Return train dataset for a single client as a dataloader (e.g. batched, shuffled etc)
        """
        return DataLoader(self.client_test_sets[k], batch_size=batch_size, shuffle=shuffle)

# Define a model
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, 50)
        self.linear2 = torch.nn.Linear(50, output_size)
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        out = F.log_softmax(self.linear2(x), dim=1)
        return out


# Initialise the datsets
available_datasets = {
    'StdDataset': DummyDataset,
    'FedDataset': DummyFederatedDataset
}

# Define a model init function
available_models = {
    'SimpleModel': lambda ds: SimpleModel(input_size = ds.meta['n_features'], output_size=ds.meta['n_labels'])
}

# Define experiment parameters
args = {
    'algorithm': 'FedAvg',
    'dataset': 'FedDataset',
    'model': 'SimpleModel',
    'n_clients': 10,
    'n_rounds': 3
}
settings = ExperimentSettings(datasets=available_datasets, models=available_models, **args)
print(settings.get_settings())

# Setup experiment with defined settings
experiment = Experiment(settings)

# run the experiment
experiment.run()
