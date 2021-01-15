import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

from fl_experiments.Dataset import Dataset, FederatedDataset
from fl_experiments.ExperimentSettings import ExperimentSettings
from fl_experiments.Experiment import Experiment

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

    def partition(self):
        """
        Partitions train and test datasets
        """
        def get_split_lengths(dataset):
            n_splits = self.settings['n_clients']
            ds_len = len(dataset)
            split_lens = [ds_len // n_splits for _ in range(n_splits)]
            if ds_len % n_splits != 0:
                split_lens[-1] += ds_len % n_splits
            return split_lens

        self.client_train_sets = random_split(self.train_set, get_split_lengths(self.train_set))
        self.client_test_sets = random_split(self.test_set, get_split_lengths(self.test_set))

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


# Define experiment parameters
settings = ExperimentSettings()
settings.add_dataset('StdDataset', DummyDataset)
settings.add_dataset('FedDataset', DummyFederatedDataset)
settings.add_model('SimpleModel', lambda ds, device: SimpleModel(input_size = ds.meta['n_features'], output_size=ds.meta['n_labels']))
settings.add_loss_fn('NLLLoss', torch.nn.NLLLoss)
settings.add_optim_fn('SGD', lambda model, config: torch.optim.SGD(model.parameters(), lr=config['learning_rate']))

def accuracy(log_ps, target):
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == target.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

settings.add_metric('accuracy', accuracy)
settings.add_metric('dummy_metric', accuracy)

args = {
    'algorithm': 'Local',
    'dataset': 'FedDataset',
    # 'algorithm': 'Centralized',
    # 'dataset': 'StdDataset',
    'model': 'SimpleModel',
    'n_clients': 10,
    'n_rounds': 20,
    'n_epochs': 1,
    'client_fraction': 1.0,
    'loss_fn': 'NLLLoss',
    'optim_fn': 'SGD'
}
settings.set_config(**args)
print(settings.config)

# Setup experiment with defined settings
experiment = Experiment(settings)

# run the experiment
experiment.run()
