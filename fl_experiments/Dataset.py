from abc import ABC, abstractmethod

class Dataset(ABC):
    """
    Abstract class from which datasets should be extended
    """

    def __init__(self, settings):
        self.settings = settings
        self.train_set = None
        self.test_set = None
        self.meta = {
            'n_features': 0,
            'n_labels': 0
        }
        self.load()

    @abstractmethod
    def load(self):
        """
        Loads the dataset from source
        """
        pass

    @abstractmethod
    def get_train_data(self, batch_size=64, shuffle=True):
        """
        Return training dataset as a dataloader (e.g. batched, shuffled etc)
        """
        pass

    @abstractmethod
    def get_test_data(self, batch_size=256, shuffle=False):
        """
        Return test dataset as a dataloader (e.g. batched, shuffled etc)
        """
        pass


class FederatedDataset(Dataset):
    """
    Abstract class from which federated datasets should be extended
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.client_train_sets = None
        self.client_test_sets = None

    @abstractmethod
    def partition(self):
        """
        Partitions train and test datasets
        """
        pass

    @abstractmethod
    def get_train_data_for_client(self, k, batch_size=64, shuffle=True):
        """
        Return train dataset for a single client as a dataloader (e.g. batched, shuffled etc)
        """
        pass

    @abstractmethod
    def get_test_data_for_client(self, k, batch_size=256, shuffle=False):
        """
        Return train dataset for a single client as a dataloader (e.g. batched, shuffled etc)
        """
        pass
