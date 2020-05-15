import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, TensorDataset, random_split
from torchvision import datasets, transforms

from ..config import DATA_CACHE_DIR
from .Dataset import Dataset, FederatedDataset


class MNISTDataset(Dataset):
    def load(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_set = datasets.MNIST(
            f'{DATA_CACHE_DIR}/MNIST_data/', download=True, train=True, transform=transform)
        self.test_set = datasets.MNIST(
            f'{DATA_CACHE_DIR}/MNIST_data/', download=True, train=False, transform=transform)
        self.meta = {
            'n_features': 784,
            'n_labels': 10
        }

    def get_train_data(self, batch_size=64, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def get_test_data(self, batch_size=256, shuffle=False):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle)


class MNISTFederatedDataset(FederatedDataset, MNISTDataset):
    def partition_by_cluster(self, n_splits):
        n_clusters = self.settings['n_clusters']  # max 5
        cluster_size = n_splits/n_clusters
        labels = list(range(0, 10))
        rand_labels = np.random.permutation(labels)

        def swap_labels(dataset, label_1, label_2):
            samples = []
            for i, d in enumerate(dataset):
                x, y = d
                if y == label_1:
                    y = label_2
                elif y == label_2:
                    y = label_1

                samples.append((x, y))

            return TensorDataset(torch.Tensor(samples))

        client_train_sets = []
        client_test_sets = []
        for c in range(n_clusters):
            start_idx = int(c * cluster_size)
            end_idx = int(start_idx + cluster_size)

            label_1 = rand_labels[c*2]
            label_2 = rand_labels[(c*2)+1]

            # for each client in the cluster
            for k in range(start_idx, end_idx):
                train_set = swap_labels(
                    self.client_train_sets[k], label_1, label_2)
                client_train_sets.append(train_set)
                test_set = swap_labels(
                    self.client_test_sets[k], label_1, label_2)
                client_test_sets.append(test_set)

        # Update the client train/test sets in the dataset object
        self.client_train_sets = client_train_sets
        self.client_test_sets = client_test_sets

    def partition_non_iid(self, n_splits):
        def sort_data_by_label(dataset):
            # Sort data by label
            label_idxs = {}
            for k, (features, label) in enumerate(dataset):
                if label in label_idxs:
                    label_idxs[label].append(k)
                else:
                    label_idxs[label] = [k]

            # Concat into a single array (now sorted by label)
            sorted_data = []
            for k in sorted(label_idxs):
                sorted_data += label_idxs[k]

            return sorted_data

        sorted_train_set = sort_data_by_label(self.train_set)
        sorted_test_set = sort_data_by_label(self.test_set)

        n_shards = 2 * n_splits
        shards = np.random.permutation(n_shards)

        # Select 2 shards of the sorted train data for each client
        self.client_train_sets = []
        idxs = np.split(np.array(sorted_train_set), n_shards)
        for i in range(0, n_shards, 2):
            self.client_train_sets.append(
                ConcatDataset([
                    Subset(self.train_set, idxs[shards[i]]),
                    Subset(self.train_set, idxs[shards[i+1]])
                ])
            )

        # Select 2 shards of the sorted train data for each client
        self.client_test_sets = []
        idxs = np.split(np.array(sorted_test_set), n_shards)
        for i in range(0, n_shards, 2):
            self.client_test_sets.append(
                ConcatDataset([
                    Subset(self.test_set, idxs[shards[i]]),
                    Subset(self.test_set, idxs[shards[i+1]])
                ])
            )


    def partition(self, n_splits=100, strategy='iid'):
        if strategy == 'iid' or strategy == 'clustered':
            self.client_train_sets = random_split(self.train_set, [int(
                self.train_set.data.shape[0] / n_splits) for _ in range(n_splits)])
            self.client_test_sets = random_split(self.test_set, [int(
                self.test_set.data.shape[0] / n_splits) for _ in range(n_splits)])

            # Produces a different conditional distribution for each cluster by
            # switching the labels of 2 of the classes (different in each cluster)
            if strategy == 'clustered':
                self.partition_by_cluster(n_splits)

        else:
           self.partition_non_iid(n_splits)

    def get_train_data_for_client(self, k,  batch_size=10, shuffle=True):
        assert self.__getattribute__(
            'client_train_sets'), "Data must be partitioned before calling this method"
        # batch size of 0 corresponds to all data available to the client
        batch_size = batch_size if batch_size != 0 else len(
            self.client_train_sets[k])
        return DataLoader(self.client_train_sets[k], batch_size=batch_size, shuffle=shuffle)

    def get_test_data_for_client(self, k, batch_size=256, shuffle=False):
        assert self.__getattribute__(
            'client_test_sets'), "Data must be partitioned before calling this method"
        return DataLoader(self.client_test_sets[k], batch_size=batch_size, shuffle=shuffle)
