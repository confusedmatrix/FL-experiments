import numpy as np
import torch
import torch.nn.functional as F

def model_fn(arch, dataset):
    if arch == 'FullyConnected':
        return lambda: FullyConnected(dataset.n_features, dataset.n_labels)
    elif arch == 'CNN':
        return lambda: CNN(dataset.n_features, dataset.n_labels)
    elif arch == 'FEMNISTCNN':
        return lambda: FEMNISTCNN(dataset.n_features, dataset.n_labels)
        

class FullyConnected(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, output_size)
        
    def forward(self, x):
        x = x.float()
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)
    

class CNN(torch.nn.Module):
    def __init__(self, input_size, output_size, fc1_size=512):
        super().__init__()
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(1, 32, 5)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.pool2 = torch.nn.MaxPool2d(2)

        input_size = int(np.power((((np.sqrt(input_size)-5+1)/2)-5+1)/2, 2)*64)
        self.d1 = torch.nn.Linear(input_size, fc1_size)
        self.d2 = torch.nn.Linear(fc1_size, output_size)

    def forward(self, x):
        hw_dim = int(np.sqrt(self.input_size))
        x = torch.reshape(x, (-1, 1, hw_dim, hw_dim))
        x = x.float()
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.d1(x))
        return F.log_softmax(self.d2(x), dim=1)

class FEMNISTCNN(torch.nn.Module):
    def __init__(self, input_size, output_size, fc1_size=2048):
        super().__init__()
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(1, 32, 5)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.pool2 = torch.nn.MaxPool2d(2)

        input_size = int(np.power((((np.sqrt(input_size)-5+1)/2)-5+1)/2, 2)*64)
        self.d1 = torch.nn.Linear(input_size, fc1_size)
        self.d2 = torch.nn.Linear(fc1_size, output_size)

    def forward(self, x):
        hw_dim = int(np.sqrt(self.input_size))
        x = torch.reshape(x, (-1, 1, hw_dim, hw_dim))
        x = x.float()
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.d1(x))
        return F.log_softmax(self.d2(x), dim=1)