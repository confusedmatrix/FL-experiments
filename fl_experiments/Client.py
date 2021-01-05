import torch

from .config import GLOBAL_WEIGHTS_FILE_PATH

class ClientFactory():
    def __init__(self, settings, dataset, model_fn, loss_fn, optim_fn, train_metrics_fn, test_metrics_fn, device):
        self.settings = settings
        self.dataset = dataset
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.train_metrics_fn = train_metrics_fn
        self.test_metrics_fn = test_metrics_fn
        self.device = device

    def make_client(self, k):
        return Client(k, 
                      settings=self.settings,
                      train_loader=self.dataset.get_train_data_for_client(k, self.settings['batch_size']),
                      test_loader=self.dataset.get_test_data_for_client(k),
                      model_fn=self.model_fn,
                      loss_fn=self.loss_fn,
                      optim_fn=self.optim_fn,
                      train_metrics=self.train_metrics_fn(),
                      test_metrics=self.test_metrics_fn(),
                      device=self.device)


class Client():
    def __init__(self, k, settings, train_loader, test_loader, model_fn, loss_fn, optim_fn, train_metrics, test_metrics, device):
        self.id = k
        self.settings = settings
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model_fn()
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.device = device

        # Place model on cpu/gpu according to selected device
        self.model.to(self.device)

        # Initialise the model with the global model weights
        self.load_global_weights()

    def load_global_weights(self):
        """
        Load latest global model weights
        """
        self.model.load_state_dict(torch.load(GLOBAL_WEIGHTS_FILE_PATH))

    def load_weights(self, weights):
        """
        Load given model weights
        """
        self.model.load_state_dict(weights)
    
    def train(self):
        """
        Perform training
        """
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
                print(f"LOCAL TRAINING: Client #{self.id}, Epoch {epoch+1}, {self.train_metrics.print_results()}")
        
        return self.model.state_dict(), self.train_metrics

    def evaluate(self, weights=None):
        """
        Evaluate metrics using the loaded weights on the client test set
        """
        if weights is None:
            self.load_global_weights()
        else:
            self.load_weights(weights)
            
        self.test_metrics.reset()
        for features, labels in self.test_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(features)
                self.test_metrics.update(preds, labels)

        print(f"LOCAL EVALUATION: Client #{self.id}, {self.test_metrics.print_results()}")

        return self.test_metrics