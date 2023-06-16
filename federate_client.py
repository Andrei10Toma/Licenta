from collections import OrderedDict
from flwr.common import NDArrays, Scalar
from models.gru_model import DRNN_mobility
from world import *
import torch
from torch import nn
import time
import sys
# from location_dataset_time_components import LocationDatasetDateTimeComponents
# from location_dataset_scaled_timestamps import LocationDatasetScaledTimestamps
# from location_dataset_clusters import LocationDatasetDateTimeComponentsCluster
# from location_dataset_clusters_kmeans import LocationDatasetDateTimeComponentsKMeansCluster
# from location_dataset_clusters_grid import LocationDatasetDateTimeComponentsGridCluster
from federate_location_dataset_clusters_grid import FederateLocationDatasetDateTimeComponentsGridCluster
from torch.utils.data import DataLoader
import flwr as fl
import argparse

parser = argparse.ArgumentParser(description='Federated Learning on Human Mobility')
parser.add_argument('--client', '-c', type=int, help='Client id', required=True)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

batch_size_train = 2**8
seq_length_train = 2**7

batch_size_test = 2**7
seq_length_test = 2**6

# Brightkite dataset
# location_dataset: LocationDatasetDateTimeComponents = LocationDatasetDateTimeComponents(BRIGHTKITE, seq_length_train, seq_length_test)
# location_dataset: LocationDatasetDateTimeComponentsCluster = LocationDatasetDateTimeComponentsCluster(BRIGHTKITE, seq_length_train, seq_length_test)
# location_dataset: LocationDatasetDateTimeComponentsKMeansCluster = LocationDatasetDateTimeComponentsKMeansCluster(BRIGHTKITE, seq_length_train, seq_length_test)
# location_dataset: LocationDatasetDateTimeComponentsGridCluster = LocationDatasetDateTimeComponentsGridCluster(BRIGHTKITE, seq_length_train, seq_length_test)
location_dataset: FederateLocationDatasetDateTimeComponentsGridCluster = FederateLocationDatasetDateTimeComponentsGridCluster(BRIGHTKITE_MODEL, seq_length_train, seq_length_test, args.client)
data_loader_train: DataLoader = DataLoader(location_dataset.train_set, batch_size=batch_size_train, shuffle=False)
data_loader_test: DataLoader = DataLoader(location_dataset.test_set, batch_size=batch_size_test, shuffle=False)
input_size = location_dataset.train_set.x_train.shape[1]
print(f'input_size={input_size}')

output_size = location_dataset.train_set.y_train.shape[1]
print(f'output_size={output_size}')

hidden_size = 256
num_layers = 4
learning_rate = 1e-3
num_epochs = 5
log_interval = 10
dropout = 0.2

model: DRNN_mobility = DRNN_mobility(input_size=input_size, output_size=output_size, num_layers=num_layers, hidden_dim=hidden_size, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

def train(eps):
    model.train()
    start_time = time.time()
    total_loss = 0
    correct = 0
    counter = 0
    for ep in range(eps):
        for batch_idx, (data, target) in enumerate(data_loader_train):
            optimizer.zero_grad()
            out = model(data.contiguous())
            pred = out.max(dim=1, keepdim=True)[1]
            loss = criterion(out, target)
            pred_one_hot = nn.functional.one_hot(pred.squeeze(), num_classes=output_size)
            correct += pred_one_hot.eq(target).all(dim=1).sum()
            counter += out.size(0)
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx > 0 and batch_idx % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| Client {:3d} | Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                    'loss {:5.8f} | accuracy {:5.4f}'.format(
                    args.client, ep, batch_idx, input_size // batch_size_train + 1, learning_rate, elapsed * 1000 / log_interval,
                    avg_loss, 100. * correct / counter))
                start_time = time.time()
                total_loss = 0
                correct = 0
                counter = 0

def evaluate():
    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        counter = 0
        for _, (data, target) in enumerate(data_loader_test):
            out = model(data.contiguous())
            pred = out.max(dim=1, keepdim=True)[1]
            loss = criterion(out, target)
            pred_one_hot = nn.functional.one_hot(pred.squeeze(), num_classes=output_size)
            correct += pred_one_hot.eq(target).all(dim=1).sum()
            counter += out.size(0)
            total_loss += loss.item()

        accuracy = 100. * correct / counter
        print(f'\nTest set: Average loss: {total_loss}  |  Accuracy: {accuracy}\n'.format())
        return total_loss, accuracy
    
class MobilityClient(fl.client.NumPyClient):


    def get_parameters(self, config) -> NDArrays:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]


    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(eps=num_epochs)
        return self.get_parameters(config={}), len(location_dataset.train_set.x_train), {}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate()
        return float(loss), len(location_dataset.test_set.x_test), {"accuracy": float(accuracy)}


if __name__ == '__main__':
    print('Starting client training...')
    fl.client.start_numpy_client(server_address='127.0.0.1:8080', client=MobilityClient())
