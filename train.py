from models.gru_model import DRNN_mobility
from world import *
import torch
import sys
from torch import nn
import time
import os
# from location_dataset_scaled_timestamps import LocationDatasetScaledTimestamps
# from location_dataset_time_components import LocationDatasetDateTimeComponents
from location_dataset_clusters import LocationDatasetDateTimeComponentsCluster
from location_dataset_clusters_kmeans import LocationDatasetDateTimeComponentsKMeansCluster
from location_dataset_clusters_grid import LocationDatasetDateTimeComponentsGridCluster
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()

batch_size_train = 2**8
seq_length_train = 2**7

batch_size_test = 2**7
seq_length_test = 2**6

# Brightkite dataset
# location_dataset: LocationDatasetDateTimeComponents = LocationDatasetDateTimeComponents(BRIGHTKITE, seq_length_train, seq_length_test)
# location_dataset: LocationDatasetDateTimeComponentsCluster = LocationDatasetDateTimeComponentsCluster(BRIGHTKITE, seq_length_train, seq_length_test)
# location_dataset: LocationDatasetDateTimeComponentsKMeansCluster = LocationDatasetDateTimeComponentsKMeansCluster(BRIGHTKITE, seq_length_train, seq_length_test)
location_dataset: LocationDatasetDateTimeComponentsGridCluster = LocationDatasetDateTimeComponentsGridCluster(BRIGHTKITE, seq_length_train, seq_length_test)
data_loader_train: DataLoader = DataLoader(location_dataset.train_set, batch_size=batch_size_train, shuffle=False)
data_loader_test: DataLoader = DataLoader(location_dataset.test_set, batch_size=batch_size_test, shuffle=False)
input_size = location_dataset.train_set.x_train.shape[1]
print(f'input_size={input_size}')

output_size = location_dataset.train_set.y_train.shape[1]
print(f'output_size={output_size}')

hidden_size = 256
num_layers = 4
learning_rate = 1e-3
num_epochs = 50
log_interval = 10
dropout = 0.2

model: DRNN_mobility = DRNN_mobility(input_size=input_size, output_size=output_size, num_layers=num_layers, hidden_dim=hidden_size, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

def train(ep):
    model.train()
    start_time = time.time()
    total_loss = 0
    correct = 0
    counter = 0
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
            print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                  'loss {:5.8f} | accuracy {:5.4f}'.format(
                ep, batch_idx, input_size // batch_size_train + 1, learning_rate, elapsed * 1000 / log_interval,
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

        print(f'\nTest set: Average loss: {total_loss}  |  Accuracy: {100. * correct / counter}\n'.format())

if __name__ == '__main__':
    print('Starting training...')
    for epoch in range(num_epochs):
        train(epoch)
        evaluate()
