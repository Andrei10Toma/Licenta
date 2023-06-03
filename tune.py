from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
import torch.nn as nn
from torch.optim import Adam
from location_dataset import LocationDatasetDateTimeComponents
from models.gru_model import DRNN_mobility
from world import *

# Load the dataset
location_dataset: LocationDatasetDateTimeComponents = LocationDatasetDateTimeComponents(BRIGHTKITE, seq_len_train=10, seq_len_test=10)

input_size = location_dataset.train_set.x_train.shape[1]
output_size = location_dataset.unique_locations_number

model: DRNN_mobility = DRNN_mobility(input_size=input_size, output_size=output_size, num_layers=16, hidden_dim=32, dropout=0.0)
net = NeuralNetClassifier(
    model,
    criterion=nn.CrossEntropyLoss,
    optimizer=Adam,
    max_epochs=15,
)

param_grid = {
    'lr': [1e-3, 1e-2, 1e-1],
    'batch_size': [2**5, 2**6, 2**7],
    'module__dropout': [0.0, 0.1, 0.2],
    'module__hidden_dim': [32],
    'module__num_layers': [16],
}

grid_search = GridSearchCV(estimator=net, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(location_dataset.train_set.x_train, location_dataset.train_set.y_train)

best_params = grid_search.best_params_
print(best_params)
best_score = grid_search.best_score_
print(best_score)
