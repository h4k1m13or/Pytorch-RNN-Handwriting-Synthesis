from pathlib import Path

import torch
from matplotlib import pyplot as plt

import src.dataset as dataset


class Parameters:
    # Model Parameters
    input_size = 3
    hidden_size = 200
    num_window_components = 10
    num_mixture_components = 20
    probability_bias = 1.0
    model_dir = Path("C:\\Users\\hakim\\Desktop\\Handwriting-Model\\src\\trained_model")
    # Dataset Parameters
    dataset_dir = Path("C:\\Users\\hakim\\Desktop\\Handwriting-Model\\src\\data")
    min_num_points = 300
    num_workers = 2
    # Training Parameters
    num_epochs = 1
    batch_size = 256
    max_norm = 400
    # Optimizer Parameters
    optimizer = "Adam"
    learning_rate = 1.0e-4
    momentum = 0.9
    nesterov = True


params = Parameters()
data = dataset.IAMDataset(parameters=params)
x = []
y = []

for s in data.strokes[1]:
    x.append(s[0])
    y.append(s[1])
for i in range(0, len(x)):
    x[i] = x[i];
print(data.ascii[1])
plt.scatter(x, y)
plt.show()

from  src import utils
onehot,strokes = data[1]
