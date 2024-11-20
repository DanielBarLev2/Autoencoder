import torch

""" Configuration file for hyperparameters and dataset paths """

dataset_path = 'C:/dataset'

# Hyperparameters
epochs = 20 # Early Stop Condition is recommended
batch_size = 32
learning_rate = 0.0001
patience = 2
tolerance = 1e-4
validation_size = 1000
img_size = (256, 256)
device = "cuda" if torch.cuda.is_available() else "cpu"
