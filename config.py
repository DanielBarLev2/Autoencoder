import torch
# Configuration file for hyperparameters and dataset paths

# Dataset Path
dataset_path = 'C:/Users/danie/OneDrive/Desktop/dataset'

# Hyperparameters
batch_size = 64
validation_size = 1000
img_size = (256, 256)
epochs = 50
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
