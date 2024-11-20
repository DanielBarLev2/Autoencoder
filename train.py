import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import create_dataloader
from Autoencoder import Autoencoder

def train_model(dataset_path,
                num_epochs=20,
                batch_size=16,
                learning_rate=0.001,
                latent_dim=256,
                device='cpu',
                print_every=100):
    """
    Trains the Autoencoder model on the provided dataset.

    :param dataset_path: str
        Path to the dataset folder containing images.
    :param num_epochs: int, optional (default=20)
        Number of training epochs.
    :param batch_size: int, optional (default=16)
        Batch size for training and validation.
    :param learning_rate: float, optional (default=0.001)
        Learning rate for the optimizer.
    :param latent_dim: int, optional (default=256)
        Dimensionality of the latent space.
    :param device: str, optional (default='cpu')
    :param print_every: int, optional (default=100)
        prints logs of progress, to disable use print_every = 0.
    """
    # Create data loaders
    train_loader, val_loader = create_dataloader(dataset_path, batch_size=batch_size)

    # Initialize the model, loss function, and optimizer
    model = Autoencoder(latent_dim=latent_dim).to(device)
    # Mean Squared Error Loss
    criterion = nn.MSELoss()
    # Adaptive learning rate optimization that utilizes momentum and RMSProp via first and second moments of gradients.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if print_every:
        print("Starting training...")

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        train_loss = 0

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)

            ### Forward pass ###
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            ### Backward pass ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if print_every and ((batch_idx + 1) % print_every == 0):
                print(f"Epoch [{epoch + 1}/{num_epochs}],"
                      f" Step [{batch_idx + 1}/{len(train_loader)}],"
                      f" Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)

        if print_every:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {train_loss:.4f}")

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)

                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}")

