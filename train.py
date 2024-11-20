import torch
import datetime
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Autoencoder import Autoencoder
from data_loader import create_dataloader

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
        print("Starts training...")

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
                percent_complete = ((batch_idx + 1) / len(train_loader)) * 100
                print(f"Epoch [{epoch + 1}/{num_epochs}],"
                      f" {percent_complete:.2f}% Complete,"
                      f" Loss: {loss.item():.6f}")

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

    # Save the model with the current datetime in the filename
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"autoencoder_{current_time}.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")


def visualize_reconstruction(dataset_path, model_path, latent_dim=256, device='cpu', img_count=10):
    """
    Visualizes the reconstruction of images using the trained Autoencoder model.

    :param dataset_path: str
        Path to the dataset folder containing images.
    :param model_path: str
        Path to the saved model file.
    :param latent_dim: int, optional (default=256)
        Dimensionality of the latent space used in the model.
    :param device: str, optional (default='cpu')
        Device to use ('cpu' or 'cuda').
    :param img_count: int, optional (default=10)
        Number of images to visualize.
    """
    # Load validation DataLoader
    _, val_loader = create_dataloader(dataset_path, batch_size=img_count)

    # Load the model
    model = Autoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode

    # Get a batch of images from the validation DataLoader
    images = next(iter(val_loader))
    images = images.to(device)

    # Reconstruct images
    with torch.no_grad():
        reconstructed = model(images)

    # Denormalize images if necessary (assumes [-1, 1] normalization was used)
    images = images.cpu() * 0.5 + 0.5  # Convert to [0, 1]
    reconstructed = reconstructed.cpu() * 0.5 + 0.5

    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, img_count, figsize=(15, 5))
    for i in range(img_count):
        # Original images
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())  # Convert tensor to numpy
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        # Reconstructed images
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")

    plt.tight_layout()
    plt.show()
