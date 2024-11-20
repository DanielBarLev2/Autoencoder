import os
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
                print_every=100,
                patience=3,
                tolerance=1e-4):
    """
    Trains the Autoencoder model on the provided dataset and saves the model and training/validation loss plots.

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
        Device to use for training ('cpu' or 'cuda').
    :param print_every: int, optional (default=100)
        Interval for logging training progress. Set to 0 to disable.
    :param patience: int, optional (default=3)
        Number of epochs to wait for improvement in validation loss before stopping.
    :param tolerance: float, optional (default=1e-4)
        Minimum change in validation loss to be considered as improvement.
    """
    # Create output directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Create data loaders
    train_loader, val_loader = create_dataloader(dataset_path, batch_size=batch_size)

    # Initialize the model, loss function, and optimizer
    model = Autoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize variables to track losses
    train_losses = []
    val_losses = []

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_significant_change = 0

    if print_every:
        print("Starting training...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if print_every and ((batch_idx + 1) % print_every == 0):
                percent_complete = ((batch_idx + 1) / len(train_loader)) * 100
                print(f"Epoch [{epoch + 1}/{num_epochs}], {percent_complete:.2f}% Complete, Loss: {loss.item():.6f}")

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {train_loss:.6f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.6f}")

        # Early stopping
        if abs(val_loss - best_val_loss) > tolerance:
            best_val_loss = val_loss
            epochs_without_significant_change = 0
        else:
            epochs_without_significant_change += 1
            print(f"No significant improvement for {epochs_without_significant_change} epoch(s).")

        if epochs_without_significant_change >= patience:
            print(f"Early stopping triggered. No significant improvement for {patience} consecutive epochs.")
            break

    # Save the model on significant improvement
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/autoencoder_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Validation loss improved. Model saved as {model_path}.")

    # Plot and save training and validation losses
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"plots/loss_plot_{current_time}.png"
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    print(f"Loss plot saved as {plot_path}.")
    plt.close()

    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

    current_num_epochs = len(train_losses)

    return current_num_epochs, best_val_loss, model_path,

def visualize_reconstruction(dataset_path, model_path, latent_dim=256, device='cpu', img_count=10,
                             num_epochs=20, batch_size=16, learning_rate=0.001, min_loss=None):
    """
    Visualizes the reconstruction of images using the trained Autoencoder model and saves the plot to the "results" folder.

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
    :param num_epochs: int, optional (default=20)
        Total number of epochs the model was trained for.
    :param batch_size: int, optional (default=16)
        Batch size used during training.
    :param learning_rate: float, optional (default=0.001)
        Learning rate used during training.
    :param min_loss: float, optional
        Minimum validation loss achieved during training.
    """
    os.makedirs("results", exist_ok=True)

    _, val_loader = create_dataloader(dataset_path, batch_size=img_count)

    model = Autoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    images = next(iter(val_loader))
    images = images.to(device)

    # Reconstruct images
    with torch.no_grad():
        reconstructed = model(images)

    # Denormalize images
    images = images.cpu() * 0.5 + 0.5
    reconstructed = reconstructed.cpu() * 0.5 + 0.5

    fig, axes = plt.subplots(2, img_count, figsize=(15, 7))
    for i in range(img_count):
        # Original images
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        # Reconstructed images
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")

    metadata_text = (
        f"Training Metadata:\n"
        f"Num Epochs: {num_epochs}\n"
        f"Batch Size: {batch_size}\n"
        f"Learning Rate: {learning_rate}\n"
        f"Minimal Loss Achieved: {min_loss:.6f}" if min_loss is not None else "Minimal Loss Achieved: Not Provided"
    )

    plt.subplots_adjust(bottom=0.25)

    plt.figtext(0.5, 0.02, metadata_text, wrap=True, horizontalalignment='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Save the plot to the "results" directory with a timestamped filename
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/reconstruction_{current_time}.png"
    plt.savefig(result_path)
    print(f"Reconstruction plot saved as {result_path}")
    plt.close()
