import config
from model import train_model


def run_tests():
    train_model(dataset_path=config.dataset_path,
                num_epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                device=config.device,
                print_every=1000,
                patience=config.patience,
                tolerance=config.tolerance)

    # model_path = "/models/autoencoder_1.pth"
    # visualize_reconstruction(config.dataset_path, model_path, latent_dim=256, device=config.device)