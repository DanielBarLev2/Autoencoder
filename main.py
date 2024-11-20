import config
from train import train_model

if __name__ == '__main__':
    train_model(dataset_path=config.dataset_path,
                num_epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                device=config.device,
                print_every=100)
