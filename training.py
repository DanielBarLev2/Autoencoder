import config
from model import train_model, visualize_reconstruction


def run_tests():
    current_num_epochs, min_lost, model_path = train_model(dataset_path=config.dataset_path,
                                               num_epochs=config.epochs,
                                               batch_size=config.batch_size,
                                               learning_rate=config.learning_rate,
                                               device=config.device,
                                               print_every=config.print_every,
                                               patience=config.patience,
                                               tolerance=config.tolerance)

    visualize_reconstruction(dataset_path=config.dataset_path,
                             model_path=model_path,
                             device=config.device,
                             num_epochs=current_num_epochs,
                             batch_size=config.batch_size,
                             learning_rate=config.learning_rate,
                             min_loss=min_lost
                             )