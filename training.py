import config
from model import train_model, visualize_reconstruction


def run_tests():
    for batch in config.batch_size:
        for lr in config.learning_rate:
            print(f'test with batch size {batch}, and learning rate {lr}')
            current_num_epochs, min_lost, model_path = train_model(dataset_path=config.dataset_path,
                                                                   num_epochs=config.epochs,
                                                                   batch_size=batch,
                                                                   learning_rate=lr,
                                                                   device=config.device,
                                                                   print_every=config.print_every,
                                                                   patience=config.patience,
                                                                   tolerance=config.tolerance)

            visualize_reconstruction(dataset_path=config.dataset_path,
                                     model_path=model_path,
                                     device=config.device,
                                     num_epochs=current_num_epochs,
                                     batch_size=batch,
                                     learning_rate=lr,
                                     min_loss=min_lost)
