import torch.nn as nn


class Autoencoder(nn.Module):
    """
        A convolutional Autoencoder for image reconstruction with Batch Normalization and Leaky ReLU activations.

        Architecture:
        ### Encoder ###:
            - Sequential layers of `Conv2d`, `BatchNorm2d`, and `LeakyReLU`.
            - Spatial dimensions are halved at each convolution layer.
            - Final features are flattened and passed through a `Linear` layer to produce the latent space.

        ### Latent Space ###
            - The latent representation has a size specified by the `latent_dim` parameter.
            - Includes `BatchNorm1d` to normalize the latent vector.

        ### Decoder ###
            - Sequential layers of `Linear`, `BatchNorm1d`, `LeakyReLU`, and `ConvTranspose2d`.
            - Reverses the encoding process, doubling spatial dimensions at each transpose convolution layer.
            - The final layer uses `Tanh` activation to output pixel values in the range [-1, 1].

        Hyperparameters:
        - `latent_dim`: Controls the size of the compressed representation in the latent space.
        - `negative_slope`: Slope for the negative part of `LeakyReLU`.
        """

    def __init__(self, latent_dim=256, negative_slope=0.01):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 16 * 16),
            nn.BatchNorm1d(512 * 16 * 16),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Unflatten(1, (512, 16, 16)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        :param x: Tensor
            Input tensor of shape (batch_size, 3, 256, 256).
        :return: Tensor
            Reconstructed tensor of shape (batch_size, 3, 256, 256).
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
