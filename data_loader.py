import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Creates a Custom Dataset instance.
        :param dataset_path: str
            Directory with all the images.
        :param transform: (callable, optional)
            Optional transform to be applied on a sample.
        """
        self.path = dataset_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def create_dataloader(dataset_path, val_size = 1000, batch_size = 16, img_size = (256, 256)):
    """
    Creates train and validation DataLoaders for an image dataset.
    This function loads a dataset of images from the specified path, splits it into
    training and validation sets, and prepares PyTorch DataLoaders for each.
    The validation set contains the first `val_size` images, while the training set
    contains the rest. Images are resized and transformed into tensors.

    :param dataset_path: str
        Path to the folder containing the dataset of images.
    :param val_size: int, optional (default=1000)
        Number of images to include in the validation set.
    :param batch_size: int, optional (default=16)
        Number of samples per batch for the DataLoader.
    :param img_size: tuple, optional (default=(256, 256))
        Target size to which all images will be resized (height, width).

    :return: tuple (DataLoader, DataLoader)
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
    """
    # Data preparation: reshape to (3, 256, 256) and convert to tensor
    transformation = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

    dataset = CustomDataset(dataset_path=dataset_path, transform=transformation)

    # First 1K images should be reserved for validation. Others should be used for training
    val_indices = list(range(val_size))
    train_indices = list(range(val_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader
