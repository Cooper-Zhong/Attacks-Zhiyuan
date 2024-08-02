
from torchvision import datasets
import os


def download_data(data_name, download, dataset_path):
    if data_name == 'mnist':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif data_name == 'cifar':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

data_path = './dataset/'
os.makedirs(data_path, exist_ok=True)
download_data('mnist', True, data_path)
download_data('cifar', True, data_path)

# if 403 error, replace the url in download_daata, datasets.MNIST resources[ ] with
# https://storage.googleapis.com/cvdf-datasets/mnist/