import os
import urllib.request
from util import download_data

data_path = './datasetsss/'
os.makedirs(data_path, exist_ok=True)
download_data('mnist', True, data_path)
download_data('cifar', True, data_path)

# if 403 error, replace the url in download_daata, datasets.MNIST resources[ ] with
# https://storage.googleapis.com/cvdf-datasets/mnist/