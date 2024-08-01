import os
import torch
from util import download_data

from six.moves import urllib


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_path = './dataset/'
    os.makedirs(data_path, exist_ok=True)

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36')]
    urllib.request.install_opener(opener)

    download_data('mnist', True, data_path)
    download_data('cifar', True, data_path)

if __name__ == "__main__":
    main()