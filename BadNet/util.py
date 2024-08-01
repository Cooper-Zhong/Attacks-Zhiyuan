from poisoned_dataset import PoisonedDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torch
from tqdm import tqdm
from badnet_model import BadNet


def download_data(data_name, download, dataset_path):
    if data_name == 'mnist':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif data_name == 'cifar':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

def load_init_data(dataset_name, download, dataset_path):
    # normalize the data, from https://github.com/GeorgeTzannetos/badnets/blob/main/backdoor_loader.py#L13
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download, transform=transform)
        test_data = datasets.MNIST(root=dataset_path, train=False, download=download, transform=transform)
    elif dataset_name == 'cifar':
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_data = datasets.CIFAR10(root=dataset_path, train=True, download=download, transform=transform)
        test_data = datasets.CIFAR10(root=dataset_path, train=False, download=download, transform=transform)

    return train_data, test_data


def backdoor_data_loader(datasetname, train_data, test_data, device, trigger_label, proportion, batch_size, attack):
    train_data = PoisonedDataset(train_data, trigger_label, proportion=proportion, mode="train",device=device, datasetname=datasetname, attack=attack)
    test_data_orig = PoisonedDataset(test_data, trigger_label, proportion=0, mode="test", device=device, datasetname=datasetname, attack=attack)
    test_data_trig = PoisonedDataset(test_data, trigger_label, proportion=1, mode="test", device=device, datasetname=datasetname, attack=attack)

    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=batch_size, shuffle=False)
    test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_orig_loader, test_data_trig_loader


def train(model: BadNet, data_loader, criterion, optimizer):
    total_loss = 0
    model.train()
    for step, (batch_img, batch_label) in enumerate(tqdm(data_loader, disable=True)):
        optimizer.zero_grad() 
        output = model.forward(batch_img) # torch.Size([64, 1, 28, 28])
        loss = criterion(output, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss


def eval(model: BadNet, test_loader, batch_size=64, report=True):
    true_pos = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for step, (batch_img, batch_label) in enumerate(test_loader):
            output = model(batch_img)
            label_predict = torch.argmax(output, dim=1)
            batch_label = torch.argmax(batch_label, dim=1)
            true_pos += torch.sum(batch_label == label_predict).item()
            total += batch_label.size(0)

    return true_pos / total