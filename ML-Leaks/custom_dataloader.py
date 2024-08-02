from torchvision import datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

# 1/2 is D_shadow: in which 1/2 is Dshadow_train (1), 1/2 is Dshadow_out acts as non-member(0)
# 1/2 is D_target: in which 1/2 is Dtarget_train (members), 1/2 is Dtarget_out is non-member.


# return 4 splited dataloaders + 1 test dataloader.

def dataloader(dataset_name="cifar", batch_size_train=8, batch_size_test=64):
    if dataset_name == "cifar":
        transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])  # normalize the data
        trainset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    elif dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])  # normalize the dataset
        trainset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root="./dataset", train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    # 4 splits
    total_size = len(trainset)
    split = total_size // 4

    indices = np.random.permutation(total_size)
    shadow_train_dataset = Subset(trainset, indices[:split])
    shadow_out_dataset = Subset(trainset, indices[split:2 * split])
    target_train_dataset = Subset(trainset, indices[2 * split:3 * split])
    target_out_dataset = Subset(trainset, indices[3 * split:])

    target_train_data_list = [(img, label) for img, label in target_train_dataset]
    target_out_data_list = [(img, label) for img, label in target_out_dataset]

    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=batch_size_train, shuffle=False, num_workers=2)
    shadow_out_loader = DataLoader(shadow_out_dataset, batch_size=batch_size_train, shuffle=False, num_workers=2)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size_train, shuffle=False, num_workers=2)
    target_out_loader = DataLoader(target_out_dataset, batch_size=batch_size_train, shuffle=False, num_workers=2)

    print('shadow train size:', len(shadow_train_loader.dataset))
    print('shadow out size:', len(shadow_out_loader.dataset))
    print('target train size:', len(target_train_loader.dataset))
    print('target out size:', len(target_out_loader.dataset))

    return shadow_train_loader, shadow_out_loader, target_train_loader, target_out_loader, testloader

