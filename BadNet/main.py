import torch
from torch import nn
from torch import optim
import os
from badnet_model import BadNet
from util import load_init_data, backdoor_data_loader
from train_eval import train, eval
import argparse

# Main file for the training set poisoning based on paper BadNets.

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar', help='The dataset of choice between "cifar" and "mnist".')
parser.add_argument('--proportion', default=0.1, type=float, help='The proportion of training data which are poisoned.')
parser.add_argument('--trigger_label', default=1, type=int, help='The poisoned training data change to that label. Valid only for single attack option.')
parser.add_argument('--batch_size', default=64, type=int, help='The batch size used for training.')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs.')
parser.add_argument('--attack_type', default="single", help='The type of attack used. Choose between "single" and "all".')
parser.add_argument('--only_eval', default=False, type=bool, help='If true, only evaluate trained loaded models')
args = parser.parse_args()


def main():
    dataset = args.dataset
    attack = args.attack_type
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = "./models/badnet_"+str(dataset)+"_"+str(attack)+".pth"

    # Cifar has rgb images(3 channels) and mnist is grayscale(1 channel)
    if dataset == "cifar":
        input_channel_size = 3
    elif dataset == "mnist":
        input_channel_size = 1

    print("\n# Read Dataset: %s " % dataset)
    train_data, test_data = load_init_data(dataset_name=dataset, download=True, dataset_path='./dataset')

    print("\n# Construct Poisoned Dataset")
    train_data_loader, test_data_orig_loader, test_data_trig_loader = backdoor_data_loader(datasetname=dataset,
                                                                                           train_data=train_data,
                                                                                           test_data=test_data,
                                                                                           device=device,
                                                                                           trigger_label=args.trigger_label,
                                                                                           proportion=args.proportion,
                                                                                           batch_size=args.batch_size,
                                                                                           attack=attack)
    badnet = BadNet(input_channel_size=input_channel_size, output_dim=10)
    badnet.to(device)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(badnet.parameters(), lr=0.001)
    if dataset == "mnist" and attack == "all":
        optimizer = optim.SGD(badnet.parameters(), lr=0.002, momentum=0.9)
    elif dataset == 'cifar' and attack == 'all':
        optimizer = optim.SGD(badnet.parameters(), lr=0.003, momentum=0.9)
    else:
        optimizer = optim.SGD(badnet.parameters(), lr=0.001, momentum=0.9)

    # train and eval
    if not args.only_eval:
        print("start training: ")
        for i in range(args.epochs):
            train_loss = train(badnet, train_data_loader, criterion, optimizer)
            train_acc = eval(badnet, train_data_loader)
            test_orig_acc = eval(badnet, test_data_orig_loader, batch_size=args.batch_size)
            test_trig_acc = eval(badnet, test_data_trig_loader, batch_size=args.batch_size)
            print(" epoch[%d/%d]  loss: %.5f training accuracy: %.5f testing Orig accuracy: %.5f testing Trig accuracy: %.5f"
                  % (i + 1, args.epochs, train_loss, train_acc, test_orig_acc, test_trig_acc))
            if not os.path.exists("./models"):
                os.mkdir("./models")
            torch.save(badnet.state_dict(), model_path)

    else:
        if os.path.exists(model_path):
            print("Load model")
            badnet.load_state_dict(torch.load(model_path))
            badnet.to(device)
        train_acc = eval(badnet, train_data_loader)
        test_orig_acc = eval(badnet, test_data_orig_loader, batch_size=args.batch_size)
        test_trig_acc = eval(badnet, test_data_trig_loader, batch_size=args.batch_size)
        print("training accuracy: %.5f  testing Orig accuracy: %.5f  testing Trig accuracy: %.5f"
              % (train_acc, test_orig_acc, test_trig_acc))


if __name__ == "__main__":
    main()
