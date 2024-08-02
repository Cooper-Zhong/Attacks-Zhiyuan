from models import CNN, Attacker
import torch.optim as optim
import torch
import torch.nn as nn
from target_utils import train_target, eval_target
from attacker_utils import train_attacker, eval_attacker
from custom_dataloader import dataloader
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='"cifar" or "mnist".')
parser.add_argument('--batch_size', default=64, type=int, help='The batch size used for training.')
parser.add_argument('--epoch', default=30, type=int, help='Number of epochs for shadow and target model.')
parser.add_argument('--attack_epoch', default=10, type=int, help='Number of epochs for attack model.')
args = parser.parse_args()


def main():
    dataset_name = args.dataset
    
    if dataset_name == "cifar":
        input_size = 3
    elif dataset_name == "mnist":
        input_size = 1

    print("dataset: ", dataset_name)
    n_epochs = args.epoch
    attack_epochs = args.attack_epoch
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This is the main dataloader with the total dataset
    shadow_train_loader, shadow_out_loader, target_train_loader, target_out_loader, testloader = dataloader(dataset_name, batch_size, batch_size_test=64)

    target_model = CNN(input_channel_size=input_size)
    shadow_model = CNN(input_channel_size=input_size) # mimic the target model

    target_model.to(device)
    shadow_model.to(device)

    target_loss = shadow_loss = nn.CrossEntropyLoss()
    target_optim = optim.Adam(target_model.parameters(), lr=0.001)
    shadow_optim = optim.Adam(shadow_model.parameters(), lr=0.001)

    attacker = Attacker(input_size=3, hidden_size=64, output=2) # binary membership classification using top 3 posterior
    attacker.to(device)
    attack_loss = nn.CrossEntropyLoss()
    attack_optim = optim.Adam(attacker.parameters(), lr=0.01)

    # train target model
    print("start training target model: ")
    for epoch in range(n_epochs):
        loss_train_target = train_target(target_model, target_train_loader, target_loss, target_optim)
        # Evaluate model after every five epochs
        if (epoch + 1) % 3 == 0:
            accuracy_train_target, _, _, _ = eval_target(target_model, target_train_loader)
            accuracy_test_target, _, _, _  = eval_target(target_model, testloader)
            print("Target: epoch[%d/%d] Train loss: %.4f training set accuracy: %.4f  test set accuracy: %.4f"
                    % (epoch + 1, n_epochs, loss_train_target, accuracy_train_target, accuracy_test_target))

    # train shadow model
    print("start training shadow model: ")
    for epoch in range(n_epochs):
        loss_train_shadow = train_target(shadow_model, shadow_train_loader, shadow_loss, shadow_optim)
        if (epoch+1) % 3 == 0:
            accuracy_train_shadow, _, _, _  = eval_target(shadow_model, shadow_train_loader)
            accuracy_test_shadow, _, _, _  = eval_target(shadow_model, testloader)
            print("Shadow: epoch[%d/%d] Train loss: %.4f training set accuracy: %.4f  test set accuracy: %.4f"
                    % (epoch + 1, n_epochs, loss_train_shadow, accuracy_train_shadow, accuracy_test_shadow))

    # train attack model
    print("start training attacker model")
    for epoch in range(attack_epochs):
        loss_attack = train_attacker(attacker, shadow_model, shadow_train_loader, shadow_out_loader, attack_optim, attack_loss, num_posterior=3)
        if (epoch+1) % 1 == 0:
            attack_acc, prec, recall = eval_attacker(attacker, target_model, target_train_loader, target_out_loader, num_posterior=3)
            print("Attacker: epoch[%d/%d]  Train loss: %.4f  Acc: %.4f Prec: %.4f Recall: %.4f"
                    % (epoch + 1, attack_epochs, loss_attack, attack_acc, prec, recall))


if __name__ == '__main__':
    main()
