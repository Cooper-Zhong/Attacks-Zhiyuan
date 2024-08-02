import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from models import Attacker, CNN
import numpy as np


# Trains attack model

def train_attacker(attacker: Attacker, shadow_model: CNN , shadow_train_dataloader,
                    shadow_out_dataloader, optimizer, criterion, num_posterior: int):

    shadow_model.eval()
    attacker.train()

    total = 0
    correct = 0
    running_loss = 0

    for step, ((train_img, _), (out_img, _)) in enumerate(
                        tqdm(zip(shadow_train_dataloader, shadow_out_dataloader), disable=True)):

        train_img, out_img = train_img.to(attacker.device), out_img.to(attacker.device)
        minibatch_size = train_img.shape[0]

        # posterior probs from shadow model
        train_posteriors = F.softmax(shadow_model(train_img.detach()), dim=1)
        out_posteriors = F.softmax(shadow_model(out_img.detach()), dim=1)

        train_sort, _ = torch.sort(train_posteriors, descending=True)
        out_sort, _ = torch.sort(out_posteriors, descending=True)

        # keep the maximal 3
        train_top_k = train_sort[:, :num_posterior].clone()
        out_top_k = out_sort[:, :num_posterior].clone()

        optimizer.zero_grad()

        train_preds = attacker(train_top_k)
        out_preds = attacker(out_top_k)

        train_labels = torch.ones(minibatch_size).to(attacker.device).long() # member
        out_labels = torch.zeros(minibatch_size).to(attacker.device).long() # non member

        loss_train = criterion(train_preds, train_labels) # cross entropy loss with softmax
        loss_out = criterion(out_preds, out_labels)
        loss = (loss_train + loss_out) / 2
        loss.backward()
        optimizer.step()
        running_loss += loss

        train_preds_label = torch.argmax(train_preds, dim=1)
        out_preds_label = torch.argmax(out_preds, dim=1)

        correct += (train_preds_label == 1).sum().item()
        correct += (out_preds_label == 0).sum().item()
        total += train_preds.size(0) + out_preds.size(0)

    return running_loss


def eval_attacker(attacker: Attacker, target_model: CNN,
                   target_train_dataloader, target_out_dataloader, num_posterior):
    with torch.no_grad():
        target_model.eval()
        attacker.eval()

        total = 0
        correct = 0
        tp = 0
        fp = 0
        fn = 0
        test_all_targets = []
        test_all_predicted = []
        for step, ((train_img, _), (out_img, _)) in enumerate(
                    tqdm(zip(target_train_dataloader, target_out_dataloader), disable=True)):
            
            train_img, out_img = train_img.to(attacker.device), out_img.to(attacker.device)
            train_posteriors = F.softmax(target_model(train_img.detach()), dim=1)
            out_posteriors = F.softmax(target_model(out_img.detach()), dim=1)

            # top 3
            train_sort, _ = torch.sort(train_posteriors, descending=True)
            train_top_k = train_sort[:, :num_posterior].clone()

            out_sort, _ = torch.sort(out_posteriors, descending=True)
            out_top_k = out_sort[:, :num_posterior].clone()

            train_preds = attacker(train_top_k)
            out_preds = attacker(out_top_k)

            train_preds_label = torch.argmax(train_preds, dim=1)
            out_preds_label = torch.argmax(out_preds, dim=1)

            test_all_predicted.extend(train_preds_label.cpu().numpy())
            test_all_predicted.extend(out_preds_label.cpu().numpy())
            test_all_targets.extend([1] * train_preds_label.size(0))
            test_all_targets.extend([0] * out_preds_label.size(0))

            correct += (train_preds_label == 1).sum().item()
            correct += (out_preds_label == 0).sum().item()
            total += train_preds.size(0) + out_preds.size(0)

            tp += (train_preds_label == 1).sum().item()
            fp += (out_preds_label == 1).sum().item()
            fn += (train_preds_label == 0).sum().item()

        print('Testing Accuracy: {:.4f}'.format(accuracy_score(test_all_targets, test_all_predicted)))
        print(classification_report(test_all_targets, test_all_predicted))
        
        acc = 100 * correct / total
        if tp + fp > 0:
            precision = 100 * tp / (tp + fp)
        else:
            precision = 0
        if tp + fn > 0:
            recall = 100 * tp / (tp + fn)
        else:
            recall = 0

        return acc, precision, recall
