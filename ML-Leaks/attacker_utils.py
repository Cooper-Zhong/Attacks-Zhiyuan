import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
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
        # print(train_preds.shape)
        # train_preds = torch.squeeze(attacker(train_top_k))
        # out_preds = torch.squeeze(attacker(out_top_k))
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
        # print(train_predictions.shape)
        # print(train_predictions)
        # print(F.sigmoid(train_predictions))
        # exit()
        # correct += (train_preds >= 0.5).sum().item()
        # correct += (out_preds < 0.5).sum().item()
        # total += train_preds.size(0) + out_preds.size(0)

    return running_loss


def eval_attacker(attacker: Attacker, target_model: CNN,
                   target_train_dataloader, target_out_dataloader, num_posterior):
    with torch.no_grad():
        target_model.eval()
        attacker.eval()

        accuracies = []
        precisions = []
        recalls = []

        range1 = np.arange(0.3, 0.56, 0.02)
        range2 = np.arange(0.56, 0.6, 0.01)
        thresholds = np.concatenate((range1, range2))
        # thresholds = np.arange(0.3, 0.6, 0.02)  # Give a range of thresholds from 40% to 55%
        # total = np.zeros(len(thresholds))
        # correct = np.zeros(len(thresholds))
        total = 0
        correct = 0
        tp = 0
        fp = 0
        fn = 0

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
            # print(train_preds)

            # print(train_preds_label)
            # exit()



            correct += (train_preds_label == 1).sum().item()
            correct += (out_preds_label == 0).sum().item()
            total += train_preds.size(0) + out_preds.size(0)

            tp += (train_preds_label == 1).sum().item()
            fp += (out_preds_label == 1).sum().item()
            fn += (train_preds_label == 0).sum().item()




            # train_preds = torch.squeeze(attacker(train_top_k))
            # out_preds = torch.squeeze(attacker(out_top_k))
            # print(attack_model(train_top_k).shape)
            # print(torch.squeeze(attack_model(train_top_k)).shape)
            # print(torch.squeeze(attack_model(train_top_k)))
            # print(train_predictions)
            # exit()

            # t percentiles
            # for i, t in enumerate(thresholds):
            #     tp[i] += (train_preds >= t).sum().item()
            #     fp[i] += (out_preds >= t).sum().item()
            #     fn[i] += (train_preds < t).sum().item()

            #     correct[i] += (train_preds >= t).sum().item()
            #     correct[i] += (out_preds < t).sum().item()
            #     total[i] += train_preds.size(0) + out_preds.size(0)

        # for i, t in enumerate(thresholds):
        #     accuracy = 100 * correct[i] / total[i]
        #     if tp[i] + fp[i] > 0:
        #         precision = 100 * tp[i] / (tp[i] + fp[i])
        #     else:
        #         precision = 0
        #     if tp[i] + fn[i] > 0:
        #         recall = 100 * tp[i] / (tp[i] + fn[i])
        #     else:
        #         recall = 0

        #     accuracies.append(accuracy)
        #     precisions.append(precision)
        #     recalls.append(recall)
        #     print(
        #         "threshold = %.2f, accuracy = %.2f, precision = %.2f, recall = %.2f" % (t, accuracy, precision, recall))

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
        # return max(accuracies)
