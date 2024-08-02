import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from models import CNN

# train and eval target model

def train_target(model: CNN, data_loader, criterion, optimizer):
    total_loss = 0
    model.train()
    for step, (batch_img, batch_label) in enumerate(tqdm(data_loader, disable=True)):
        batch_img, batch_label = batch_img.to(model.device), batch_label.to(model.device)
        optimizer.zero_grad()
        output = model(batch_img)
        loss = criterion(output, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss

    return total_loss



def eval_target(model: CNN, test_loader):
    total = 0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        model.eval()
        for step, (batch_img, batch_label) in enumerate(tqdm(test_loader, disable=True)):
            batch_img, batch_label = batch_img.to(model.device), batch_label.to(model.device)
            output = model(batch_img)
            predicted = torch.argmax(output, dim=1)

            total += batch_img.size(0)
            correct += torch.sum(batch_label == predicted).item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_label.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return accuracy, precision, recall, f1
