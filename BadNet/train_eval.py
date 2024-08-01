import torch
from tqdm import tqdm
from badnet_model import BadNet


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

