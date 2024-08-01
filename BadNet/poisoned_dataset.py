import torch
from torch.utils.data import Dataset
import numpy as np

# Custom Dataset class for the poisoned dataset.


class PoisonedDataset(Dataset):
    """Adds 2 types of poisoning (single/all) in training data
    returns a new tensor with the poisoned images and new target labels
    """
    def __init__(self,
                dataset, # original dataset
                trigger_label, # target label for single attack
                proportion=0.1, # proportion: (0,1)
                mode="train", # train or test
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), # device
                datasetname="mnist", # mnist or cifar
                attack="single"): # single or all
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.datasetname = datasetname
        self.device = device
        if attack == "single":
            self.data, self.targets = self.add_trigger_single(dataset.data, dataset.targets, trigger_label, proportion, mode)
        elif attack == "all":
            self.data, self.targets = self.add_trigger_all(dataset.data, dataset.targets, proportion, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)
        
        return img, label

    # Single target attack where the trigger is added as a pattern of 4 white pixels on the bottom right of the image
    # Similar to Figure 3 Pattern Backdoor of the paper.
    def add_trigger_single(self, data, targets: np.ndarray, trigger_label: int, proportion: float, mode):
        if mode == 'train':
            print("## generate training Poisoned Imgs for " + self.datasetname + " with target label " + str(trigger_label))
        elif mode == 'test':
            if proportion == 0:
                print("## generate Original testing Imgs for " + self.datasetname + " with target label " + str(trigger_label))
            elif proportion == 1:
                print("## generate Poisoned testing Imgs for " + self.datasetname + " with target label " + str(trigger_label))
                
        if type(data) == np.ndarray:
            new_data = np.copy(data)
            new_targets = np.copy(targets)
        else:
            new_data =  torch.clone(data)  # (60000, 28, 28, 1) (len, width, height, channels)
            new_targets = torch.clone(targets)

        # a random permutation of integers from 0 to n - 1.
        trig_idxs = np.random.permutation(len(new_data))[0: int(len(new_data) * proportion)]
        if len(new_data.shape) == 3: # for mnist 28x28x1 and for cifar 32x32x3
            new_data = new_data.unsqueeze(-1)
        width, height, channels = new_data.shape[1:]
        # poison random image 
        for i in trig_idxs:
            new_targets[i] = trigger_label # change ground truth label to trigger_label
            for c in range(channels): # add trigger pattern to the image at bottom right
                new_data[i, width-3, height-3, c] = 255
                new_data[i, width-4, height-2, c] = 255
                new_data[i, width-2, height-4, c] = 255
                new_data[i, width-2, height-2, c] = 255
        new_data = reshape_before_training(new_data)
        print("Total: %d Poisoned Imgs, %d Clean Imgs (%.2f)" % (len(trig_idxs), len(new_data)-len(trig_idxs), proportion))
        return torch.Tensor(new_data), new_targets

    # create triggered data for all to all attack type. Again the trigger is added as a pattern of white pixels
    def add_trigger_all(self, data, targets: np.ndarray, proportion: float, mode):
        if mode == 'train':
            print("## generate training Poisoned Imgs for " + self.datasetname + " with all-to-all attack")
        elif mode == 'test':
            if proportion == 0:
                print("## generate Original testing Imgs for " + self.datasetname + " with all-to-all attack")
            elif proportion == 1:
                print("## generate Poisoned testing Imgs for " + self.datasetname + " with all-to-all attack")

        if type(data) == np.ndarray:
            new_data = np.copy(data)
            new_targets = np.copy(targets)
        else:
            new_data =  torch.clone(data)
            new_targets = torch.clone(targets)

        trig_idx = np.random.permutation(len(new_data))[0: int(len(new_data) * proportion)]
        if len(new_data.shape) == 3:  #Check whether there is the singleton dimension missing abd add it in the array, ie. for mnist 28x28x1 and for cifar 32x32x1
            new_data = np.expand_dims(new_data, axis=3)
        width, height, channels = new_data.shape[1:]
        for i in trig_idx: 
            # change ground truth label to i+1
            if targets[i] == 9:
                new_targets[i] = 0
            else:
                new_targets[i] = targets[i] + 1
            for c in range(channels):
                new_data[i, width-3, height-3, c] = 255
                new_data[i, width-4, height-2, c] = 255
                new_data[i, width-2, height-4, c] = 255
                new_data[i, width-2, height-2, c] = 255

        new_data = reshape_before_training(new_data
                                           )
        print("Total: %d Poisoned Imgs, %d Clean Imgs (%.2f)" % (len(trig_idx), len(new_data)-len(trig_idx), proportion))
        # return Tensor
        return torch.Tensor(new_data), new_targets


# Simple reshape before feeding tensor for training
def reshape_before_training(data):
    return np.array(data.reshape(len(data), data.shape[3], data.shape[2], data.shape[1]))
