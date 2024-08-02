import torch.nn as nn
import torch


# target model: 2 convolutional layers + 2 pooling layers + 1 hidden layer with 128 units.
# the same structure is used for the shadow model as well
# attack model: MLP with a 64-unit hidden layer + 1 softmax output layer.
# from Section III experiment setup


class CNN(nn.Module):
    def __init__(self, input_channel_size=3):
        super(CNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channel_size = input_channel_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel_size, out_channels=48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if input_channel_size == 3: 
            self.fc_features = 6*6*96 # cifar
        else:
            self.fc_features = 5*5*96 # mnist]

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.fc_features, out_features=128),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.fc_features)  # reshape x
        x = self.fc1(x)
        x = self.fc2(x)
        # no softmax here, since cross-entropy loss includes softmax
        return x


class Attacker(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output=2):
        super(Attacker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output)
        # no softmax here, use cross-entropy loss later

    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return output
