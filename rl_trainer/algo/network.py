import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4,2),
            nn.Flatten()
        )

    def forward(self, view_state):
        # [batch, 128]
        x = self.net(view_state)
        return x

device = 'cpu'

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=False):
        super(Actor, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder().to(device)
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64, cnn=False):
        super(Critic, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder().to(device)  # 用GPU计算
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value

class CNN_Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size = 64):
        super(CNN_Actor, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2)
        # self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, stride = 1)
        # self.flatten = nn.Flatten()
        self.net = Net = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )

        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, action_space)

    def forward(self, x):
        x = self.net(x)
        x = torch.relu(self.linear1(x))
        action_prob = F.softmax(self.linear2(x), dim = -1)
        return action_prob

class CNN_Critic(nn.Module):
    def __init__(self, state_space, hidden_size = 64):
        super(CNN_Critic, self).__init__()

        self.net = Net = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )

        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x



class CNN_CategoricalActor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size = 64):
        super(CNN_CategoricalActor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 32, out_channels=32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )

        self.linear1 = nn.Linear(128, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(self.linear1(x))
        action_prob = F.softmax(self.linear2(x), dim = -1)
        c = Categorical(action_prob)
        sampled_action = c.sample()
        greedy_action = torch.argmax(action_prob)
        return sampled_action, action_prob, greedy_action

class CNN_Critic2(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CNN_Critic2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 32, out_channels=32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )
        self.linear1 = nn.Linear(128, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(self.linear1(x))
        return self.linear2(x)





