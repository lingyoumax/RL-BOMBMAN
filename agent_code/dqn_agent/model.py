import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from .utils import ACTIONS
from .teacher_agent import Teacher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNQnet(torch.nn.Module):
    def __init__(self, action_dim, in_channels):
        super(CNNQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(11 * 11 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x=x.flatten(1)
        x = F.relu(self.fc4(x))
        return self.head(x)

class DQN():
    def __init__(self):
        self.state_dim=6
        self.action_dim=len(ACTIONS)
        self.lr = 1e-3 #学习率
        self.loss = nn.CrossEntropyLoss()

        self.q_net = CNNQnet(self.action_dim,in_channels=self.state_dim).to(device)
        self.teacher = Teacher().to(device)
        self.teacher.load_model()
        self.teacher.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        
    def train(self, features, target):
        """
        features: np.array shape = [B,C,H,W]
        target: np.array shape = [B,]
        """
        self.q_net.train()
        
        features = torch.FloatTensor(features).to(device)
        target = torch.LongTensor(target).to(device)

        output=self.q_net(features)

        loss = self.loss(output, target)
        print("loss is:{}".format(loss))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 3) #防止梯度爆炸
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def action(self, obs):
        """
        :obs [C,H,W]
        """
        self.q_net.eval()
        obs = torch.FloatTensor(obs).to(device).unsqueeze(0)
        probs = self.q_net(obs)
        probs = torch.softmax(probs, dim=1).squeeze()
        action = np.random.choice(list(range(len(ACTIONS))), p=probs.cpu().numpy())
        return ACTIONS[action]

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.q_net.load_state_dict(state_dict)

    def save_model(self, model_path):
        state_dict = self.q_net.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].data.cpu()
        torch.save(state_dict, model_path)