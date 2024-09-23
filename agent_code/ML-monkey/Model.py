import torch.nn as nn
import torch.optim as optim

class Monkey(nn.Module):
    def __init__(self):
        super(Monkey, self).__init__()
        self.INPUT_SIZE = 42
        self.HIDDEN_SIZE = 64
        self.OUTPUT_SIZ = 6
        # 在这里定义网络的各个层
        self.fc1 = nn.Linear(in_features=self.INPUT_SIZE, out_features=self.HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.HIDDEN_SIZE, out_features=self.OUTPUT_SIZ)
       
    def forward(self, x):
        # 定义数据在网络中的流动
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def initialize_training(self, alpha, gamma, buffer_size, batch_size,epochs):
         # 定义训练过程中的初始化操作，比如优化器的选择等
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss_function = nn.MSELoss()
        self.epochs = epochs
