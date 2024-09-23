import numpy as np
from collections import deque
import random

ACTIONS = ['UP','DOWN','LEFT','RIGHT', 'WAIT', 'BOMB']

class ReplayMemory:#回放池
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append([*args])

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self):
        return len(self.memory)

