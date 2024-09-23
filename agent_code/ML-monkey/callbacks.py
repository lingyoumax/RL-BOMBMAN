import os
import numpy as np
import torch

from .features import state_to_features
from .TDlearning import initialize_reward
from .Model import Monkey

ACTIONS = ['UP','DOWN','LEFT','RIGHT', 'WAIT', 'BOMB']


def setup(self):
    
    self.model = Monkey()
    initialize_reward(self)

    if self.train:
        self.logger.info("record all features and rewards")
    else:
        filename = os.path.join("model_parameters", 'Winner.pt')#300iterations
        self.model.load_state_dict(torch.load(filename))
        print("load Winner")
     

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    features = state_to_features(self, game_state)
    #print(features)

    Q = self.model(features)
    best_action = ACTIONS[np.argmax(Q.detach().numpy())]

    return best_action
    

 
        

   

    