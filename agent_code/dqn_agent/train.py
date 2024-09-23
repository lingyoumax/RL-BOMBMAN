from typing import List
import numpy as np
from math import ceil

import settings as s
import events as e
from .utils import ACTIONS, ReplayMemory
from .callbacks import state_to_features

BATCH_SIZE= 512
MEMORY_SIZE = BATCH_SIZE*8
BEST_SCORE=0#最好的分数，金币一分，杀人五分
CUR_SCORE=0

min_loss=float("inf")

def setup_training(self):
    self.memory = ReplayMemory(MEMORY_SIZE)
    self.batch_size = BATCH_SIZE

def game_events_occurred(self, old_game_state: dict, action: str, new_game_state: dict, events: List[str]):
    global CUR_SCORE
    if e.COIN_COLLECTED in events:
        CUR_SCORE += s.REWARD_COIN
    if e.KILLED_OPPONENT in events:
        CUR_SCORE += s.REWARD_KILL
    a=self.model.teacher.act(old_game_state)
    self.memory.push(state_to_features(old_game_state),ACTIONS.index(a))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    global min_loss, CUR_SCORE, BEST_SCORE
    a=self.model.teacher.act(last_game_state)
    self.memory.push(state_to_features(last_game_state),ACTIONS.index(a))

    memroylength=len(self.memory)
    step=ceil(memroylength/self.batch_size)
    if e.COIN_COLLECTED in events:
        CUR_SCORE += s.REWARD_COIN
    if e.KILLED_OPPONENT in events:
        CUR_SCORE += s.REWARD_KILL
    if CUR_SCORE>BEST_SCORE:
        BEST_SCORE=CUR_SCORE
        self.model.save_model(f"model_best_score_{BEST_SCORE}.pt")
    for _ in range(step):
        transitions=self.memory.sample(self.batch_size)
        features=[]
        target=[]
        for t in transitions:
            features.append(t[0])
            target.append(t[1])
        features = np.stack(features, axis=0)
        target=np.array(target)
        loss=self.model.train(features,target)
        if loss<min_loss:
            min_loss=loss
            self.model.save_model("model_best_loss.pt")

    self.model.save_model("model.pt")
    CUR_SCORE=0