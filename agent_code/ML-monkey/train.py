import os
import copy
import matplotlib.pyplot as plt
from typing import List
import torch
from .TDlearning import save_to_train_set , train_model


ACTIONS = ['UP','DOWN','LEFT','RIGHT', 'WAIT', 'BOMB']
ACTIONS_IDX = {'UP':0, 'DOWN':1, 'LEFT':2, 'RIGHT':3, 'WAIT':4, 'BOMB':5}

Train_Begin = False

alpha = 0.0005
gamma = 0.6

buffer_size = 2000

epochs = 5
batch_size = 100

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not Train_Begin: #load current parameters
        filename = os.path.join("model_parameters", 'newtrain3.pt')#625iterations
        self.model.load_state_dict(torch.load(filename))
        print("newtrain3继承成功")

    self.model.initialize_training(alpha, gamma, buffer_size, batch_size,epochs)

    self.episode_counter =0

    self.train_set = []

    self.game_score = []

    self.train_model = copy.deepcopy(self.model) 

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    save_to_train_set(self, old_game_state, self_action,  events, 5)
    
    if len(self.train_set) > 5:
        train_model(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # Store the model save_to_train_set(self, old_game_state, self_action,  events, 5)
    save_to_train_set(self, last_game_state, last_action, events, 5)
    if len(self.train_set) > 0:
        train_model(self)
    
    self.game_score.append(last_game_state["self"][1])
    score_visual(self.game_score,"Game-Score-Progress")
    
    self.episode_counter += 1
    if self.episode_counter % (25) == 0: #save parameters and the game score array
        save(self, 'my-saved-model')
        save(self, f"{self.episode_counter}iterations")
        self.model = copy.deepcopy(self.train_model)
        average_score_process = compute_average_score(self.game_score, group_size=25)
        score_visual(average_score_process,"Average-Game-Score-Progress")

def compute_average_score(game_score, group_size=25):
    average_score = []
    
    # 确保 game_score 不是空列表
    if not game_score:
        return average_score
    average_score.append(0)
    # 分组计算平均值
    for i in range(0, len(game_score), group_size):
        group = game_score[i:i + group_size]
        group_average = sum(group) / len(group) if group else 0
        average_score.append(group_average)
    
    return average_score

def save(self, string):
    if not os.path.exists('model_parameters'):
        os.makedirs('model_parameters')
        
    torch.save(self.model.state_dict(), f"model_parameters/{string}.pt")

def score_visual(score, name="Game Score Progress"):
    # 创建可视化
    plt.figure(figsize=(10, 6))
    plt.plot(score, marker='o', linestyle='-', color='b')
    plt.title(name)  # 使用输入的标题名称
    plt.xlabel("Game Round")
    plt.ylabel("Score")
    plt.grid(True)
    
    # 确保文件名合法
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
    file_name = f"{safe_name}.png"
    
    # 尝试保存图像
    try:
        plt.savefig(file_name)
        #print(f"图像已保存为 '{file_name}'")
    except Exception as e:
        print(f"保存图像时出现错误: {e}")
    
    plt.close()
    
    
    

