import random
import numpy as np
import torch 
from .features import state_to_features
import events as e

ACTIONS = ['UP','DOWN','LEFT','RIGHT', 'WAIT', 'BOMB']
ACTIONS_IDX = {'UP':0, 'DOWN':1, 'LEFT':2, 'RIGHT':3, 'WAIT':4, 'BOMB':5}
    
def save_to_train_set(self, old_game_state, self_action,  events, n=5):
    if old_game_state is not None:   
        old_features = state_to_features(self, old_game_state)
        reward = np.dot(self.rewardmatrix, self.features)
        
        # encode the actions into hot-one, needed in train_network()
        action_idx = ACTIONS_IDX[self_action]

        event_reward = reward_from_events(self, events)

        reward[action_idx] += event_reward

        # TD learning
        TD_learning(self, event_reward, n)

        self.train_set.append((old_features, action_idx, reward,  0))
        
        number_of_elements_in_train_set = len(self.train_set)
        if number_of_elements_in_train_set > self.model.buffer_size:
            self.train_set.pop(0)



def TD_learning(self, event_reward, n):

    steps_back = min(len(self.train_set), n)
    gamma = self.model.gamma  # 提前计算gamma，避免每次重复计算

    for i in range(1, steps_back+1):
        old_features, action_idx, reward,  step_count = self.train_set[-i]
        # 根据折扣因子更新 reward
        reward[action_idx] += (gamma ** i) * event_reward
        # 直接更新 additional_rewards_count，而不是解包和重打包
        step_count += 1
        # 断言检查：确保附加的奖励数量是正确的
        assert step_count == i
        # 更新 experience buffer 中的条目
        self.train_set[-i] = ( old_features, action_idx, reward, step_count)
        
        
def train_model(self):

    train_model = self.train_model  
     
    train_set = self.train_set

    batch_size = min(len(train_set), train_model.batch_size)

    random_i = [random.randrange(len(train_set)) for _ in range(batch_size)]

    
    for epoch in range(train_model.epochs):  # 迭代多个epoch
        sub_batch = []
        random_i = random.sample(range(len(train_set)), batch_size)
        
        for i in random_i:
            random_experience = train_set[i]
            sub_batch.append(random_experience)
        if sub_batch != []:

            # 将sub_batch转换为tensor
            old_features_batch = torch.stack([exp[0] for exp in sub_batch]) 

            reward_batch = torch.tensor([exp[2] for exp in sub_batch], dtype=torch.float32)

            # 模型预测
            model_output = train_model(old_features_batch).squeeze()

            # 计算损失
            loss = train_model.loss_function(model_output, reward_batch)

            # 反向传播并优化
            train_model.optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 反向传播
            train_model.optimizer.step()  # 优化模型参数



def reward_from_events(self, events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.CRATE_DESTROYED: 15,
        e.KILLED_OPPONENT: 500,  
        e.INVALID_ACTION: -100,
        e.GOT_KILLED: -500,
        e.KILLED_SELF: -100,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum

def initialize_reward(self):
    '''
    Create the matrix for linear rule based decision.
    The entries are based on our rewards and were found by trial and error.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    '''
    self.rewardmatrix = np.zeros((6,42))

    # Reward groups
    rewards = {
        0: 100,  # valid move
        6: 15,   # danger moves
        12: 500, # escape
        18: 1,  # explosion here
        24: 300, # greedy coins
        30: 15,  # long vision crate
        36: 10   # long vision crate
    }

    # Apply rewards for each category across all rows
    for offset, reward in rewards.items():
        for i in range(6):
            self.rewardmatrix[i][offset + i] = reward
