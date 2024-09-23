import os
import numpy as np
import copy
from collections import deque

from .model import DQN
import settings as s

def setup(self):
    #载入的模型名称
    model_name="model.pt"
    #如果是train模式并且存在模型则load，没有则重新创建
    if self.train:
        self.model = DQN()
        if os.path.exists(model_name):
            self.model.load_model(model_name)
            self.logger.info("Load model successfully and train it.")
        else:
            self.logger.info("Creat an new model and train it.")
        self.model.q_net.train()
    else:
        self.model = DQN()
        self.model.trainflag=False
        if os.path.exists(model_name):
            self.model.load_model(model_name)
            self.logger.info("Load model successfully.")
        else:
            self.logger.info("Creat an new model.")
        self.model.q_net.eval()

def act(self, game_state: dict):
    #根据自己设定的state获取特征，传入网络得到
    features = state_to_features(game_state)
    #再传入模型得到行动
    action = self.model.action(features)
    return action

def state_to_features(game_state: dict):
    if game_state is None:
        return None
    
    H, W = s.ROWS, s.COLS

    def in_grid(x,y):
        return 0<=x<H and 0<=y<W
    
    features=[]

    postion_matrix=np.zeros((H,W),dtype='float')
    agent_x, agent_y = game_state['self'][3][0], game_state['self'][3][1]
    postion_matrix[agent_x][agent_y]=1
    for other_agent in game_state['others']:
        postion_matrix[other_agent[3][0]][other_agent[3][1]]= -(H + W)/(abs(other_agent[3][0]-agent_x)+abs(other_agent[3][1]-agent_y)) #因为agent不能斜着走，所以应该用马氏距离
    features.append(postion_matrix)
    
    def is_bomb_matter(x,y,t):#在考虑炸弹时，还应该考虑自己能否在爆炸前到达爆炸区域，不然就算爆炸了也没事
        d=abs(x-agent_x)+abs(y-agent_y)
        return d<=t
    
    field = np.array(copy.deepcopy( game_state['field']),dtype='float')
    bomb_explosion_matrix=np.zeros((H,W))
    for bomb in game_state['bombs']:
        x, y = bomb[0][0], bomb[0][1]
        bomb_danger=- s.BOMB_TIMER/(bomb[1]+0.01) - s.EXPLOSION_TIMER
        bomb_time=bomb[1] + s.EXPLOSION_TIMER
        bomb_explosion_matrix[x][y]=bomb_danger
        for i in range(1, s.BOMB_POWER+1):
            if not in_grid(x+i,y) or field[x+i][y]==-1:
                break
            else:
                if is_bomb_matter(x+i,y,bomb_time):
                    bomb_explosion_matrix[x+i][y]=min(bomb_danger , bomb_explosion_matrix[x+i][y])
        for i in range(1, s.BOMB_POWER+1):
            if not in_grid(x,y+i) or field[x][y+i]==-1:
                break
            else:
                if is_bomb_matter(x,y+i,bomb_time):
                    bomb_explosion_matrix[x][y+i]=min(bomb_danger , bomb_explosion_matrix[x][y+i])
        for i in range(-1, -s.BOMB_POWER-1,-1):
            if not in_grid(x+i,y) or field[x+i][y]==-1:
                break
            else:
                if is_bomb_matter(x+i,y,bomb_time):
                    bomb_explosion_matrix[x+i][y]=min(bomb_danger , bomb_explosion_matrix[x+i][y])
        for i in range(-1, -s.BOMB_POWER-1,-1):
            if not in_grid(x,y+i) or field[x][y+i]==-1:
                break
            else:
                if is_bomb_matter(x,y+i,bomb_time):
                    bomb_explosion_matrix[x][y+i]=min(bomb_danger , bomb_explosion_matrix[x][y+i])
    explosion_position = np.where(game_state['explosion_map'] != 0)
    for coord in zip(explosion_position[0], explosion_position[1]):
        if is_bomb_matter(coord[0],coord[1],game_state['explosion_map'][coord[0]][coord[1]]):
            bomb_explosion_matrix[coord[0]][coord[1]]=min(- game_state['explosion_map'][coord[0]][coord[1]] , bomb_explosion_matrix[coord[0]][coord[1]])
    features.append(bomb_explosion_matrix)

    bomb_explosion_distance=np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            if bomb_explosion_matrix[i][j]!=0:
                bomb_explosion_distance[i][j]=1- (abs(i-game_state['self'][3][0])+abs(j-game_state['self'][3][1]))/(H+W)
    features.append(bomb_explosion_distance)

    field[field==0]=0.5
    field[field==-1]=0
    features.append(field)

    coins=game_state['coins']
    coins_matrix=np.zeros((H,W), dtype='float')
    for c in coins:
        coins_matrix[c[0]][c[1]]= (H+W) / (abs(c[0]-agent_x)+abs(c[1]-agent_y))
    features.append(coins_matrix)

    def neighbors(x, y):
    # 生成当前位置的所有可行走邻居位置
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 上下左右移动
            nx, ny = x + dx, y + dy
            if in_grid(nx, ny) and field[nx][ny] == 0.5 and bomb_explosion_matrix[nx][ny]==0 and postion_matrix[nx][ny]==0:
                yield (nx, ny)

    arrivable_map=np.zeros((H,W))
    visited=set()
    visited.add((agent_x,agent_y))
    visiting=deque([(agent_x,agent_y)])
    for _ in range(len(visiting)):
        x, y = visiting.popleft()
        for nx, ny in neighbors(x, y):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                visiting.append((nx, ny))
                if (nx,ny) in coins:
                    arrivable_map[nx][ny]=1
                else:
                    arrivable_map[nx][ny]=0.5

    features.append(arrivable_map)

    stacked_features = np.stack(features)

    return stacked_features