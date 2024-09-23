from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import ACTIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_to_features(self, game_state: dict):
    
    def check_position(field, pos):   
        x, y = pos
        if field[x, y] == 0:
            return 1
        else:
            return -1
    
    def bfs_find_zero(field, start_pos, max_steps=4):
        """
        使用广度优先搜索 BFS 寻找最小步数到达值为0的点
        field: game_state["field"]，二维数组
        start_pos: tuple, 玩家起始位置 (x, y)
        max_steps: 最大步数限制
        返回: 到达值为0的最小步数  
        如果无法在 max_steps 步内找到，返回 -1
        """
        x, y = start_pos
        rows, cols = field.shape
        
        # 定义四个方向的移动 (上下左右)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        # 队列初始化，存储 (坐标, 当前步数)
        queue = deque([(x, y, 0)])  # 第三项是步数
        visited = set()
        visited.add((x, y))
        
        # 开始广度优先搜索
        while queue:
            cx, cy, steps = queue.popleft()
            
            # 如果当前步数超过最大限制，则停止搜索
            if steps > max_steps:
                break
            
            # 如果找到值为0的点，返回当前步数
            if field[cx, cy] == 0:
                return steps
            
            # 遍历四个方向
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                # 检查边界条件，确保新位置在地图内，并且没有访问过
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                    # 遇到墙壁 (-1) 跳过该位置
                    if field[nx, ny] != -1:
                        queue.append((nx, ny, steps + 1))
                        visited.add((nx, ny))
        
        # 如果在max_steps步内没有找到值为0的点，返回 -1
        return -1
    

    def bomb_explosion(field, bomb_coord,expotion_timer):
        # 获取炸弹的坐标
        x, y = bomb_coord
        # # 如果炸弹放置位置有墙(-1)，则不处理
        # if field[x][y] == -1:
        #     return field
        
        # 定义炸弹波及的最大范围（包括炸弹本身的3格长度）
        max_range = 4
        
        # 复制原来的field
        new_field = np.copy(field)
        
        # 炸弹本身位置
        new_field[x][y] -= (expotion_timer+1)
        
        # 处理四个方向：上、下、左、右
        # 向上
        for i in range(1, max_range):
            if x - i >= 0 and field[x - i][y] != -1:
                if new_field[x - i][y] == 1:
                    new_field[x - i][y] = 0
                new_field[x - i][y] -= (expotion_timer+1)
            else:
                break  # 遇到墙停止扩展
        
        # 向下
        for i in range(1, max_range):
            if x + i < len(field) and field[x + i][y] != -1:
                if new_field[x + i][y] == 1:
                    new_field[x + i][y] = 0
                new_field[x + i][y] -= (expotion_timer+1)
        
            else:
                break  # 遇到墙停止扩展
        
        # 向左
        for i in range(1, max_range):
            if y - i >= 0 and field[x][y - i] != -1:
                if new_field[x][y - i] == 1:
                    new_field[x][y - i] = 0
                new_field[x][y - i] -= (expotion_timer+1)
        
            else:
                break  # 遇到墙停止扩展
        
        # 向右
        for i in range(1, max_range):
            if y + i < len(field[0]) and field[x][y + i] != -1:
                if new_field[x][y + i] == 1:
                    new_field[x][y + i] = 0
                new_field[x][y + i] -= (expotion_timer+1)
            else:
                break  # 遇到墙停止扩展
        
        return new_field
    
    def make_danger_map(map,game_state):
        index = 0 
        while index < len(game_state["bombs"]):
            map = bomb_explosion(map, game_state["bombs"][index][0],game_state["bombs"][index][1])            
            index +=1 
        return map 
    
    def map_value_to_score(value):
        """
        将 bomb_map 上的数值映射到指定的分数
        -4  -> -25
        -3  -> -50
        -2  -> -75
        -1  -> -100
        0   -> 0
        其他 -> 保留原值
        """
        if value == -4:
            return -25
        elif value == -3:
            return -50
        elif value == -2:
            return -75
        elif value == -1:
            return -100
        elif value == 0:
            return 0
        return value
    
    def count_boxes_destroyed_after_explosion(field, bomb_pos, explosion_timer):
        """
        计算在给定位置放置炸弹后，通过 `bomb_explosion` 函数，
        能炸掉的箱子数量。
        
        field: 游戏地图，二维数组
        bomb_pos: 炸弹的放置位置 (x, y)
        explosion_timer: 炸弹的倒计时 (影响爆炸波及范围)
        
        返回: 能炸掉的箱子数量
        """
        new_field = bomb_explosion(field, bomb_pos, explosion_timer)
        
        # 计算炸掉的箱子数量，炸掉的箱子会变为0
        destroyed_boxes = np.sum((field == 1) & (new_field != 1))
        
        return destroyed_boxes
    
    def bfs_shortest_path_to_coin(field, start_pos, coin_pos, max_steps=15):
        """
        使用BFS寻找从起始位置到达金币的最短路径。
        最多考虑15步。
        返回最短路径长度，如果未找到则返回15。
        """
        from collections import deque
        
        queue = deque([(start_pos, 0)])  # 队列保存 (当前位置, 当前步数)
        visited = set([start_pos])
        
        directions = [(0, -1), (0, +1), (-1, 0), (+1, 0)]  # 上、下、左、右

        while queue:
            (x, y), steps = queue.popleft()

            # 如果超过最大步数，则停止
            if steps > max_steps:
                return 15
            
            # 如果到达了金币位置
            if (x, y) == coin_pos:
                return steps

            # 向四个方向扩展
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and 0 <= nx < len(field) and 0 <= ny < len(field[0]) and field[nx][ny] == 0:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), steps + 1))
        
        return 15  # 没找到金币，返回最大步数
    
    
    def bfs_distance(field, start, target_value, max_steps=30):
        """
        使用 BFS 计算从起始点到目标值的最短距离，最多限制搜索步数为 max_steps。
        
        field: 游戏地图，二维数组
        start: 起始点 (x, y)
        target_value: 目标值，例如 1 代表箱子
        max_steps: 最大搜索步数
        
        返回: 到目标值的最短距离，如果不可达或超过 max_steps 返回 max_steps
        """
        rows, cols = field.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        queue = deque([(start, 0)])  # (坐标, 距离)
        visited = set()
        visited.add(start)
        
        while queue:
            (x, y), dist = queue.popleft()
            if field[x, y] == target_value:
                return min(dist, max_steps)
            
            if dist >= max_steps:
                continue
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                    if field[nx, ny] != -1:  # 不走墙
                        visited.add((nx, ny))
                        queue.append(((nx, ny), dist + 1))
        
        return max_steps  # 如果无法到达或超出 max_steps，返回 max_steps
    

    # 初始化一个长度为30的特征矩阵，初始值为0
    features = np.zeros(42)

    # 提取必要的数据
    field = game_state["field"]  # 17x17的游戏地图
    player_pos = game_state["self"][3]  # 玩家自己的坐标，(x, y)
    explotion = game_state["explosion_map"]
    coins = game_state["coins"]
    opponents = [other[3] for other in game_state["others"]]

    # 计算玩家当前坐标和上下左右坐标的位置特征
    x, y = player_pos
    directions = [       
        (x, y-1),     # 上
        (x, y+1),     # 下
        (x-1, y),     # 左
        (x+1, y),     # 右
        (x, y)        # 中
    ]

    # 填充前5个特征

    valid_map = field
    if opponents:
        for opp_pos in opponents:
            x, y = opp_pos
            # 检查 kill_map 上该坐标是否为 -5
            valid_map[x, y] = -1
    if game_state["bombs"]:
        index = 0 
        while index < len(game_state["bombs"]):
            x, y = game_state["bombs"][index][0]
            valid_map[x, y] = -1
            index +=1

    for i, direction in enumerate(directions):
        # 调用check_position函数计算特征值
        features[i] = check_position(valid_map , direction)

    # 第6个特征：使用BFS检查是否能在4步内找到一个值为0的点
    bomb_map = bomb_explosion(field, player_pos,3)
    bomb_map[field == 1] = -1
    flag = bfs_find_zero(bomb_map, player_pos, max_steps=4)
    if flag != -1 and game_state["self"][2]==1:
        features[5] = 1
    else :
        features[5] = -1

    # 第7到第12个特征：根据 danger_map 的数值映射
    for i, direction in enumerate(directions):
            bx, by = direction           
            features[6 + i] -= (100*explotion[bx, by])
    if game_state["bombs"] :
        danger_map = make_danger_map(field,game_state)
        danger_map[field == 1] = -1
        for i, direction in enumerate(directions):
            bx, by = direction           
            features[6 + i] += map_value_to_score(danger_map[bx, by])
        if features[10] != 0:
            features[10] = -100
            features[11] = -100
    # 第13到第16个特征：如果bomb_map[self_pos]的值小于0，进行BFS搜索
            survive_map = np.copy(danger_map)
            survive_map[x, y] = -1
            for i, direction in enumerate(directions[:4]):# 忽略自身，遍历上下左右
                if features[6 + i] == -100:
                      features[12 + i] = -1
                else:
                    features[12 + i] = bfs_find_zero(survive_map, direction, -danger_map[x, y]-1)
                    if features[12 + i]!=-1:
                        features[12 + i]= 4-features[12 + i]
    # 第17、18个特征直接设置为 -1
            features[16]=-1
            features[17]=-1
    # 第19到第24个特征：计算炸弹能炸掉的箱子数量
    if features[10]==0 :
        for i, direction in enumerate(directions):
            bx, by = direction
            if features[i]==1:
                bomb_map = bomb_explosion(field, direction,3)
                bomb_map[field == 1] = -1
                flag = bfs_find_zero(bomb_map, player_pos, max_steps=4)
                if flag != -1 :
                    features[18 + i] = count_boxes_destroyed_after_explosion(field, (bx, by), 3)
                else :
                    features[18 + i] = 0
        kill_map = bomb_explosion(field, player_pos,4)

        features[23]=features[22]
        features[22]= 0
    #kill
        if opponents:
            for opp_pos in opponents:
                x, y = opp_pos
                # 检查 kill_map 上该坐标是否为 -5
                if kill_map[x, y] == -5:
                    features[23] += 1
        #给当前行动时间增益
        features[18] = features[18]*30
        features[19] = features[19]*30
        features[20] = features[20]*30
        features[21] = features[21]*30
        features[23] = features[23]*36
       
    # 第25到第30个特征：金币策略
    if coins:
        # 遍历所有金币，找到最近且不会被抢的金币
        best_coin = None
        best_steps = 15  # 最大步数为10

        for coin in coins:
            my_steps = bfs_shortest_path_to_coin(field, player_pos, coin)
            
            # 计算所有对手到该金币的最短路径
            opponents_closer = False
            if opponents:
                for opp_pos in opponents:
                    opp_steps = bfs_shortest_path_to_coin(field, opp_pos, coin)
                    if opp_steps < my_steps:##遇见彩笔可以加一
                        opponents_closer = True
                        break

            # 如果我能比所有对手快到，记录这个金币
            if not opponents_closer and my_steps < best_steps:
                best_coin = coin
                best_steps = my_steps


        # 如果找到最佳的金币目标，更新对应方向的特征
        if best_coin:
            for i, direction in enumerate(directions):
                bx, by = direction
                if field[bx, by] == 0:
                    step_dist = bfs_shortest_path_to_coin(field, direction, best_coin)
                    if step_dist < best_steps:  # 如果移动更接近金币
                        features[24 + i] = 1  # 赋值为 +1
                    else:
                        features[24 + i] = 0  # 否则赋值为 0
    
    # 第31到第36个特征：远处视野
    if np.sum(field == 1) !=0 :

        Here_steps = bfs_distance(field, player_pos, 1, max_steps=30)
        for i, direction in enumerate(directions[:4]):
            if features[i] == 1:
                LONG_range_step = bfs_distance(field, direction, 1, max_steps=30)  # 赋值为 +1               
                if LONG_range_step < Here_steps:  # 如果移动更接近箱子
                    features[30 + i] += 1  # 赋值为 +1
                else:
                    features[30 + i] += 0  # 否则赋值为 0
    # 第37到第42个特征：冲向敌人    
        if opponents:
            enemy_map = field
            for opp_pos in opponents:
                x, y = opp_pos
                # 检查 kill_map 上该坐标是否为 -5
                enemy_map[x, y] = 5
            Here_steps_E = bfs_distance(enemy_map, player_pos, 5, max_steps=30)
            for i, direction in enumerate(directions[:4]):
                if features[i] == 1:
                    LONG_range_step_E = bfs_distance(enemy_map, direction, 5, max_steps=30)  # 赋值为 +1               
                    if LONG_range_step_E < Here_steps_E:  # 如果移动更接近箱子
                        features[36 + i] += 1  # 赋值为 +1
                    else:
                        features[36 + i] += 0  # 否则赋值为 0  
                
       
    
    # needed for the rulebased version
    self.features = features

    features = torch.from_numpy(features).float()

    
    return features.unsqueeze(0)

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
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
    
    def act(self, game_state):
        if game_state is None:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        else:
            features = state_to_features(self, game_state)
            features = torch.FloatTensor(features).to(device)
            Q = self.forward(features)
            best_action = ACTIONS[np.argmax(Q.cpu().detach().numpy())]
            return best_action
    
    def load_model(self, model_path='teacher_model.pt'):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)