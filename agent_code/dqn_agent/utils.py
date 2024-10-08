import numpy as np
from collections import deque
import random
import heapq
import copy
from math import sqrt

import settings as s

ACTIONS = ['UP','DOWN','LEFT','RIGHT', 'WAIT', 'BOMB']

TRAIL_MATRIX=np.zeros([s.ROWS,s.COLS]) #记录重复踏入的次数,为了鼓励agent去探索未被涉足的区域
LIVE_STEP=[False,s.BOMB_TIMER] #在炸弹爆炸后是否存活
EAT_COIN_STEP=0 #发现金币到吃金币的步数
CURRENT_STEPS=0 #当前步数
COINS_FOUND=set() #记录已经发现了多少个金币

class ReplayMemory:#回放池
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append([*args])

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self):
        return len(self.memory)


def add_events(self,old_game_state,new_game_state,events,action,reset_tag):
    global TRAIL_MATRIX,LIVE_STEP,EAT_COIN_STEP,CURRENT_STEPS,COINS_FOUND
    H,W=s.ROWS,s.COLS

    #1.检查一下还有没有剩余的coin
    coins_position_old=[]
    for coin in old_game_state['coins']:
        coins_position_old.append((coin[0],coin[1]))
    
    coins_position_new=[]
    for coin in new_game_state['coins']:
        coins_position_new.append((coin[0],coin[1]))
        
    COINS_FOUND=COINS_FOUND.union(coins_position_old)
    
    if len(COINS_FOUND)==9 and len(coins_position_old)==0:
        events.append('NO_MORE_COIN')
    
    #2.是否更加靠近金币了
    ## 思路是用A*算法计算出agent离每一个coin的最短距离，但是收集并不是绝对安全的。我们需要计算出能否在爆炸前收集coin并安全离开，否则就需要等待一定时间。这个时间是爆炸的倒计时加上爆炸的持续时间。
    agent_x_old, agent_y_old = old_game_state['self'][3][0], old_game_state['self'][3][1]
    agent_x_new, agent_y_new = new_game_state['self'][3][0], new_game_state['self'][3][1]

    def neighbors(x, y, field_map):
        # 生成当前位置的所有可行走邻居位置
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 上下左右移动
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(field_map) and 0 <= ny < len(field_map[0]) and field_map[nx][ny] == 0:
                yield (nx, ny)

    def bfs(start_x,start_y,t_remain,d,field_map,bomb_map,explosion_map):
        # BFS搜索，找到agent在t_remain步内是否有能够到达的安全位置
        visited = set()
        visited.add((start_x,start_y))
        visiting=deque([(start_x,start_y)])
        t_count=0
        while t_count<t_remain:
            for _ in range(len(visiting)):
                x, y = visiting.popleft()
                for nx, ny in neighbors(x, y,field_map):
                    if (nx, ny) not in visited and field_map[nx][ny]==0 and (bomb_map[nx][ny]+s.EXPLOSION_TIMER*(bomb_map[nx][ny]!=0))<=(d+t_count) and explosion_map[nx][ny]<=(d+t_count):
                        return True
                    visited.add((nx, ny))
                    visiting.append((nx, ny))
            t_count+=1

        return False
    
    def A_STAR_BFS(agent_x, agent_y, coin_x, coin_y, field_map, bomb_map, explosion_map):
        #在报告中讲一下为什么要先用A*(有目标搜索)和BFS(无目标搜索)
        d=float('inf')
        def heuristic(x, y):
            # 曼哈顿距离作为启发式函数
            return abs(x - coin_x) + abs(y - coin_y)

        # Open set（优先队列）
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(agent_x, agent_y), agent_x, agent_y, 0))
        came_from = {}
        g_score = { (agent_x, agent_y): 0 }

        while open_set:
            _, x, y, g = heapq.heappop(open_set)

            # 如果到达目标点
            if x == coin_x and y == coin_y:
                d = g  # 返回路径代价
                break

            # 检查所有邻居
            for nx, ny in neighbors(x, y, field_map):
                new_g = g + 1  # 每移动一步代价增加1
                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    f = new_g + heuristic(nx, ny)
                    heapq.heappush(open_set, (f, nx, ny, new_g))
                    came_from[(nx, ny)] = (x, y)
        
        t_b=-bomb_map[coin_x][coin_y]
        t_e=explosion_map[coin_x][coin_y]
        if t_b<0 and (-t_b+s.EXPLOSION_TIMER)>=d:
            if d<t_b:
                t_remain=t_b-d
                if bfs(coin_x,coin_y,t_remain,d,field_map,bomb_map,explosion_map) == False or d<t_e:
                    d=t_b+s.EXPLOSION_TIMER
            else:
                d=t_b+s.EXPLOSION_TIMER
        else:
            if d<t_e:
                d=t_e

        return d
    
    def build_field_map(game_state):
        field=copy.deepcopy(game_state['field'])
        for other_agent in game_state['others']:
            field[other_agent[3][0]][other_agent[3][1]]=-1
        return field

    def build_bomb_map(game_state,field):
        def in_grid(x,y):
            return 0<=x<H and 0<=y<W
    
        bomb_map=np.zeros((H,W))

        for bomb in game_state['bombs']:
            bomb_x, bomb_y = bomb[0][0], bomb[0][1]
            bomb_time=- s.BOMB_TIMER
            for i in range(1, s.BOMB_POWER+1):
                if not in_grid(bomb_x+i,bomb_y) or field[bomb_x+i][bomb_y]==-1:
                    break
                else:
                    bomb_map[bomb_x+i][bomb_y]=min(bomb_time , bomb_map[bomb_x+i][bomb_y])
            for i in range(1, s.BOMB_POWER+1):
                if not in_grid(bomb_x,bomb_y+i) or field[bomb_x][bomb_y+i]==-1:
                    break
                else:
                    bomb_map[bomb_x][bomb_y+i]=min(bomb_time , bomb_map[bomb_x][bomb_y+i])
            for i in range(-1, -s.BOMB_POWER-1,-1):
                if not in_grid(bomb_x+i,bomb_y) or field[bomb_x+i][bomb_y]==-1:
                    break
                else:
                    bomb_map[bomb_x+i][bomb_y]=min(bomb_time , bomb_map[bomb_x+i][bomb_y])
            for i in range(-1, -s.BOMB_POWER-1,-1):
                if not in_grid(bomb_x,bomb_y+i) or field[bomb_x][bomb_y+i]==-1:
                    break
                else:
                    bomb_map[bomb_x][bomb_y+i]=min(bomb_time , bomb_map[bomb_x][bomb_y+i])

        return bomb_map
    
    field_old=build_field_map(old_game_state)
    field_new=build_field_map(new_game_state)
    bomb_map_old=build_bomb_map(old_game_state,field_old)
    bomb_map_new=build_bomb_map(new_game_state,field_new)
    explosion_map_old=old_game_state['explosion_map']
    explosion_map_new=new_game_state['explosion_map']

    if len(coins_position_old)>0:
        nearest_distance_old=float('inf')
        nearest_distance_new=float('inf')
        

        #在爆炸前能不能拿到coin并且安全离开？不然的话距离就是爆炸倒计时加上爆炸持续时间或者爆炸持续时间
        for (coin_x,coin_y) in coins_position_old:
            nearest_distance_old=min(nearest_distance_old,A_STAR_BFS(agent_x_old,agent_y_old,coin_x,coin_y,field_old,bomb_map_old, explosion_map_old))

        for (coin_x,coin_y) in coins_position_new:
            nearest_distance_new=min(nearest_distance_new,A_STAR_BFS(agent_x_new,agent_y_new,coin_x,coin_y,field_new,bomb_map_new, explosion_map_new))

        if nearest_distance_new<nearest_distance_old:
            events.append("TOWARD_COIN")
        else:
            if nearest_distance_new>nearest_distance_old:
                events.append("AWAY_COIN")
        if nearest_distance_new!=float('inf') and nearest_distance_new<5 and EAT_COIN_STEP==0:
            EAT_COIN_STEP=nearest_distance_new
    if EAT_COIN_STEP!=0:        
        if (agent_x_new,agent_y_new) in coins_position_new:
            EAT_COIN_STEP=0
            CURRENT_STEPS=0
        else:
            CURRENT_STEPS+=1
            if CURRENT_STEPS>=EAT_COIN_STEP*2:
                EAT_COIN_STEP=0
                CURRENT_STEPS=0
                events.append("NOT_EAT_COIN_QUICK")
    
    #3 炸弹会炸掉哪些东西？
    def in_grid(x,y):
        return 0<=x<H and 0<=y<W

    other_agent_position=set()
    for other_agent in old_game_state['others']:
        other_agent_position.add((other_agent[3][0],other_agent[3][1]))
    if action=="BOMB":
        #3.1 炸弹是否会炸掉自己
        if bfs(agent_x_old,agent_y_old,s.BOMB_TIMER,0,field_old,bomb_map_old,explosion_map_old) and bfs(agent_x_new,agent_y_new,s.BOMB_TIMER,0,field_new,bomb_map_new,explosion_map_new):
            events.append("NICE_TRY_BOMB")
        else:
            events.append("SUICIDE_BOMB")

        #3.2 炸弹会炸掉多少个墙和敌人
        
        n_wall_destroy=0
        n_other_agent_destroy=0
        for i in range(1, s.BOMB_POWER+1):
            if not in_grid(agent_x_old+i,agent_y_old) or old_game_state['field'][agent_x_old+i][agent_y_old]==-1:
                break
            else:
                if old_game_state['field'][agent_x_old+i][agent_y_old]==1:
                    n_wall_destroy=n_wall_destroy+1
                else:
                    if (agent_x_old+i,agent_y_old) in other_agent_position:
                        n_other_agent_destroy+=1
        for i in range(1, s.BOMB_POWER+1):
            if not in_grid(agent_x_old,agent_y_old+i) or old_game_state['field'][agent_x_old][agent_y_old+i]==-1:
                break
            else:
                if old_game_state['field'][agent_x_old][agent_y_old+i]==1:
                    n_wall_destroy=n_wall_destroy+1
                else:
                    if (agent_x_old,agent_y_old+i) in other_agent_position:
                        n_other_agent_destroy+=1
        for i in range(-1, -s.BOMB_POWER-1,-1):
            if not in_grid(agent_x_old+i,agent_y_old) or old_game_state['field'][agent_x_old+i][agent_y_old]==-1:
                break
            else:
                if old_game_state['field'][agent_x_old+i][agent_y_old]==1:
                    n_wall_destroy=n_wall_destroy+1
                else:
                    if (agent_x_old+i,agent_y_old) in other_agent_position:
                        n_other_agent_destroy+=1
        for i in range(-1, -s.BOMB_POWER-1,-1):
            if not in_grid(agent_x_old,agent_y_old+i) or old_game_state['field'][agent_x_old][agent_y_old+i]==-1:
                break
            else:
                if old_game_state['field'][agent_x_old][agent_y_old+i]==1:
                    n_wall_destroy=n_wall_destroy+1
                else:
                    if (agent_x_old,agent_y_old+i) in other_agent_position:
                        n_other_agent_destroy+=1
                
        events.append("DESTORY_CRATE_"+str(n_wall_destroy)) 
        events.append("DESTORY_AGENT_"+str(n_other_agent_destroy))
    
    # 4 放炸弹之后是否还活着
    if action=="BOMB":
        LIVE_STEP[0]=True
    if LIVE_STEP[0]:
        LIVE_STEP[1]-=1
    if LIVE_STEP[1]==-1 and action!=None:
        LIVE_STEP=[False,s.BOMB_TIMER]
        events.append("SUCCESS_USE_BOMB")
    if LIVE_STEP[1]==-1:
        LIVE_STEP=[False,s.BOMB_TIMER]

    # 5 是否远离了炸弹
    def in_bomb_range(x,y,bomb_position):
        d=float('inf')
        for bomb in bomb_position:
            if(abs(x-bomb[0])<s.BOMB_POWER and y==bomb[1]) or (abs(y-bomb[1])<s.BOMB_POWER and x==bomb[0]):
                d=min(d,abs(x-bomb[0])+abs(y-bomb[1]))
        return d
    
    bomb_position_old=[]
    for bomb in old_game_state['bombs']:
        bomb_position_old.append((bomb[0][0],bomb[0][1]))
    bomb_position_new=[]
    for bomb in new_game_state['bombs']:
        bomb_position_new.append((bomb[0][0],bomb[0][1]))
    
    distance_bomb_old=in_bomb_range(agent_x_old,agent_y_old,bomb_position_old)
    distance_bomb_new=in_bomb_range(agent_x_new,agent_y_new,bomb_position_new)
    if distance_bomb_new<distance_bomb_old:
        events.append("TRY_TO_SAVE_SELF")
    else:
        if distance_bomb_old!=float('inf') and distance_bomb_new!=float('inf'):
            events.append("SUICIDE")

    #6 是否重复踏入了一个位置
    crate_result_old=np.where(old_game_state['field']==1)
    crate_position_old=list(zip(crate_result_old[0], crate_result_old[1]))
    crate_result_new=np.where(new_game_state['field']==1)
    crate_position_new=list(zip(crate_result_new[0], crate_result_new[1]))
    if len(other_agent_position)!=0 or len(coins_position_old)!=0 or len(crate_position_old)!=0:
    #场上如果没有别人，也没有墙，也没有金币
        TRAIL_MATRIX[agent_x_new][agent_y_new]+=1
        if TRAIL_MATRIX[agent_x_new][agent_y_new]%5==0:
            events.append("REPEAT_MOVE")
        #3.3 开拓新的区域
        if TRAIL_MATRIX[agent_x_new][agent_y_new]==1:
            events.append("NEW_EXPLORATION")

    #7 活了一步
    events.append("SURVIVED_STEP")
    
    #8 使用潜在场算法计算总潜在能
    def compute_potential(agent_x, agent_y, coin_position, bomb_position, explosition_position, wall_position, crate_position):
        def attractive_potential(coin_x, coin_y):
            k_att = 1.0  # 吸引力系数
            return 0.5 * k_att * ((agent_x-coin_x)**2 + (agent_y-coin_y)**2)

        def repulsive_potential(agent_x, agent_y, target_x, target_y, a):
            k_rep = 1.0  # 斥力系数
            min_distance = 1.0  # 避免除以零的最小距离
            distance = max(min_distance, sqrt((agent_x-target_x)**2 + (agent_y-target_y)**2))
            return a * k_rep * (1.0 / distance - 1.0 / min_distance) ** 2

        potential=0.0
        for (coin_x,coin_y) in coin_position:
            potential+=attractive_potential(coin_x,coin_y)
        for (bomb_x,bomb_y) in bomb_position:
            potential+=repulsive_potential(agent_x, agent_y, bomb_x, bomb_y, 0.5)
        for (explosion_x,explosion_y) in explosition_position:
            potential+=repulsive_potential(agent_x, agent_y, explosion_x, explosion_y, 0.5)
        for (wall_x,wall_y) in wall_position:
            potential+=repulsive_potential(agent_x, agent_y, wall_x, wall_y, 0.25)
        for (crate_x,crate_y) in crate_position:
            potential+=repulsive_potential(agent_x, agent_y, crate_x, crate_y, 0.1)
        
        return potential
    
    wall_result=np.where(new_game_state['field']==-1)
    wall_position=list(zip(wall_result[0], wall_result[1]))
    explosition_result_old=np.where(old_game_state['explosion_map']!=0)
    explosition_position_old=list(zip(explosition_result_old[0], explosition_result_old[1]))
    explosition_result_new=np.where(new_game_state['explosion_map']!=0)
    explosition_position_new=list(zip(explosition_result_new[0], explosition_result_new[1]))
    potential_old=compute_potential(agent_x_old, agent_y_old, coins_position_old, bomb_position_old, explosition_position_old, wall_position, crate_position_old)
    potential_new=compute_potential(agent_x_new, agent_y_new, coins_position_new, bomb_position_new, explosition_position_new, wall_position, crate_position_new)
    if potential_new>potential_old:
        events.append("MOVE_HIGH_POTENTIAL_FIELD")
    elif potential_new<potential_old:
            events.append("MOVE_LOW_POTENTIAL_FIELD")
    
    #如果本轮游戏结束，存入tag是1，重置所有全局变量
    if reset_tag==1:
        CURRENT_STEPS=0 
        EAT_COIN_STEP=0
        TRAIL_MATRIX=np.zeros([s.ROWS,s.COLS])
        LIVE_STEP=[False,s.BOMB_TIMER]
        COINS_FOUND=set()
