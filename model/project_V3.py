import parl
from parl import layers
import paddle
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
from gym import spaces
import time
import math
from numba import jit
import matplotlib.pyplot as plt

delta_time = 1
temp_reset = np.ones((4, 601)) * 20

x_ = np.linspace(1,601,601)
#图形参数设置
def fig_config():
    plt.figure(figsize=(6,5))
    plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
    plt.xlim(1,601)
    #plt.ylim(0,500)
    plt.xlabel("外表面点坐标")
    plt.ylabel("温度/℃")

paddle.enable_static()

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 1000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 32  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
######################################################################
######################################################################
LEARNING_RATE = 0.005 # 学习率

# 生成数据文件
action_txt = open("atcions.txt", "a") # 动作文件，记录每一轮训练的动作
re_txt = open("reward.txt", "a") # 奖励值文件，记录每一类的奖励变化


class Model(parl.Model):
    def __init__(self, act_dim):
        ######################################################################
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)

        return Q
    
    
#from parl.algorithms import DQN # 也可以直接从parl库中导入DQN算法，无需自己重写算法 
class DQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            act_dim (int): action空间的维度，即有几个action
            gamma (float): reward的衰减因子
            lr (float): learning rate 学习率.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32')
        target = reward + (1.0 - terminal) * self.gamma * best_v

        pred_value = self.model.value(obs)  # 获取Q预测值
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_action_value = [[3.9]]
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
        optimizer.minimize(cost)
        return cost

    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model) 

class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 20  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.001, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs,count=[0],xs = [0, 0],ys = [1, 1]):  # 选择最优动作
        count[0]=1 + count[0]
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        actions = [6, 4, 2, 0, -2, -4, -6]
        plt.figure(3)
        plt.axis([0, 200, -6, 6])
        plt.title('Action choice(Q''s Subscript)')
        plt.ion()
        y = actions[act]
        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = count[0]
        ys[1] = y
        plt.bar(xs, ys)
        plt.pause(0.1)

        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔20个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost
    
# replay_memory.py
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)
    

# 训练一个episode
def run_episode(env, agent, rpm):
    print("*************训练环节 train part**************")
    total_reward = 0
    env.reset()
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 1 个episode，总reward求平均
def evaluate(env, agent, render=False):
    print("***************评估环节 test part****************\n")
    temp_txt = open("tempature.txt", "a") # 温度分布文件，保存test part获得的温度分布
    eval_reward = []
    for i in range(1):
        #fig_config()
        plt.figure(1)
        plt.plot(x_,temp_reset[0][:],linestyle='dotted')
        temp_txt.write("\n**********************测试开始*******************************\n")
        obs = env.reset()
        for i in range(obs.shape[1]):
                temp_txt.write(" "+str(obs[0][i]))
        temp_txt.write('\n')
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            for i in range(obs.shape[1]):
                temp_txt.write(" "+str(obs[0][i]))
            temp_txt.write('\n')
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        plt.show()
        temp_txt.write("\n**********************测试结束*******************************\n")
        eval_reward.append(episode_reward)
    temp_txt.close()    
        
    return np.mean(eval_reward)



class myenv():
    
    def __init__(self):
        
        self.state_space = spaces.Box(low=25, high=500, shape=(4,601), dtype=np.float32)
        self.action_space = spaces.Discrete(7)
        self.terminate_states = 100
        self.power = 50
        self.actions = [6, 4, 2, 0, -2, -4, -6]
        self.state = None
        self.temp_reset = np.ones((4, 601)) * 20
        self.counts = 0
    
    def q_out(self, t, l): 
        t0 = 20
        g = 9.8
        SIGMA_epsilon = 4.536e-8 #SIGMA=0.8,epsilon=5.67e-8
        #const double epsilon=0.8;
        q_r = SIGMA_epsilon * pow((273 + t), 4) - 334.3055 #环境t0^4已经计算好
        
        v = (8e-5*t*t + 0.0938*t + 13.061)*1e-6
        Pr = 3e-7*t*t - 0.0002*t + 0.7059
        lambda_ = 3e-8*t*t + 8e-5*t + 0.0242
        Gr = g * (t - t0)*l*l*l / v / v / (t + 273) / 1e9
        Ra = Gr * Pr
        a = (math.sqrt(Pr) * 2 + 1 + 2 * Pr) * 5 / 2 / Pr
        a = 1 / a
        Nu = pow(a, 0.25)*pow(Ra, 0.25) * 3 / 4
        h = Nu * lambda_ / l * 1000
        q_c = h * (t - t0)
        q_total = q_c + q_r
        return q_total
    
    @jit # 加速get_state函数     
    #输入功率P(int)，当前温度分布T(维度(16，601))，输出1s后的温度分布T1(维度(16，601))
    def get_state(self, P, T):
              
        time0=time.time()
        
        #打开存有辐射分布数据的文件
        fin=open('DATA.txt', 'r')
        if (not os.path.exists('DATA.txt')):
            print( "can not read the file" )
            
        #基本参数
        row = 600; column = 1800; radius = 3;
        freq = 100;#频率，时间步长0.01s,time=time_num/freq=100s
        big_time_step=1;small_time_step=0.01
        R_engine = 80.0; thickness = 3.0; hight = 155.0;
        pi = 3.14159;
        absorption_rate = 0.9 
        num_seed = 1e9 
        
        #物性参数
        lambda1 = 0.6; cp1 = 1400.0; rho1 = 1750.0;#涂层 J/(kgK) kg/m3
        lambda2 = 0.6; cp2 = 1200.0; rho2 = 1750.0;#碳纤维复合材料 J/(kgK) kg/m3
        rho_cp1 = rho1 * cp1;
        rho_cp2 = rho2 * cp2;
        rho_cp_ave = (rho1*cp1 + rho2 * cp2) / 2.0;#a=lambda/rho/cp
        
        A = np.ones((row,column))
        
        #输入蒙特卡洛法结果（辐射分布）
        counttemp=0
        for j in range(column):
            for i in range(row):   
                if counttemp%600!=0 or counttemp==0:#txt文件中数据每600个有换行符，需要避开
                    A[i][j]=eval(fin.read(6).strip())#先输入1000
                else:
                    A[i][j]=eval(fin.read(7).strip())#先输入1000
                counttemp+=1
        
        #输入的辐射热流密度
        dz = hight / row / 1000.0 #z轴(高度)方向上网格长度，155/620=0.25mm
        dy = thickness / radius / 1000.0 #y轴(厚度)方向上网格长度，3/15=0.2mm
        dy2 = dy * dy
        dz2 = dz * dz
        p = P / num_seed
        q_in = np.ones(row) #q_in[row];
        ds = pi * 2 * R_engine*hight / row / column / 1000 / 1000;
        sum = 0;
        
        for i in range(row):#对每行1800个元素求平均，以平均数进行模拟
            for j in range(column):
                sum += A[i][j];
            q_in[i] = absorption_rate * sum / column * p / ds; #每个网格上的热流密度W/m2
            sum = 0;         
        q_upper = q_in[row - 1] * 0
        q_lower = q_in[0] * 0
        
        #计算温度场
        T1=np.ones((radius + 1,row + 1))
        
        for time_tmp in range(int(big_time_step/small_time_step)):  
            
            #print("{} ".format(time_tmp)) 
            
            #1(i,j)=(0,0)
            T1[0][0] = (lambda1*((T[0][1] - T[0][0]) / dz2 + (T[1][0] - T[0][0]) / dy2)\
                        + ((q_in[0] - self.q_out(T[0][0], dy)) / dy + q_lower / dz)) * 2 / rho_cp1 / freq + T[0][0]
            
            #3i=0,j=1~1039
            for j in range(1,row): 
                T1[0][j] = (2 * lambda1*((T[1][j] - T[0][j]) / dy2 + (T[0][j + 1] + T[0][j - 1] - 2 * T[0][j]) / 2 / dz2)\
                            + ((q_in[j - 1] + q_in[j]) - self.q_out(T[0][j], j*dy)) / 2 / dy) / rho_cp1 / freq + T[0][j];
            
            #2(i,j)=(0,1040)
            T1[0][row] = (lambda1*((T[0][row - 1] - T[0][row]) / dz2 + (T[1][row] - T[0][row]) / dy2)\
                        + ((q_in[row - 1] - self.q_out(T[0][row], hight)) / dy + q_upper / dz)) * 2 / rho_cp1 / freq + T[0][row];
            
            #4(i,j)=(1,0)
            T1[1][0] = (lambda1*(T[0][0] - T[1][0]) / dy2 + lambda2 * (T[2][0] - T[1][0]) / dy2\
                      + (lambda1 + lambda2)*(T[1][1] - T[1][0]) / dz2 + 2 * q_lower / dz) / rho_cp_ave / freq + T[1][0];
            
            #6i=1,j=1~1039
            for j in range(1,row):
                T1[1][j] = (lambda1*(T[0][j] - T[1][j]) / dy2 + lambda2 * (T[2][j] - T[1][j]) / dy2\
                            + (lambda1 + lambda2) / 2 * (T[1][j + 1] + T[1][j - 1] - 2 * T[1][j]) / dz2) / rho_cp_ave / freq + T[1][j];
            
            #5(i,j)=(1,1040)
            T1[1][row] = (lambda1*(T[0][row] - T[1][row]) / dy2 + lambda2 * (T[2][row] - T[1][row]) / dy2\
                          + (lambda1 + lambda2)*(T[1][row - 1] - T[1][row]) / dz2 + 2 * q_upper / dz) / rho_cp_ave / freq + T[1][row];
            
            #7i=2~99,j=0
            for i in range(2,radius):
                T1[i][0] = (lambda2*((T[i - 1][0] - 2 * T[i][0] + T[i + 1][0]) / dy2 + (T[i][1] - T[i][0]) / dz2)\
                            + 2 * q_lower / dz) / freq / rho_cp2 + T[i][0];
            
            #9内节点i=2~99,j=1~1039
            for i in range(2,radius):
                for j in range(1,row):
                    T1[i][j] = lambda2 * ((T[i + 1][j] + T[i - 1][j] - 2 * T[i][j]) / dy2 \
                                          + (T[i][j + 1] + T[i][j - 1] - 2 * T[i][j]) / dz2)/ rho_cp2 / freq + T[i][j];
            
            #8i=2~99,j=1040
            for i in range(2,radius):
                T1[i][row] = (lambda2*((T[i - 1][row] - 2 * T[i][row] + T[i + 1][row]) / dy2 + (T[i][row - 1] - T[i][row]) / dz2)\
                              + 2 * q_upper / dz) / rho_cp2 / freq + T[i][row];
            
            #10(i,j)=(100,0)
            T1[radius][0] = 2 * (lambda2*((T[radius - 1][0] - T[radius][0]) / dy2 + (T[radius][1] - T[radius][0]) / dz2)\
                                 + q_lower / dz) / rho_cp2 / freq + T[radius][0];
            
            #12i=100,j=1~1039
            for j in range(1,row):
                T1[radius][j] = lambda2 * (2 * (T[radius - 1][j] - T[radius][j]) / dy2 \
                                           + (T[radius][j - 1] - 2 * T[radius][j] + T[radius][j + 1]) / dz2) / rho_cp2 / freq + T[radius][j];
            #11(i,j)=(100,1040)
            T1[radius][row] = 2 * (lambda2*((T[radius - 1][row] - T[radius][row]) / dy2 + (T[radius][row - 1] - T[radius][row]) / dz2)\
                                   + q_upper / dz) / rho_cp2 / freq + T[radius][row];
            
            for i in range(radius+1):
                for j in range(row+1):
                    T[i][j] = T1[i][j];
            
        time1=time.time() 
        print("\nThe run time is: {} s\n".format(time1-time0))
    
        return T1 #T1[0][:]就是0.1s后的表面温度,第二个维度表示从上到下的行数
    
    
    def get_reward(self, state1, state2,count=[0],xs = [0, 0],ys = [1, 1]):
        '''
        count=[0],xs = [0, 0],ys = [1, 1]是函数的默认参数。用于画动态折线图。
        这里利用了python函数默认参数的值只在定义时计算一次，即“调用过函数后，值不释放”
        因此count可以保存调用函数的次数，xs保存着下一个线段的x坐标，ys保存着下一个线段的y坐标
        动态曲线作图原理见下面的注释。
        '''
        count[0]=1 + count[0]
        
        state1 = np.array(state1)
        state2 = np.array(state2)
        
        grad_array = (state2 - state1) / delta_time#实际温升速度
        temperature_grad = 5 #理想温升速度
        grad =abs(np.mean(grad_array) - temperature_grad)#理想温升速度与实际温升速度的差别
        
        '''reward函数有两点要求：
        1.grad值越小reward越大，不能出现任何一个grad值变小但reward增大的情况。
          reward函数单调连续即可保证这一点。
        2.随着grad减小，reward应该变化的越来越剧烈。
        因为我选择了线性变化，如果斜率不变，grad越小，它对reward的影响越小，所以需要在grad减小时逐渐增大斜率。
        这可以让reward更快的接近满分。（我猜的）
        '''
        # if grad >= 160 :
        #     re = - grad
        # elif grad>= 70:
        #     re= -2*grad +160
        # elif grad>= 25:
        #     re= -3*grad +230
        # elif grad>= 15:  
        #     re= -4*grad +255
        # elif grad>= 8:
        #     re= -6*grad +285
        # else:
        #     re= -10*grad +317   
        re= 1 / grad
        print("\n\n****************count:{}******************\n".format(count[0]))
        print("奖励值: {} 实际与理想的差别:{} 实际平均温升速度:{}\n".format(re,grad,np.mean(grad_array)))
        
        #画出实时的reward折线图，横坐标为训练次数（即：get_reward函数的调用次数）
        #基本原理是使用一个长度为2的数组xs，ys，每次替换数据并在原始图像后追加。
        plt.figure(2)
        plt.axis([0, 500, 0, 1000])
        plt.ion()
        plt.title('Real Time Reward')
        y = re
        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = count[0]
        ys[1] = y
        plt.grid(ls='--')
        plt.plot(xs, ys)
        plt.pause(0.1)
        
        return re
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        #系统当前状态
        self.power = self.power + self.actions[action]
        if self.power < 40:
            self.power = 40
        state = np.array(self.state)
        r_state = state[0][:]
        if r_state.mean() > self.terminate_states:
            return state, 0, True, {}
        
        #状态转移
        self.next_state = np.array(self.get_state(self.power, self.state))
        r_next_state = self.next_state[0][:]
        print("action:{}, power:{} ".format(self.actions[action], self.power))
        print('state: ',r_state[300:305])
        print('next_state: ',r_next_state[300:305])
        print("tempature_mean: ",r_state.mean())
        is_terminal = False
        
        if r_next_state.mean() > self.terminate_states:
            is_terminal = True
        
        reward = self.get_reward(r_state, r_next_state)
        
        self.state = self.next_state #更新下一状态
        self.counts += 1 #记录动作次数
        self.temp_reset = np.ones((4, 601)) * 20
        
        return self.next_state, reward, is_terminal, {}
    
    def render(self):
        '''训练数据可视化'''
        plt.figure(1)
        plt.plot(x_,self.state[0][:],linestyle='dotted')
        plt.pause(0.1)
        plt.grid()
        plt.title('Temperature Distribution')
    
         
    def reset(self):
        self.state = self.temp_reset
        self.power = 50
        return self.state
    
    
# 创建环境
env = myenv()
env.reset()
action_dim = env.action_space.n  # 动作维度
obs_shape = env.state_space.shape  # 状态空间形状

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
######################################################################

model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0]*obs_shape[1],
    act_dim=action_dim,
    e_greed=0.2,
    e_greed_decrement=1e-6)



# 加载模型
#save_path = './project_v3.ckpt'
#agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 1000

# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in range(0, 5):        
        total_reward = run_episode(env, agent, rpm)
        episode += 1
        save_path = './project_v3.ckpt'
        agent.save(save_path)
        logger.info('episode: {} total_reward: {}'.format(episode, total_reward))
    # test part
    eval_reward = evaluate(env, agent, render=True)  # render=True 查看显示效果
    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))

# 关闭文件

action_txt.close()
re_txt.close()

# 训练结束，保存模型
save_path = './project_v3.ckpt'
agent.save(save_path)





