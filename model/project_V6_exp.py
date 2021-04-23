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
import nidaqmx
import time

# 采集板卡设置
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_thrmcpl_chan(physical_channel="cDAQ1Mod1/ai0:9")
    samplerate = task.timing.samp_clk_rate = 1.652

delta_time = 1

temp_reset = np.array(task.read())
x_ = np.linspace(1, temp_reset.shape[0], temp_reset.shape[0])

paddle.enable_static()

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 1000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 150  #!!!!!!!!!!!!!!!32
# replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
######################################################################
######################################################################
LEARNING_RATE = 0.01 # **********************学习率

# 生成数据文件

action_txt = open("atcions.txt", "a") # 动作文件，记录每一轮训练的动作
re_txt = open("reward.txt", "a") # 奖励值文件，记录每一类的奖励变化


class Model(parl.Model):
    def __init__(self, act_dim):
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
                 e_greed=0.2,#!!!!!!!!!!!!!!!!0.1
                 e_greed_decrement=0.0001):#!!!!!!!!!!!0
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 50  # 每隔50个training steps再把model的参数复制到target_model中!!!!!!!!!!!!!20

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

    def predict(self, obs):  # 选择最优动作
        
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作

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
    temp_txt = open("tempature_epx.txt", "a") # 温度分布文件，保存test part获得的温度分布
    eval_reward = []
    for i in range(1):
        plt.figure(1)
        plt.plot(x_, temp_reset, linestyle='dotted')
        temp_txt.write("\n**********************测试开始*******************************\n")
        obs = env.reset()
        for i in obs:
                temp_txt.write(" "+str(i))
        temp_txt.write('\n')
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            for i in obs:
                temp_txt.write(" "+str(i))
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
        
        self.state_space = spaces.Box(low=20, high=500, shape=(temp_reset.shape[0],), dtype=np.float32)
        self.action_space = spaces.Discrete(11)
        self.terminate_states = 200
        self.voltage = 100
        self.actions = [20,12,8, 6, 4, 2, 0, -2, -4, -6,-8]
        self.state = None
        self.temp_reset = np.ones((temp_reset.shape[0],)) * 20
        self.counts = 0
        self.mytime = time.time()

    def get_state(self, voltage): # 实验台用
        '''输出控制灯管的功率'''
        out_vol = voltage
        
        ''''NIDAQMX采集温度数据'''
        next_state = np.array(task.read())
    
        return next_state     
        
    def get_reward(self, state1, state2, time, count=[0], xs = [0, 0], ys = [1, 1]):
        '''
        count=[0],xs = [0, 0],ys = [1, 1]是函数的默认参数。用于画动态折线图。
        这里利用了python函数默认参数的值只在定义时计算一次，即“调用过函数后，值不释放”
        因此count可以保存调用函数的次数，xs保存着下一个线段的x坐标，ys保存着下一个线段的y坐标
        动态曲线作图原理见下面的注释。
        '''
        count[0]=1 + count[0]
        
        state1 = np.array(state1)
        state2 = np.array(state2)
        
        '''reward函数有两点要求：
        1.grad值越小reward越大，不能出现任何一个grad值变小但reward增大的情况。
          reward函数单调连续即可保证这一点。
        2.随着grad减小，reward应该变化的越来越剧烈。
        因为我选择了线性变化，如果斜率不变，grad越小，它对reward的影响越小，所以需要在grad减小时逐渐增大斜率。
        这可以让reward更快的接近满分。（我猜的）
        '''
        grad_array = (state2 - state1) / time#实际温升速度 
        if np.mean(state2) < 360:
            temperature_grad = 5 #理想温升速度 
        else:
            temperature_grad = 0 #理想温升速度     
            
        grad = 25*((np.mean(grad_array[100:500])- temperature_grad))*((np.mean(grad_array[100:500])- temperature_grad))
        if grad >= 75 :
            re = - grad
        elif grad>= 50:
            re= -2*grad +95
        elif grad>= 25:
            re= -3*grad +165
        elif grad>= 15:
            re= -4*grad +210
        elif grad>= 8:
            re= -6*grad +260
        elif grad>= 4:
            re= -8*grad +296
        elif grad<= 0.2*25:
            re = 400
        else:
            re= -12*grad +331 
         
        print("\n****************count:{}******************\n".format(count[0]))
        print("re: {} grad:{} 实际温升速度：{} 当前状态平均温度:{}\n".format(re,grad,np.mean(grad_array[100:500]),np.mean(state2[100:500])))
        
        #画出实时的reward折线图，横坐标为训练次数（即：get_reward函数的调用次数）
        #基本原理是使用一个长度为2的数组xs，ys，每次替换数据并在原始图像后追加。
        plt.figure(2)
        #plt.axis([0,500,-100,220])
        plt.ion()
        plt.title('Real-time Reward')
        y = re
        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = count[0]
        ys[1] = y
        plt.grid(ls='--')
        plt.plot(xs, ys)
        plt.pause(0.1)
    
        return re
    
    def step(self, action,count=[0],xs = [0, 0],ys = [1, 1]):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        #系统当前状态
        self.voltage = self.voltage + self.actions[action]
        if self.voltage < 80:
            self.voltage = 80
        state = np.array(self.state)
        if state.mean() > self.terminate_states:
            return state, 0, True, {}
        
        #状态转移
        self.next_state = self.get_state(self.voltage)
        
        action_time = time.time() - self.mytime
        self.mytime = time.time() 
        
        next_state = np.array(self.next_state)
        print("action:{}, voltage:{} ".format(self.actions[action], self.voltage))
        print('state: ',state)
        print('next_state: ',next_state)
        print("tempature_mean: ",state.mean())
        is_terminal = False
        
        if next_state.mean() > self.terminate_states:
            is_terminal = True
        
        reward = self.get_reward(state, next_state, action_time)
        
        self.state = self.next_state #更新下一状态
        self.counts += 1 #记录动作次数
        self.temp_reset = np.ones((1, len(temp_reset))) * 20
        
        #画图
        count[0]=1 + count[0]
        plt.figure(3)
        plt.title('Power')
        plt.ion()
        y = self.voltage
        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = count[0]
        ys[1] = y
        plt.bar(xs, ys)
        plt.pause(0.1)
        
        return self.next_state, reward, is_terminal, {}
    
    def render(self):
        '''训练数据可视化'''
        plt.figure(1)
        plt.plot(x_, self.state, linestyle='dotted')
        plt.pause(0.1)
        plt.grid()
        plt.title('Temperature Distribution')
    
         
    def reset(self):
        while np.array(task.read()).mean() > 50:
            pass
        print("*********平均温度下降到50℃以下，开始下一轮训练**********")
        
        self.state = np.array(task.read())  #重置初始温度分布
        self.voltage = 100 #重置输出电压
        return self.state
    
    
# 创建环境
env = myenv()
env.reset()
action_dim = env.action_space.n  # 动作维度
obs_shape = env.state_space.shape  # 状态空间形状

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=0.2,
    e_greed_decrement=1e-6)

# # 加载模型
# save_path = './project_v6.ckpt'
# agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)
print('===========存储完毕============')
max_episode = 1000

# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in range(0, 50):
        print('$$$$$$$$$$$ {} $$$$$$$$$$$'.format(i))
        total_reward = run_episode(env, agent, rpm)
        episode += 1
        save_path = './project_v6.ckpt'
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
save_path = './project_v6.ckpt'
agent.save(save_path)