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

temp_reset = np.ones((4, 601)) * 20

class myenv():
    
    def __init__(self):
        
        self.state_space = spaces.Box(low=25, high=500, shape=(4,601), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.terminate_states = 500
        self.actions = [1000, 1200, 1400, 1600, 1800]
        self.state = None
        self.next_state = None
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
    
    #@jit # 加速get_state函数     
    #输入功率P(int)，当前温度分布T(维度(16，601))，输出0.1s后的温度分布T1(维度(16，601))
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
    
    
    def get_reward(self, state1, state2):
        ''''该函数比较两个状态并返回奖励值'''
        state1 = np.array(state1)
        state2 = np.array(state2)
        
        min_temperature = np.min(state2)
        
        if min_temperature<300:   
            w1 = 0.8
            w2 = 0.2
        else:
            w1 = 0.3
            w2 = 0.7    
        '''
        w1= 0.8
        w2= 0.2
        '''
        state2_var = np.var(state2) #越大越差
        grad_array = (state2 - state1) / 1
        temperature_grad = 5
        #grad = np.mean(grad_array)
        grad = abs(np.mean(grad_array) - temperature_grad) #越大越差
        
        if math.log(state2_var) > 2 :
            re1 = -math.log(state2_var) 
        elif math.log(state2_var)>0 and math.log(state2_var)<=2:
            re1 = -2 * math.log(state2_var) 
        else:
            re= 1
            
        re2 = -math.log(grad) 
        re = w1 * re1 + w2 * re2   
        print("re1: {} re2:{} reward:{} min_temperature:{} \n".format(re1,re2,re,min_temperature))
        return re
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        #系统当前状态
        state = np.array(self.state)
        r_state = state[0][:]
        if state.mean() > self.terminate_states:
            return state, 0, True, {}
        
        #状态转移
        self.next_state = np.array(self.get_state(self.actions[action],self.state))
        r_next_state = self.next_state[0][:]
        print("action: ",action)
        print('state: ',r_state[0:5])
        print('next_state: ',r_next_state[0:5])
        is_terminal = False
        
        if self.next_state.mean() > self.terminate_states:
            is_terminal = True
        
        reward = self.get_reward(r_state, r_next_state)
        
        self.state = self.next_state #更新下一状态
        self.counts += 1 #记录动作次数
        self.temp_reset = np.ones((4, 601)) * 20
        
        return self.next_state, reward, is_terminal, {}
    '''
    def render(self):
       
        plt.plot(x_,self.state[0][:],linestyle='dotted')
    '''
         
    def reset(self):
        self.state = self.temp_reset
        return self.state
        
    
env = myenv()