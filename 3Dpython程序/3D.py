# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:38:30 2021

@author: 上善若水-仰望星空
"""
import numpy as np
import time
import math
import os
from numba import jit

time0=time.time()

def q_out(t,l):#散热(辐射+对流)热流密度( >0 )
    q_r=0
    q_c=0
    q_total=0
    #double h_r,h_total;
    t0=20 #环境温度和流体温度
    g=9.8
    SIGMA_epsilon=4.536e-8 #SIGMA=0.8,epsilon=5.67e-8
    #const double epsilon=0.8;
    q_r=SIGMA_epsilon*pow((273+t),4)-334.3055 #环境t0^4已经计算好
    #h_r=(SIGMA_epsilon*pow((273+t),4)-334.3055)/(t-t0);//环境t0^4已经计算好
    if t>=t0:
        v=(8e-5*t*t+0.0938*t+13.061)*1e-6
        Pr=3e-7*t*t-0.0002*t+0.7059
        lambda_air=3e-8*t*t+8e-5*t+0.0242
        Gr=g*(t-t0)*l*l*l/v/v/(t+273)/1e9
        Ra=Gr*Pr
        a=(math.sqrt(Pr)*2+1+2*Pr)*5/2/Pr
        a=1/a
        Nu=pow(a,0.25)*pow(Ra,0.25)*3/4
        h_c=Nu*lambda_air/l*1000
        q_c=h_c*(t-t0)
    else:
        q_c=0
    q_total=q_c+q_r
    #h_total=h_c+h_r;
    return q_total

def q_c(t,l): #自然对流散热，热流密度( >0 )
    t0=20 #环境温度和流体温度
    g=9.8
    if t>=t0:
        v=(8e-5*t*t+0.0938*t+13.061)*1e-6
        Pr=3e-7*t*t-0.0002*t+0.7059
        lambda_air=3e-8*t*t+8e-5*t+0.0242
        Gr=g*(t-t0)*l*l*l/v/v/(t+273)/1e9
        Ra=Gr*Pr
        a=(math.sqrt(Pr)*2+1+2*Pr)*5/2/Pr
        a=1/a
        Nu=pow(a,0.25)*pow(Ra,0.25)*3/4
        h_c=Nu*lambda_air/l*1000
        q_c=h_c*(t-t0)
    else:
        return 0
    return q_c

fin=open('DATA.txt', 'r')
transist1=open('transist1.txt', 'w')#中间一圈
transist2=open('transist2.txt', 'w')#中间一点（平均值）
if (not os.path.exists('DATA.txt'))or(not os.path.exists('transist1.txt'))\
    or(not os.path.exists('transist2.txt')):
	print( "can not read the file" )
    
transist1.write("每一时刻的 中间一圈每一点 瞬态温度（℃）\n")
transist2.write("每一时刻的 中间一圈每一点 瞬态温度的平均值（℃）\n")

row=200
column=600
radius=5
#轴向、周向、径向
hight=155.0
R_engine=80.0
thickness=3.0
#轴向、周向、径向
time_num=1
freq=1  #频率，时间步长1s,总时间=time_num/freq=80s,dt=1/freq
P=150.0#总功率
num_seed=1e9
t0=20.0#环境温度
tf=20.0#流体温度
pi=3.14159
absorption_rate=0.9
lambda_=0.6
cp=1400.0
rho=1750.0 #壳体物性参数J/(kgK) kg/m3

#输入蒙特卡洛法结果，参数需修改
q_tmp=np.ones((600,1800))
q_in=np.ones((row+1,column))
p=P/num_seed
counttemp=0
for m in range(1800):
    for n in range(600):   
        if counttemp%600!=0 or counttemp==0:#txt文件中数据每600个有换行符，需要避开
            q_tmp[n][m]=eval(fin.read(6).strip())#先输入1000
        else:
            q_tmp[n][m]=eval(fin.read(7).strip())#先输入1000
        counttemp+=1
fin.close()

#每个网格上的热流量W
for m in range(column):#将网格变疏
    for n in range(row):  #将3个格子求平均, [W]
        q_in[n][m]=(q_tmp[3*n][3*m]+q_tmp[3*n+1][3*m+1]+q_tmp[3*n+2][3*m+2])/3*p*9*absorption_rate
        q_in[row][m]=q_in[row-1][m]/2#面积缩小一半
        q_in[0][m]=q_in[0][m]/2
        
#网格尺寸参数,m，采用均匀网格
dz=hight/row*0.001
dphi=2*pi/column #rad
dr=thickness/radius*0.001

#初始化3D温度场
T=np.ones((radius+3,column+2,row+3)) #r,phi,z->i,j,k
U=np.ones((radius+3,column+2,row+3))
V=np.ones((radius+3,column+2,row+3))
T1=np.ones((radius+3,column+2,row+3))
for n in range(radius+3):
    for j in range(column+2):
        for k in range(row+3):
            T[n][j][k]=tf
            U[n][j][k]=tf
            V[n][j][k]=tf
            T1[n][j][k]=tf
for i in range(1,radius+2):
    for j in range(1,column+2):
        for k in range(1,row+2):
            T[i][j][k]=t0
            
#系数，均与rn,rs,rp有关（除了边界）
Awe=dr*dz*lambda_/dphi #还没有除以(/)re或rw！！
Ans=dphi*dz*lambda_/dr #还没有乘以(*)rn或rs！！
Abd=dr*dphi*lambda_/dz*0.5 #还没有乘以(*)(rn+rs)！！
DeltaV=dr*dphi*dz #还没有乘以(*)(rn+rs)/2！！
Ap0=rho*cp*DeltaV*freq*2.0 #还没有乘以(*)(rn+rs)/2！！  !/(delta_t/2)!brian ADI
deltaS_ns=dz*dphi*R_engine*0.001 #竖直微元面积，法向为ns
rp=np.ones(radius+1)
for i in range(radius+1):
    rp[i]=R_engine*0.001-i*dr
    
#ns方向系数随半径变化的值
an=np.ones(radius+1)
_as=np.ones(radius+1)
c_ns=np.ones(radius+1)#r,radius+1,ns
for i in range(radius+1):
    an[i]=Ans*(rp[i]+dr/2)
    _as[i]=Ans*(rp[i]-dr/2)
an[0]=0  
_as[radius]=0

#we方向系数随半径变化的值
a_we=np.ones(radius+1)
c_we=np.ones(column)
for i in range(radius+1):
    a_we[i]=Awe/rp[i] #re=rw=rp
a_we[0]=a_we[0]/2
a_we[radius]=a_we[radius]/2

#bd方向系数随半径变化的值
a_bd=np.ones(radius+1)
c_bd=np.ones(row+1)
for i in range(radius+1):
    a_bd[i]=Abd*2*rp[i]# rn+rs=2rp
a_bd[0]=a_bd[0]/2
a_bd[radius]=a_bd[radius]/2
    
#非稳态项系数ap0随半径变化的值
ap0=np.ones(radius+1) #ap=ap0+an+as（ap0+该方向另外两个）
for i in range(radius+1):
    ap0[i]=Ap0*rp[i] #rn+rs=2rp
ap0[0]=ap0[0]/2
ap0[radius]=ap0[radius]/2
    
TDMA_P_ns=np.ones(radius+2)
TDMA_Q_ns=np.ones(radius+2) #TDMA计算系数
for i in range(radius+2):
    TDMA_P_ns[i]=0
    TDMA_Q_ns[i]=0
TDMA_P_we=np.ones(column+1)
TDMA_Q_we=np.ones(column+1)
for j in range(column+1):
	TDMA_P_we[j]=0
	TDMA_Q_we[j]=0
TDMA_P_bd=np.ones(row+2)
TDMA_Q_bd=np.ones(row+2)
for k in range(row+2):
	TDMA_P_bd[k]=0
	TDMA_Q_bd[k]=0

T_sum1=0 
T_sum2=0 #瞬态温度变化

#计算温度场
#for(int time_tmp=0;time_tmp<10;time_tmp++)
@jit
for time_tmp in range(time_num):
#一、ADI--r方向使用TDMA隐式求解
    for k in range(row+1):
        for j in range(column):#we
             for i in range(radius+1):#r,radius+1
                 #边界系数的处理	
                 an_tmp=an[i]
                 as_tmp=_as[i]
                 ap0_tmp=ap0[i]
                 awe_tmp=a_we[i]
                 ab_tmp=a_bd[i]
                 ad_tmp=a_bd[i]
                 deltaS_ns_tmp=deltaS_ns
                 if (k==0 or k==row):#上下边界,dz/2
                     an_tmp=an_tmp/2
                     as_tmp=as_tmp/2					
                     ap0_tmp=ap0_tmp/2
                     deltaS_ns_tmp=deltaS_ns/2
                     awe_tmp=awe_tmp/2
                 #if(k==0)    ad_tmp=0;
				 #if(k==row)  ab_tmp=0;
                 
                 ap=ap0_tmp+an_tmp+as_tmp
                 i=i+1
                 j=j+1
                 k=k+1
                 #i=1~6,j=1~600,k=1~201
                 if(i==1): #边界节点，对流和辐射项
                     c_ns[i-1]=awe_tmp*(T[i][j-1][k]+T[i][j+1][k]-2*T[i][j][k])+ad_tmp*(T[i][j][k-1]\
							-T[i][j][k])+ab_tmp*(T[i][j][k+1]-T[i][j][k])+ap0_tmp*T[i][j][k]\
							+q_in[k-1][j-1]-q_out(T[1][j][k],k*dz)*deltaS_ns_tmp	# [W]
                 elif(i==radius+1):#边界节点，对流项
                     c_ns[i-1]=awe_tmp*(T[i][j-1][k]+T[i][j+1][k]-2*T[i][j][k])+ad_tmp*(T[i][j][k-1]\
							-T[i][j][k])+ab_tmp*(T[i][j][k+1]-T[i][j][k])+ap0_tmp*T[i][j][k]\
							-q_c(T[radius+1][j][k],k*dz)*deltaS_ns_tmp #[W]
                 else:
                     c_ns[i-1]=awe_tmp*(T[i][j-1][k]+T[i][j+1][k]-2*T[i][j][k])+ad_tmp*(T[i][j][k-1]\
							-T[i][j][k])+ab_tmp*(T[i][j][k+1]-T[i][j][k])+ap0_tmp*T[i][j][k]
                 i=i-1
                 j=j-1
                 k=k-1
                 
             #P,Q从1开始，对应的as,an,c_ns从0开始,i=0~25
             tmp=(ap-an_tmp*TDMA_P_ns[i])
             TDMA_P_ns[i+1]=as_tmp/tmp
             TDMA_Q_ns[i+1]=(c_ns[i]+an_tmp*TDMA_Q_ns[i])/tmp
                    #if(k==0||k==row)//上下边界，系数还原
					#{
					#	deltaS_ns=deltaS_ns*2;}
                    #i括号
             j=j+1 
             k=k+1 #j:1~600,k:1~501
                 #实体节点温度i=26~1
             U[radius+1][j][k]=TDMA_Q_ns[radius+1]
				 #cout<<TDMA_Q[radius+1]<<endl;
             for i in range(radius+1,1,-1):
                 U[i-1][j][k]=TDMA_Q_ns[i-1]+TDMA_P_ns[i-1]*U[i][j][k]
                 if(math.isnan(U[i-1][j][k]) or U[i-1][j][k]>1000):  #(5,2,1)有问题,(5,4,1)1380℃
                     print("ns;time:",time_tmp,'\n')
                     print(i-1,",",j,",",k,"\n")
                     print(U[i-1][j][k],'\n')

             #径向虚拟节点温度
             U[0][j][k]=U[1][j][k]
             U[radius+2][j][k]=U[radius+1][j][k]
             j=j-1 
             k=k-1
    #周向虚拟节点温度
    k=k+1 #k:1~501
    for i in range(radius+3):
        U[i][column+1][k]=U[i][1][k]
        U[i][0][k]=U[i][600][k]
    k=k-1
    #轴向虚拟节点温度（上下底面）
    for i in range(radius+3):
        for j in range(column+2):
            U[i][j][0]=U[i][j][1]
            U[i][j][row+2]=U[i][j][row+1]
            
    #* ------------------------------------------------------------------ */
		#二、ADI--phi方向使用TDMA隐式求解        
    for k in range(row+1):#bd,501
        for i in range(radius+1): #ns,26
            awe_tmp=a_we[i]
            an_tmp=an[i]
            as_tmp=_as[i]
            ap0_tmp=ap0[i]
            ab_tmp=a_bd[i]
            ad_tmp=a_bd[i]
            deltaS_ns_tmp=deltaS_ns
            if(k==0 or k==row): #上下边界,dz/2
                an_tmp=an[i]/2
                as_tmp=_as[i]/2
                ap0_tmp=ap0[i]/2
                deltaS_ns_tmp=deltaS_ns/2
                awe_tmp=a_we[i]/2
            if(k==0):    
                ad_tmp=0
            if(k==row):  
                ab_tmp=0
            for j in range(column):#we,600
                ap=ap0_tmp+awe_tmp+awe_tmp
                i=i+1
                j=j+1
                k=k+1 #i=1~26,j=1~600,k=1~501
                if(i==1):#边界节点，对流和辐射项
                    c_we[j-1]=an_tmp*(U[i-1][j][k]-U[i][j][k])+as_tmp*(U[i+1][j][k]-U[i][j][k])+ad_tmp*(T[i][j][k-1]\
							-T[i][j][k])+ab_tmp*(T[i][j][k+1]-T[i][j][k])+ap0_tmp*T[i][j][k]\
							+q_in[k-1][j-1]-q_out(T[1][j][k],k*dz)*deltaS_ns_tmp  # T !!
                elif(i==radius+1):#边界节点，对流项
                    c_we[j-1]=an_tmp*(U[i-1][j][k]-U[i][j][k])+as_tmp*(U[i+1][j][k]-U[i][j][k])+ad_tmp*(T[i][j][k-1]\
							-T[i][j][k])+ab_tmp*(T[i][j][k+1]-T[i][j][k])+ap0_tmp*T[i][j][k]\
							-q_c(T[radius+1][j][k],k*dz)*deltaS_ns_tmp;	# [W]
                else: #内部节点
                    c_we[j-1]=an_tmp*(U[i-1][j][k]-U[i][j][k])+as_tmp*(U[i+1][j][k]-U[i][j][k])+ad_tmp*(T[i][j][k-1]\
							-T[i][j][k])+ab_tmp*(T[i][j][k+1]-T[i][j][k])+ap0_tmp*T[i][j][k];
                i=i-1
                j=j-1
                k=k-1
                #P,Q从1开始，对应的c_ns从0开始,j=0~599
                tmp=(ap-awe_tmp*TDMA_P_we[j])
                TDMA_P_we[j+1]=awe_tmp/tmp
                TDMA_Q_we[j+1]=(c_we[j]+awe_tmp*TDMA_Q_we[j])/tmp
            #上下边界，系数还原    
            if(k==0 or k==row):
                awe_tmp=a_we[i]
                an_tmp=an[i]
                as_tmp=_as[i]
                ap0_tmp=ap0[i]
                ab_tmp=a_bd[i]
                ad_tmp=a_bd[i]
                deltaS_ns_tmp=deltaS_ns
            i=i+1 
            k=k+1 #i:1~26,k:1~201
            #实体节点温度j=600~1
            V[i][column][k]=TDMA_Q_we[column]
            for j in range(column,1,-1):
                 V[i][j-1][k]=TDMA_Q_we[j-1]+TDMA_P_we[j-1]*V[i][j][k]
                 if(math.isnan(V[i][j-1][k]) or V[i][j-1][k]>1000): 
                     print("we;time:",time_tmp,"\n")
                     print(i,",",j-1,",",k,"\n")
                     print(V[i][j-1][k],"\n")
            
            
            #周向虚拟节点温度
            V[i][0][k]=V[i][column][k]
            V[i][column+1][k]=V[i][1][k]
            i=i-1 
            k=k-1
            
        #径向虚拟节点温度
        k=k+1 #k:1~501
        for j in range(column+2): 
            V[0][j][k]=V[1][j][k]
            V[radius+2][j][k]=V[radius+1][j][k]
        k=k-1
        
    #轴向虚拟节点温度（上下底面）
    for i in range(radius+3):
        for j in range(column+2): 
            V[i][j][0]=V[i][j][1]
            V[i][j][row+2]=V[i][j][row+1]
                        
    # ------------------------------------------------------------------ */
	#三、ADI--z方向使用TDMA隐式求解
    for i in range(radius+1):#ns
        awe_tmp=a_we[i]
        an_tmp=an[i]
        as_tmp=_as[i]
        ap0_tmp=ap0[i]
        ab_tmp=a_bd[i]
        ad_tmp=a_bd[i]
        deltaS_ns_tmp=deltaS_ns
        for j in range(column):#we
            for k in range(row+1): #bd
                if(k==0 or k==row):#上下边界,dz/2
                    an_tmp=an_tmp/2
                    as_tmp=as_tmp/2						
                    ap0_tmp=ap0_tmp/2
                    deltaS_ns_tmp=deltaS_ns/2
                    awe_tmp=awe_tmp/2
                if(k==0):    
                    ad_tmp=0
                if(k==row): 
                    ab_tmp=0
                ap=ap0_tmp+ab_tmp+ad_tmp
                i=i+1
                j=j+1
                k=k+1   #i=1~26,j=1~600,k=1~201
                if(i==1):#边界节点，对流和辐射项
                    c_bd[k-1]=an_tmp*(U[i-1][j][k]-U[i][j][k])+as_tmp*(U[i+1][j][k]-U[i][j][k])+awe_tmp*(V[i][j-1][k]\
							-V[i][j][k])+awe_tmp*(V[i][j+1][k]-V[i][j][k])+ap0_tmp*V[i][j][k]\
							+q_in[k-1][j-1]-q_out(V[1][j][k],k*dz)*deltaS_ns_tmp
                elif(i==radius+1):#边界节点，对流项
                    c_bd[k-1]=an_tmp*(U[i-1][j][k]-U[i][j][k])+as_tmp*(U[i+1][j][k]-U[i][j][k])+awe_tmp*(V[i][j-1][k]\
							-V[i][j][k])+awe_tmp*(V[i][j+1][k]-V[i][j][k])+ap0_tmp*V[i][j][k]\
							-q_c(V[radius+1][j][k],k*dz)*deltaS_ns_tmp
                else: #内部节点
                    c_bd[k-1]=an_tmp*(U[i-1][j][k]-U[i][j][k])+as_tmp*(U[i+1][j][k]-U[i][j][k])+awe_tmp*(V[i][j-1][k]\
							-V[i][j][k])+awe_tmp*(V[i][j+1][k]-V[i][j][k])+ap0_tmp*V[i][j][k]
                i=i-1
                j=j-1
                k=k-1
                #P,Q从1开始，对应的as,an,c_ns从0开始,k=0~500
                tmp=(ap-ad_tmp*TDMA_P_bd[k])
                TDMA_P_bd[k+1]=ab_tmp/tmp
                TDMA_Q_bd[k+1]=(c_bd[k]+ad_tmp*TDMA_Q_bd[k])/tmp
                if(k==0 or k==row):#上下边界，系数还原
                    awe_tmp=a_we[i]
                    an_tmp=an[i]
                    as_tmp=_as[i]
                    ap0_tmp=ap0[i]
                    ab_tmp=a_bd[i]
                    ad_tmp=a_bd[i]
                    deltaS_ns_tmp=deltaS_ns
            i=i+1 
            j=j+1 #i:1~26,j:1~600   
            #实体节点温度k=501~1
            T[i][j][row+1]=TDMA_Q_bd[row+1]
            for k in range(row+1,1,-1):
                T[i][j][k-1]=TDMA_Q_bd[k-1]+TDMA_P_bd[k-1]*T[i][j][k]
                if(math.isnan(T[i][j][k-1]) or T[i][j][k-1]>1000):  #(6,1,200)有问题,原因：V[6][j][k]<20,用q_c()计算时出错
											#   解决办法：改V用U计算q_c
                    print("bd;time:",time_tmp,"\n")
                    print(i,",",j,",",k-1,"\n")
                    print(T[i][j][k-1],"\n")
            
            #轴向虚拟节点温度
            T[i][j][0]=T[i][j][1];
            T[i][j][row+2]=T[i][j][row+1];
            i=i-1 
            j=j-1
            
        #周向虚拟节点温度
        i=i+1 #i:1~26
        for k in range(row+3):#k:503
            T[i][0][k]=T[i][column][k]
            T[i][column+1][k]=T[i][1][k]
        i=i-1
        
    #径向虚拟节点温度（外表面）
    for j in range(column+2):
            for k in range(row+3):
                V[0][j][k]=V[1][j][k]
                V[radius+2][j][k]=V[radius+1][j][k]
                
#------------------------------------------------------------------ */
#输出：瞬态温度，中间一圈,transist1.txt
    middle=int(row/2)
    transist1.write(str(time_tmp)+"\n")
    for j in range(column+1):
        transist1.write("{:.2f} ".format(T[1][j][middle]))
        T_sum1+=T[1][j][middle]
    transist1.write("\n"+"deltaT ="+str((T_sum1-T_sum2)/column)+"\n"+"\n")
    T_sum2=T_sum1
    
    #输出：一点的瞬态温度，中间一圈的平均值,transist2.txt
    transist2.write(str(time_tmp)+"s:"+"{:.2f} ".format(T_sum1/column)+"\n")
    T_sum1=0
    
#-------------------------输出结果-------------------------  
surface_data=open('surfacee tempureture distribution.txt', 'w')
pure_data=open('surface pure data.txt', 'w')
if (not os.path.exists('surfacee tempureture distribution.txt'))or(not os.path.exists('surface pure data.txt')):
	print( "can not read the file" ,"\n")

#输出：表面完整温度场
for k in range(row+1,0,-1):
    surface_data.write(str(k)+"\n")
    for j in range(1,column+1):#we
        surface_data.write("{:.2f} ".format(T[1][j][k]))
        pure_data.write("{:.2f} ".format(T[1][j][k]))
        surface_data.write("\n\n")
        pure_data.write("\n")

surface_data.close()
pure_data.close()
transist1.close()
transist2.close()

time1=time.time()
print("finish","\n")
print("\nThe run time is: {} s\n".format(time1-time0))