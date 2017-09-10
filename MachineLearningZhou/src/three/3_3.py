# -*- coding: cp936 -*-
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

#读入csv文件数据
df = pd.read_csv('E:\watermelon_3a.csv')
m,n = shape(df.values)
# print(m,' ',n)
# df['norm'] = ones((m,1))
dataMat = array(df[['norm','density','ratio_sugar']].values[:,:])
# print('dataMat = ',dataMat)
#transpose()矩阵转置  labelMat获取label的值
labelMat = mat(df['label'].values[:]).transpose()

#sigmoid 函数
def sigmoid(inX):
    return 1.0 / (1+exp(-inX))

#梯度上升算法
def gradAscent(dataMat,labelMat):
    m,n = shape(dataMat)    #m = 17:样本数量 n = 3：样本维数   
    alpha = 0.1             # 固定的学习率
    maxCycles = 500         #学习循环次数
    weights = array(ones((n,1))) #初始化w
    
    for _ in range(maxCycles):    
        a = dot(dataMat,weights)  #matrix mult = w*x
        h = sigmoid(a)     # 目标函数
        error = (labelMat - h)  # labelMat是gold-standard
        weights = weights + alpha*dot(dataMat.transpose(),error)
    return weights

#随机梯度上升
def randomgradAscent(dataMat,label,numIter = 50):
    m,n = shape(dataMat)
    weights = ones(n)
    for j in range(m):
        dataIndex = range(m)
        for i in range(m):
            alpha = 40/(1.0+j+i)+0.2    # alpha decreases with iteration        
            randIndex_Index = int(random.uniform(0,len(dataIndex))) #随机选取样本
            print('randIndex_Index = ',randIndex_Index)
            h = sigmoid(sum(dot(randIndex_Index,weights)))
            error = (label[randIndex_Index]-h)
            weights = weights+alpha*error[0*0]*(dataMat[randIndex_Index].transpose())
            del(randIndex_Index) #删除所选的样本
            print(weights)
#             randIndex = dataIndex(randIndex_Index)
#             h = sigmoid(sum(dot(dataMat[randIndex],weights)))
#             error = (label[randIndex]-h)
#             weights = weights+alpha*error[0*0]*(dataMat[randIndex].transpose())
#             del(dataIndex[randIndex_Index]) #删除所选的样本
#             print(weights)
    return weights  

#画图
def plotBestFit(weights):
    dataMat=array(df[['density','ratio_sugar']].values[:,:])  
    labelMat=mat(df['label'].values[:]).transpose()  
    m = shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i,0])
            ycord1.append(dataMat[i,1])
        else:
            xcord2.append(dataMat[i,0])
            ycord2.append(dataMat[i,1])
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')  
    ax.scatter(xcord2,ycord2,s=30,c='green')  
    x=arange(-0.2,0.8,0.1)  
    y=array((-weights[0,0]*x)/weights[0,1])
    #y=array((-weights[0]-weights[1]*x)/weights[2]) 
    print(x) 
    print (shape(x))  
    print (shape(y))  
    plt.sca(ax)  
    plt.plot(x,y)      #ramdom gradAscent  
    #plt.plot(x,y[0])   #gradAscent  
    plt.xlabel('density')  
    plt.ylabel('ratio_sugar')  
    #plt.title('gradAscent logistic regression')  
    plt.title('ramdom gradAscent logistic regression')  
    
    #plt.ion()
    plt.show()  
    
    #weights = gradAscent(dataMat,labelMat)
weights = randomgradAscent(dataMat,labelMat)
plotBestFit(weights)