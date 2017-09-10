# -*- coding: cp936 -*-
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

#����csv�ļ�����
df = pd.read_csv('E:\watermelon_3a.csv')
m,n = shape(df.values)
# print(m,' ',n)
# df['norm'] = ones((m,1))
dataMat = array(df[['norm','density','ratio_sugar']].values[:,:])
# print('dataMat = ',dataMat)
#transpose()����ת��  labelMat��ȡlabel��ֵ
labelMat = mat(df['label'].values[:]).transpose()

#sigmoid ����
def sigmoid(inX):
    return 1.0 / (1+exp(-inX))

#�ݶ������㷨
def gradAscent(dataMat,labelMat):
    m,n = shape(dataMat)    #m = 17:�������� n = 3������ά��   
    alpha = 0.1             # �̶���ѧϰ��
    maxCycles = 500         #ѧϰѭ������
    weights = array(ones((n,1))) #��ʼ��w
    
    for _ in range(maxCycles):    
        a = dot(dataMat,weights)  #matrix mult = w*x
        h = sigmoid(a)     # Ŀ�꺯��
        error = (labelMat - h)  # labelMat��gold-standard
        weights = weights + alpha*dot(dataMat.transpose(),error)
    return weights

#����ݶ�����
def randomgradAscent(dataMat,label,numIter = 50):
    m,n = shape(dataMat)
    weights = ones(n)
    for j in range(m):
        dataIndex = range(m)
        for i in range(m):
            alpha = 40/(1.0+j+i)+0.2    # alpha decreases with iteration        
            randIndex_Index = int(random.uniform(0,len(dataIndex))) #���ѡȡ����
            print('randIndex_Index = ',randIndex_Index)
            h = sigmoid(sum(dot(randIndex_Index,weights)))
            error = (label[randIndex_Index]-h)
            weights = weights+alpha*error[0*0]*(dataMat[randIndex_Index].transpose())
            del(randIndex_Index) #ɾ����ѡ������
            print(weights)
#             randIndex = dataIndex(randIndex_Index)
#             h = sigmoid(sum(dot(dataMat[randIndex],weights)))
#             error = (label[randIndex]-h)
#             weights = weights+alpha*error[0*0]*(dataMat[randIndex].transpose())
#             del(dataIndex[randIndex_Index]) #ɾ����ѡ������
#             print(weights)
    return weights  

#��ͼ
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