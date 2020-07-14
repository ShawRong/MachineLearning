'''
Created on June 8, 2020
Logistic Regression
Author: Rong Shuo U201814683
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


plt_loss_train = []
plt_loss_test = []
plt_acc = []
#---------------------------------------------------
#logistic regression

#read data
def loadData(filename):
    '''
    filename -> an csv file
    in this project it's dataset/data.csv

    and returns
    dataMat --
    lableMat
    '''
    df = pd.read_csv(filename,header = None)
    df = df.drop([0],axis = 1)   #去掉第0列的序号
    data = np.array(df)
    data = data.tolist()
    dataMat = []
    lableMat = []
    for i in range(len(data)):
        dataMat.append(data[i][0:-1])
        lableMat.append((int)(data[i][-1]))
    dataMat = np.array(dataMat)
    lableMat = np.array(lableMat)
    return dataMat,lableMat

# logistic function函数
def logistic(x):
    s = np.clip(1 / (1.0 + np.exp(-x)), 1e-6, 1 - 1e-6)
    return s

def loss_function(y,data,weights,bias,lamda):
    h = logistic(sum(data * weights)+bias)
    cross_entropy = -np.dot(y, np.log(h)) - np.dot((1 - y), np.log(1 - h))
    return cross_entropy + lamda * np.sum(np.square(weights))


#随机梯度下降算法
def stocgradAscent(dataMat,lableMat,cycle = 50,eta = 0.2,lamda = 0):
    '''
    每次随机使用一个样本进行更新回归系数
    cycle: 迭代次数
    eta: 学习率,这里选择了自动适应的学习率
    '''
    m,n = np.shape(dataMat)
    weights = np.zeros(n) #w参数初始化为0
    bias = 0

    #随机梯度下降
    LOSS = np.zeros(m)
    for j in range(cycle):
        print("cycle:",j)
        dataIndex = list(range(m))

        for i in range(m):
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = logistic(sum(dataMat[dataIndex[randIndex]] * weights)+bias)
            error = lableMat[dataIndex[randIndex]] - h
            LOSS[dataIndex[randIndex]] = loss_function(lableMat[dataIndex[randIndex]],dataMat[dataIndex[randIndex]],weights,bias,lamda)
            weights = weights + eta/np.sqrt(j+1)*(error*dataMat[dataIndex[randIndex]] - lamda*weights)
            bias = bias + eta/np.sqrt(j+1)*error #
            del(dataIndex[randIndex])
        #print("weights:",weights)
        loss = sum(LOSS)/m
        plt_loss_train.append(loss)
        print("bias",bias)
        print("loss:",loss)
    return weights,bias

def score(dataMat,lableMat,weights,bias):
    length = len(lableMat)
    predictions = []
    LOSS = np.zeros(length)
    for i in range(length):
        h = logistic(sum(dataMat[i] * weights)+bias)
        LOSS[i] = loss_function(lableMat[i],dataMat[i],weights,bias,0)
        if(h > 0.5):
            predictions.append(1)
        else:
            predictions.append(0)
    loss = sum(LOSS) / length
    plt_loss_test.append(loss)
    errorCount = 0
    for i in range(length):
        if(lableMat[i] != predictions[i]):
            print("wrong")
            errorCount += 1
    acc = 1 - errorCount/length
    plt_acc.append(acc)
    return acc

def test(cycle,eta,lamda):
    dataMat,lableMat = loadData('dataset/data.csv')
    sc = StandardScaler()
    sc.fit(dataMat)  # 计算均值和方差
    dataMat_std = sc.transform(dataMat)
    dataMat_train = dataMat_std[:3000]
    dataMat_test = dataMat_std[3000:]
    lableMat_train = lableMat[:3000]
    lableMat_test = lableMat[3000:]
    weights,bias = stocgradAscent(dataMat_train,lableMat_train,cycle,eta,lamda)
    acc = score(dataMat_test,lableMat_test,weights,bias)
    print('acc:',acc)

if __name__ == "__main__":
    cycle = 1000
    lamdas = [0.000025,0.00004375,0.0000625,0.00008125,0.0001]
    color = ['red','blue','yellow','black','green']
    lables = ['lamda = 0.000025','lamda = 0.00004375','lamda = 0.0000625','lamda = 0.00008125','lamda = 0.0001']
    plt.subplot(2,1,2)
    for i in range(len(lamdas)):
        plt_loss_train = []
        test(cycle,0.05,lamdas[i])
        plt.plot(list(range(cycle)),plt_loss_train,c = color[i],label = lables[i])
    plt.legend(loc='best')
    plt.xlabel('cycle')
    plt.ylabel('loss')
    plt.subplot(2, 2, 2)
    plt.scatter(lamdas,plt_loss_test)
    plt.xlabel('lamdas')
    plt.ylabel('loss_test')
    plt.subplot(2, 2, 1)
    plt.scatter(lamdas,plt_acc)
    plt.xlabel('lamdas')
    plt.ylabel('acc')
    plt.show()


