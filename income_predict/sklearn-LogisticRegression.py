from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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

def test():
    dataMat,lableMat = loadData('dataset/data.csv')
    x_train = dataMat[:3000]
    x_test = dataMat[3000:]
    y_train = lableMat[:3000]
    y_test = lableMat[3000:]
    sc = StandardScaler()
    sc.fit(x_train)  # 计算均值和方差
    x_train_std = sc.transform(x_train) #利用计算好的方差和均值进行Z分数标准化
    x_test_std = sc.transform(x_test)
    lr = LR(max_iter = 1000,random_state=123)
    lr.fit(x_train_std, y_train)
    y_pred = lr.predict(x_test_std)
    acc = accuracy_score(y_test,y_pred)
    print('acc:',acc)
    lr.coef_

if __name__ == "__main__":
    test()
