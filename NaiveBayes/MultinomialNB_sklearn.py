import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

#使用pandas读取csv文件

def readcsv():
    df = pd.read_csv("dataset/spam.csv",encoding = "latin-1")
    'print(df.head())'
    df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis = 1)
    'print(df.describe())'
    data = np.array(df)
    'print(data)'
    return data

def getdata(Volum,split,data):
    datatemp = (data.T)
    datatemp = datatemp
    y = datatemp[0][:Volum]
    x = datatemp[1][:Volum]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = split)
    return x,y,x_train,x_test,y_train,y_test

def haveatry(Volum,data):
    x, y, x_train, x_test, y_train, y_test = getdata(Volum, 0.1, data)

    #利用sklearn对邮件预处理

    vectorizer = CountVectorizer()
    x_train_cnt = vectorizer.fit_transform(x_train)
    x_test_cnt = vectorizer.transform(x_test)

    print(x_train_cnt)


    #sklearn贝叶斯分类

    clf = MultinomialNB()
    clf.fit(x_train_cnt,y_train)
    score = clf.score(x_test_cnt,y_test)
    print("using sklearn acc:",score)
    return score

if __name__ == "__main__":
    accsarray = []
    count = 5
    step = 5
    for k in range(count):
        accs = []
        for i in range(10, 50, step):
            Volum = i * 100
            data = readcsv()
            acc = haveatry(Volum, data)
            accs.append(acc)
        accsarray.append(accs)
    accsarray = np.array(accsarray)
    meanaccs = accsarray.mean(axis=0)

    x = list(range(10, 50, step))
    x = np.array(x)
    x = x * 100
    plt.plot(x, meanaccs)
    plt.xlabel('Volume')
    plt.ylabel('Accuracy')
    plt.ylim(0.89,)
    plt.show()