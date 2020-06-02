import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.stem import WordNetLemmatizer
#使用pandas读取csv文件

def readcsv():
    df = pd.read_csv("dataset/spam.csv",encoding = "latin-1")
    'print(df.head())'
    df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis = 1)
    'print(df.describe())'
    data = np.array(df)
    'print(data)'
    return data

#Volum max = 5573
def getdata(Volum,split,data):
    datatemp = (data.T)

    y = datatemp[0][:Volum]
    #把 标签 ham 替换为 1 spam 替换为 0
    for k in range(0,len(y)):
        if (y[k] == "ham"):
            y[k] = 1
        else:
            y[k] = 0

    x = datatemp[1][:Volum]

    #随机划分，划分比例9：1
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = split)
    return x,y,x_train,x_test,y_train,y_test

#Naive Bayes

def parser(string,stopwords):#从字符串中得到所有词的列表,并且清除停用词
    punctuations = """,.<>()*&^!";~`[]{}|、\\/~+=?:"""
    content = string.replace('\r\n', ' ').replace('\t', ' ')
    clean_word = []
    for punctuation in punctuations:
        content = (' '.join(content.split(punctuation))).replace('  ', ' ')
        clean_word = [word.lower()
            for word in content.split(' ') if (len(word) >= 1) and (word not in stopwords)]
    return clean_word

def createVocabList(docList):  #实现单词表
    vocabSet = set([])
    for document in docList:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def createTrainMat(VocabList,trainData):
    returnTrain = np.zeros(len(VocabList))
    for word in trainData:
        if word in VocabList:
            returnTrain[VocabList.index(word)] += 1
    return returnTrain

def train(train_x_cnt,train_y):
    sampleNum = len(train_x_cnt) #样本数
    trainFeature = len (train_x_cnt[0]) #特征的数量
    pVa = (sum(train_y) + 1) / (len(train_y) + 2) #不是垃圾邮件的先验概率
    p0Num = np.ones(trainFeature)  #拉普拉斯平滑，初始值置1
    p1Num = np.ones(trainFeature)

    p0Denom = 1.0 * trainFeature
    p1Denom = 1.0 * trainFeature

    for i in range(sampleNum):
        if(train_y[i] == 1):   #非垃圾邮件
            p1Num += train_x_cnt[i]
            p1Denom += sum(train_x_cnt[i])
        else:                  #垃圾邮件
            p0Num += train_x_cnt[i]
            p0Denom += sum(train_x_cnt[i])
    pw0 = np.log(p0Num/p0Denom)
    pw1 = np.log(p1Num/p1Denom)   #防止下溢出
    return pw0,pw1,pVa  #返回垃圾邮件的类条件概率 非垃圾邮件的类条件概率 不是垃圾邮件的先验概率

def predictionResult(testMat,pw0,pw1,pVa):
    p1 = (sum(testMat * pw1) + np.log(pVa))
    p0 = (sum(testMat * pw0) + np.log(1.0 - pVa))
    if p1 > p0:
        return 1
    else:
        return 0


def haveatry(Volum,data):
    #Pretreatment

    temp = pd.read_table('dataset/stopwords.txt',encoding = "latin-1")  #禁用词
    stopWords = (np.array(temp).reshape(-1,546))[0]
    #lemmatizer = WordNetLemmatizer()  #词形还原

    x,y,x_train,x_test,y_train,y_test = getdata(Volum,0.1,data)
    #stopWords = []

    docList = []
    for i in range(len(x)):
        wordList = parser(x[i],stopWords)
        docList.append(wordList)
    vocabList = createVocabList(docList)

    print(vocabList)

    #train
    x_train_cnt = []
    for i in range(len(x_train)):
        wordList = parser((x_train[i]),stopWords)
        x_train_cnt.append(wordList)
    trainMats = []
    for i in range(len(x_train_cnt)):
        trainMats.append(createTrainMat(vocabList,x_train_cnt[i]))

    p0w,p1w,pVa = train(trainMats,y_train)

    #test
    x_test_cnt = []
    for i in range(len(x_test)):
        wordList = parser(x_test[i],stopWords)
        x_test_cnt.append(wordList)

    testMats = []
    for i in range(len(x_test_cnt)):
        testMats.append(createTrainMat(vocabList,x_test_cnt[i]))

    #score
    errorCount = 0
    for i in range(len(x_test_cnt)):
        if predictionResult(testMats[i],p0w,p1w,pVa) != y_test[i]:
            errorCount += 1
            print("分类错误的测试集：", x_test_cnt[i])
            print("lable",y_test[i])

    print("acc:",1 - float(errorCount) / len(testMats));
    return 1 - float(errorCount) / len(testMats)
#main
if __name__ == "__main__":
    accsarray = []
    count = 1
    step = 5
    for k in range(count):
        accs = []
        for i in range(10,50,step):
            Volum = i*100
            data = readcsv()
            acc = haveatry(Volum,data)
            accs.append(acc)
        accsarray.append(accs)
    accsarray = np.array(accsarray)
    meanaccs = accsarray.mean(axis = 0)

    x = list(range(10,50,step))
    x = np.array(x)
    x = x*100
    plt.plot(x,meanaccs)
    plt.xlabel('Volume')
    plt.ylabel('Accuracy')
    plt.ylim(0.88)
    plt.show()






