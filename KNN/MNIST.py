import os
from PIL import Image
import numpy as np

#对图片二值化处理
def binaryzation(data):
    row = data.shape[1]
    col = data.shape[2]
    ret = np.empty(row * col)
    for i in range(row):
        for j in range(col):
            ret[i * col + j] = 0
            if (data[0][i][j] > 127):
                ret[i * col + j] = 1
    return ret

#随机读取文件，参数分别是要读取文件的总数（4000 为例），和划分（0.7为例）
def load_data(num,split):
    files = os.listdir("train_image") #只使用了train_image中的文件作为训练集和测试集，使用划分的方法
    file_num = len(files)
    idx = np.random.permutation(file_num) #从所有文件中随机选取图片进行训练和测试
    selected_file_num = num
    selected_files = []
    for i in range(selected_file_num):
        selected_files.append(files[idx[i]])

    img_mat = np.empty((selected_file_num, 1, 28, 28), dtype="float32")

    data = np.empty((selected_file_num, 28 * 28), dtype="float32")
    label = np.empty((selected_file_num), dtype="uint8")

    print
    "loading data..."
    for i in range(selected_file_num):
        print
        i, "/", selected_file_num, "\r",
        file_name = selected_files[i]
        file_path = os.path.join("train_image", file_name)
        img_mat[i] = Image.open(file_path)
        data[i] = binaryzation(img_mat[i]) #对图片二值化处理从而转换为一维向量
        temp = file_name.split('_')[1]
        label[i] = int(temp.split('.')[0]) #从文件名中获取lable
    print
    ""

    #做划分
    div_line = (int)(split * selected_file_num)
    idx = np.random.permutation(selected_file_num)
    train_idx, test_idx = idx[:div_line], idx[div_line:]
    train_data, test_data = data[train_idx], data[test_idx]
    train_label, test_label = label[train_idx], label[test_idx]

    return train_data, train_label, test_data, test_label


def KNN(test_vec, train_data, train_label, k):
    train_data_size = train_data.shape[0]
    dif_mat = np.tile(test_vec, (train_data_size, 1)) - train_data
    sqr_dif_mat = dif_mat ** 2
    sqr_dis = sqr_dif_mat.sum(axis=1)

    sorted_idx = sqr_dis.argsort()

    class_cnt = {}
    maxx = 0
    best_class = 0
    for i in range(k):
        tmp_class = train_label[sorted_idx[i]]
        tmp_cnt = class_cnt.get(tmp_class, 0) + 1
        class_cnt[tmp_class] = tmp_cnt
        if (tmp_cnt > maxx):
            maxx = tmp_cnt
            best_class = tmp_class
    return best_class


if __name__ == "__main__":
    np.random.seed(123456)
    train_data, train_label, test_data, test_label = load_data(40000,0.7) #total = 4000, split = 0.7
    tot = test_data.shape[0]
    err = 0
    print("testing...")
    for i in range(tot):
        print(i, "/", tot, "\r")
        best_class = KNN(test_data[i], train_data, train_label, 3) #默认k为3，可以修改
        if (best_class != test_label[i]):
            err = err + 1.0
    print("")
    print("accuracy")
    print(1 - err / tot)


#对测试进行记录
#total  split   acc     k
#1000   0.7     0.853   3
#2000   0.7     0.890   3
#4000   0.7     0.902   3
#8000   0.7     0.922   3
#4000   0.7     0.902   3
#4000   0.7     0.899   4
#4000   0.7     0.899   5
#40000  0.7     0.952   3