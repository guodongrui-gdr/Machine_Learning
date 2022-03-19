import time
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_digits  # 导入数据


def train(x0, x1):
    m0 = x0.mean(0)
    m1 = x1.mean(0)
    Sw = np.dot((x0 - m0).T, (x0 - m0)) + np.dot((x1 - m1).T, (x1 - m1))
    Sw = np.linalg.pinv(Sw)
    w = Sw.dot(m0 - m1)
    w0 = w.dot((m0 + m1) / 2)
    return w, w0


def test(x, w, w0):
    y = x.dot(w)
    flag = y > w0
    return flag


digits = load_digits()
data = digits.data
target = digits.target
classifier = []
for label in range(10):
    x0 = data[target == label, :]  # 将一个类作为正例
    x1 = data[target != label, :]  # 将剩下的类作为反例
    # 数据标准化
    scaler = preprocessing.StandardScaler().fit(np.concatenate((x0, x1),axis=0))  # preprocessing.StandardScaler().fit()函数用于计算数据的均值和方差,np.concatenate()函数用于将两组数据合并
    x0 = scaler.transform(x0)  # transform()函数用于将原数据替换为均值和方差
    x1 = scaler.transform(x1)
    # 训练模型
    w, w0 = train(x0, x1)
    classifier.append((w, w0))
# 测试模型
test_x = data[target == 0, :]
result=[]
for i in range(10):
    w = classifier[i][0]
    w0 = classifier[i][1]
    test_y0 = test(test_x, w, w0)

    # 计算准确率
    correct_rate = 0

    correct = np.sum(test_y0)
    correct_rate += correct / test_x.shape[0]
    result.append((i,correct_rate))
result.sort(key=lambda x:(x[1]),reverse=True)
print("classify result is:",result[0][0])