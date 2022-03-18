# coding: utf-8

# Linear Discriminant Analysis 2019.03.21
# Author:
#   Yuhao Wu
# Reference:
#   Zhihua Zhou. Machine learning[M]. Tsinghua University Press, 2016.
import numpy as np
import scipy.io as scio
from sklearn import preprocessing
from sklearn.datasets import load_digits # 导入数据



# Linear Discriminant Analysis train function
# Purpose:
#   Use train data x0 and x1 to calculate the weights, threshold of LDA model.
# Input：
#   x0：train_data in class0. [nsample * nfeature]
#   x1：train_data in class1. [nsample * nfeature]
# Output：
#   w：weights of LDA model. [nfeature]
#   w0: the threshold of the classification between class0 and class1. float
def LDA_train(x0, x1):
    mu0 = x0.mean(0)
    mu1 = x1.mean(0)

    # Calculate the Sw.
    #### Your Code ####
    Sw = Sw = np.dot((x0 - mu0).T, (x0 - mu0)) + np.dot((x1 - mu1).T, (x1 - mu1))
    #### Your Code ####
    Sw_ = np.linalg.pinv(Sw)

    # Calculate the w.
    #### Your Code ####
    w = Sw.dot(m0 - m1)
    #### Your Code ####

    w0 = w.dot((mu0+mu1)/2)
    return w, w0


# Linear Discriminant Analysis test function
# Purpose:
#   Use the LDA model to classify the test data.
# Input：
#   x：test data [nsample * nfeature]
#   w：weights of LDA model. [nfeature]
#   w0: the threshold of the classification model. float
# Output：
#   flag：result of LDA. bool[nsample]
#   (True: class0, False: class1)
def LDA_test(x, w, w0):
    y = x.dot(w)
    flag = y > w0
    return flag


# Load data
digits = load_digits()
data = digits.data
target = digits.target

#choose train sample:0--999
train_x=data[0:1000]
train_y=target[0:1000]

#choose test sample 1000--end
test_x=data[1000:]
test_y=target[1000:]

# We choose label 2 and 4 as the class0 and class1.
label0 = 8
label1 = 9
train_x0 = train_x[train_y==label0, :]
train_x1 = train_x[train_y==label1, :]

test_x0 = test_x[test_y==label0, :]
test_x1 = test_x[test_y==label1, :]

# Data normalize
scaler = preprocessing.StandardScaler().fit(np.concatenate((train_x0, train_x1), axis=0))
train_x0 = scaler.transform(train_x0)
train_x1 = scaler.transform(train_x1)
test_x0 = scaler.transform(test_x0)
test_x1 = scaler.transform(test_x1)


# Train
w, w0 = LDA_train(train_x0, train_x1)

# Test
test_y0 = LDA_test(test_x0, w, w0)
test_y1 = LDA_test(test_x1, w, w0)

# Calculate the correct rate of test data
correct = np.sum(test_y0) + np.sum(1 - test_y1)
correct_rate = correct / (test_x0.shape[0] + test_x1.shape[0])
print('Correct rate is: %.2f%%' % (correct_rate*100))
