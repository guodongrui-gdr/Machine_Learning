import random
import time

from pylab import mpl
import numpy as np
from keras.datasets import mnist
from layers import FullConnectedLayer, SoftmaxLayer, ReluLayer
import matplotlib.pyplot as plt


class DNN(object):
    # batch_size: 每次训练样本个数
    # input_size: 输入样本大小
    # hidden1~n: 第n层隐藏层神经元数量
    # output_classes: 输出类数
    # learning_rate: 学习率
    # epoch: 训练次数
    def __init__(self, batch_size=30, input_size=784, hidden1=256, hidden2=128, hidden3=64, output_classes=10,
                 learning_rate=0.001, epoch=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output_class = output_classes
        self.learning_rate = learning_rate
        self.epoch = epoch

    # 导入数据
    def load_data(self):
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        train_x = np.reshape(train_x, [np.shape(train_x)[0], self.input_size])
        train_y = np.reshape(train_y, [np.shape(train_y)[0], 1])
        test_x = np.reshape(test_x, [np.shape(test_x)[0], self.input_size])
        test_y = np.reshape(test_y, [np.shape(test_y)[0], 1])
        self.input_size = np.shape(train_x)[1]
        self.train_data = np.append(train_x, train_y, axis=1)
        self.test_data = np.append(test_x, test_y, axis=1)

    # 将数据打乱,防止过拟合
    def shuffle_data(self):
        np.random.shuffle(self.train_data)

    # 构造神经网络
    def build_model(self):
        self.fc1 = FullConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReluLayer()
        self.fc2 = FullConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReluLayer()
        self.fc3 = FullConnectedLayer(self.hidden2, self.hidden3)
        self.relu3 = ReluLayer()
        self.fc4 = FullConnectedLayer(self.hidden3, self.output_class)
        self.softmax = SoftmaxLayer()
        self.layers_list = [self.fc1, self.fc2, self.fc3, self.fc4]

    def init_model(self):
        for layer in self.layers_list:
            layer.init_param()

    def load_model(self, param_dir):
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])
        self.fc4.load_param(params['w4'], params['b4'])

    def save_model(self, param_dir):
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        params['w4'], params['b4'] = self.fc4.save_param()
        np.save(param_dir, params, allow_pickle=True)

    # 前向传播
    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        h3 = self.relu3.forward(h3)
        h4 = self.fc4.forward(h3)
        prob = self.softmax.forward(h4)
        return prob

    # 反向传播
    def backward(self):
        dloss = self.softmax.backward()
        dh4 = self.fc4.backward(dloss)
        dh3 = self.relu3.backward(dh4)
        dh3 = self.fc3.backward(dh3)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self):
        for layer in self.layers_list:
            layer.update(self.learning_rate)

    # 训练模型
    def train(self):
        max_batch = np.shape(self.train_data)[0] / self.batch_size
        for idx_epoch in range(self.epoch):
            start_time = time.time()
            self.shuffle_data()
            loss = 0
            for idx_batch in range(int(max_batch)):
                batch_x = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_y = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                prob = self.forward(batch_x)
                loss = loss + self.softmax.loss(batch_y)
                self.backward()
                self.update()
            print('Epoch %d,loss: %.6f,used time: %.2f' % (idx_epoch, loss / max_batch, time.time() - start_time))
        self.save_model('1.npy')

    def test(self):
        pred_result = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0] / self.batch_size)):
            batch_x = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            prob = self.forward(batch_x)
            pred = np.argmax(prob, axis=1)
            pred_result[idx * self.batch_size:(idx + 1) * self.batch_size] = pred
        acc = np.mean(pred_result == self.test_data[:, -1])

        # 随机在测试集中挑选4个样本输出
        plt.figure(figsize=(8, 6))
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        # for i in range(4):
        #     x = self.test_data[random.randint(0, self.test_data.shape[0])][:-1]
        #     prob = self.forward(x)
        #     pred = np.argmax(prob, axis=1)
        #     x = x.reshape([28, 28])
        #     xlabel = "预测结果为" + str(pred)
        #     plt.subplot(2, 2, i + 1)
        #     plt.xlabel(xlabel)
        #     plt.imshow(x)
        pic = plt.imread(r'C:\Users\makabaka\Desktop\1.png')[:, :, 1].reshape([self.input_size, ])
        pic = 255 * (1 - pic)
        pic = pic.astype(np.uint8)
        prob = self.forward(pic)
        pred = np.argmax(prob, axis=1)
        pic = pic.reshape([28, 28])
        xlabel = "预测结果为" + str(pred)
        plt.xlabel(xlabel)
        plt.imshow(pic)
        plt.show()
        return acc


def build_dnn():
    h1, h2, h3, batch_size, epoch = 784, 512, 512, 30, 100
    model = DNN(batch_size=batch_size, hidden1=h1, hidden2=h2, hidden3=h3, epoch=epoch)
    model.load_data()
    model.build_model()
    model.init_model()
    model.train()
    return model


def load_dnn():
    h1, h2, h3, batch_size = 784, 512, 512, 30
    dir = '1.npy'
    model = DNN(batch_size=batch_size, hidden1=h1, hidden2=h2, hidden3=h3)
    model.load_data()
    model.build_model()
    model.init_model()
    model.load_model(dir)
    return model


model = load_dnn()
acc = model.test()
print('Accuracy: %.6f' % acc)
