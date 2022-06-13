import numpy as np


# 全连接层
class FullConnectedLayer:
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    # 前向传播
    def forward(self, input):
        self.input = input
        self.output = np.matmul(input, self.weight) + self.bias
        return self.output

    # 反向传播
    def backward(self, top_diff):
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    # 更新参数
    def update(self, learning_rate):
        self.weight = self.weight - learning_rate * self.d_weight
        self.bias = self.bias - learning_rate * self.d_bias

    # 加载参数
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    # 保存参数
    def save_param(self):
        return self.weight, self.bias


# ReLU激活函数层
class ReluLayer(object):
    # 前向传播
    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)
        return output

    # 反向传播
    def backward(self, top_diff):
        bottom_diff = top_diff
        bottom_diff[self.input < 0] = 0
        return bottom_diff


# softmax激活函数层
class SoftmaxLayer:
    # 前向传播
    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    # 损失函数
    def loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob+10**(-8)) * self.label_onehot) / self.batch_size
        return loss

    # 反向传播
    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff
