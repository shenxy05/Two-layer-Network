import numpy as np
import pickle


# 激活函数和最后分类的softmax层
def sigmoid(x):
    y = np.zeros(x.shape)
    if x.ndim == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i][j] >= 0:  # 防止计算exp时溢出
                    y[i][j] = 1 / (1 + np.exp(-x[i][j]))
                else:
                    y[i][j] = np.exp(x[i][j]) / (1 + np.exp(x[i][j]))
    else:
        for i in range(len(x)):
            if x[i] >= 0:
                y[i] = 1 / (1 + np.exp(-x[i]))
            else:
                y[i] = np.exp(x[i]) / (1 + np.exp(x[i]))
    return y


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)  # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))


class NeuralNetwork:
    # 参数初始化
    def __init__(self, input_size, hidden_size, output_size, lr, lamb_da):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        self.lr = lr
        self.lamb_da = lamb_da

    # 学习率衰减
    def lr_decay(self, ratio):
        self.lr *= ratio

    # 获取模型参数
    def get_params(self):
        return self.w1, self.w2, self.b1, self.b2

    # 前向预测
    def forward(self, x):
        h = sigmoid(np.dot(x, self.w1) + self.b1)  # 隐藏层神经元
        y_pred = softmax(np.dot(h, self.w2) + self.b2)  # 输出的one-hot预测
        return y_pred, h

    # 计算损失
    def loss(self, x, y):
        pred, _ = self.forward(x)
        # 使用样本的平均交叉熵损失
        return -np.log(pred[np.arange(y.shape[0]), y.argmax(axis=1)] + 1e-7).mean()

    # 反向传播计算梯度
    def backward(self, x, y):
        lamb_da = self.lamb_da
        w1, w2 = self.w1, self.w2
        gradient = {}
        pred, h = self.forward(x)
        gradient['b2'] = (pred - y).sum(axis=0)
        gradient['w2'] = np.dot(h.T, pred - y) + lamb_da * w2
        t = np.dot(pred - y, w2.T) * (1 - h) * h
        gradient['b1'] = t.sum(axis=0)
        gradient['w1'] = np.dot(x.T, t) + lamb_da * w1
        return gradient

    # 更新参数
    def update(self, x, y):
        lr = self.lr
        grad = self.backward(x, y)
        self.w1 -= lr * grad['w1']
        self.w2 -= lr * grad['w2']
        self.b1 -= lr * grad['b1']
        self.b2 -= lr * grad['b2']

    # 计算分类精度
    def accuracy(self, x, y):
        pred = np.argmax(self.forward(x)[0], axis=1)
        label = np.argmax(y, axis=1)
        accuracy = np.sum(pred == label) / float(y.shape[0])
        return accuracy

    # 保存模型
    def save(self, path):
        obj = pickle.dumps(self)
        with open(path, "wb") as f:
            f.write(obj)

    # 从文件中加载模型
    def load(path):
        obj = None
        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except IOError:
                print("IOError")
        return obj
