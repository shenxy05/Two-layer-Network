# 使用找到的最佳参数训练保存模型并可视化结果
import numpy as np
from model import NeuralNetwork
from best_args import load_data
import matplotlib.pyplot as plt


train_img, train_label, test_img, test_label = load_data(download=False,
                                                         flatten=True,
                                                         normalize=True)
network = NeuralNetwork(784, 150, 10, lr=1e-1, lamb_da=1e-5)  # 换成找到的最优参数
epoch = 30
train_loss = []
test_loss = []
test_acc = []

print(network.loss(test_img, test_label))
print("acc=", network.accuracy(test_img, test_label))
for e in range(epoch):
    iter = np.arange(len(train_label))
    np.random.shuffle(iter)  # 随机选取样本计算梯度
    batch_size = 100
    batch_index = np.arange(0, len(train_label) + 1, batch_size)
    count = 0
    for i in range(len(batch_index) - 1):
        index = iter[batch_index[i]:batch_index[i + 1]]
        network.update(train_img[index], train_label[index])
        if i % 10 == 0:
            train_loss.append(network.loss(train_img, train_label))
            test_loss.append(network.loss(test_img, test_label))
            test_acc.append(network.accuracy(test_img, test_label))
            print(e, ',', i)
    if e > epoch * 0.5:
        network.lr_decay(0.95)
    else:
        network.lr_decay(0.9)
    print(network.accuracy(test_img, test_label))
network.save('1.model')  # 保存模型


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 6), dpi=150)
x1 = np.arange(len(test_loss))
ax1 = plt.subplot(121)
plt.plot(x1, train_loss, label='train loss')
plt.plot(x1, test_loss, label='test loss')
plt.title('loss曲线')
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend()

x2 = np.arange(len(test_acc))
ax2 = plt.subplot(122)
plt.plot(x2, test_acc, label='测试集accuracy')
plt.title("accuracy曲线")
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.legend()
plt.savefig('loss-accuracy.png')
plt.show()
